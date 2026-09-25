"""Public MLX inference runtime; prompt and result formats follow upstream Laya."""

import json
import os
import warnings
from pathlib import Path, PurePosixPath

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

from .common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    _resolve_noul_labels,
    answer_confidence,
    build_sequence,
    clamp_temperature,
    confidence_from_probs,
    render_options,
    serialize_state,
    temp_bucket,
)
from .model import DecisionModel, EncoderConfig, sanitize_weights
from .prepared import PrefixCache
from .tokenizer import Tokenizer

DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def resolve_model(model_id_or_path, *, token=None, subfolder=None, revision=None):
    if subfolder:
        part = PurePosixPath(subfolder)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError("subfolder must be a relative path inside the model repository")
    path = Path(model_id_or_path).expanduser()
    if not path.exists():
        value = str(model_id_or_path)
        if value.startswith(("/", "./", "../", "~")) or isinstance(model_id_or_path, Path):
            raise FileNotFoundError(f"Local model directory does not exist: {value}")
        prefix = subfolder.rstrip("/") + "/" if subfolder else ""
        patterns = [
            prefix + name
            for name in (
                "model.safetensors",
                "rl_agent_config.json",
                "encoder/config.json",
                "tokenizer/*",
                "mlx_config.json",
            )
        ]
        path = Path(
            snapshot_download(
                value,
                # An empty HF_TOKEN must mean "no token": passing "" through would make the Hub
                # send an empty Bearer header (upstream #264).
                token=token or os.environ.get("HF_TOKEN") or None,
                revision=revision,
                allow_patterns=patterns,
            )
        )
    if subfolder:
        path /= subfolder
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Not a complete Laya checkpoint: {path / name} is missing")
    return path


def collate_items(items, pad_id, *, pad_to_multiple=None, max_length=None):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    n, length = len(items), max(len(item["ids"]) for item in items)
    if pad_to_multiple:
        length = ((length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        if max_length is not None:
            length = min(length, max_length)
    count = max(2, max(len(item["markers"]) for item in items))
    batch = {
        "input_ids": np.full((n, length), pad_id, dtype=np.int32),
        "attention_mask": np.zeros((n, length), dtype=np.bool_),
        "marker_pos": np.zeros((n, count), dtype=np.int32),
        "marker_mask": np.zeros((n, count), dtype=np.bool_),
        "qtype": np.array([item["qtype"] for item in items], dtype=np.int32),
    }
    for i, item in enumerate(items):
        length, count = len(item["ids"]), len(item["markers"])
        batch["input_ids"][i, :length] = item["ids"]
        batch["attention_mask"][i, :length] = True
        batch["marker_pos"][i, :count] = item["markers"]
        batch["marker_mask"][i, :count] = True
    return batch


class Agent:
    def __init__(
        self,
        model_id_or_path="convaiinnovations/laya",
        device=None,
        token=None,
        subfolder=None,
        *,
        dtype="float16",
        revision=None,
        batch_size=16,
        compile=False,
        pad_to_multiple=None,
        cache_prompts=False,
        lang_temperatures=None,
    ):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}")
        if device not in (None, "gpu", "metal", "cpu"):
            raise ValueError("MLX device must be 'gpu', 'metal', or 'cpu'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.device = (
            mx.default_device() if device is None else (mx.cpu if device == "cpu" else mx.gpu)
        )
        self.dtype = DTYPES[dtype]
        self.batch_size = batch_size
        if pad_to_multiple is not None and (
            not isinstance(pad_to_multiple, int)
            or isinstance(pad_to_multiple, bool)
            or pad_to_multiple < 1
        ):
            raise ValueError("pad_to_multiple must be a positive integer or None")
        self.pad_to_multiple = pad_to_multiple
        self._prefix_cache = PrefixCache() if cache_prompts else None
        self.model_id = str(model_id_or_path)
        self.revision = revision
        self.model_dir = resolve_model(
            model_id_or_path, token=token, subfolder=subfolder, revision=revision
        )
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        self.encoder_cfg = json.loads((self.model_dir / "encoder/config.json").read_text())
        if "encoder" not in self.cfg or "head_layers" not in self.cfg:
            raise ValueError("Laya config must specify encoder and head_layers")
        enc_cfg = EncoderConfig.from_dict(self.encoder_cfg)
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        if not 4 < head_max_len < max_len <= enc_cfg.max_position_embeddings:
            raise ValueError("Expected 4 < head_max_len < max_len <= max_position_embeddings")
        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        if len(self.temperature_raw) != 3:
            raise ValueError("Calibration temperature must be a list of 3 values")
        # Keep what the checkpoint shipped for inspection, but only ever apply clampable values:
        # an entry that is not a number falls back to 1.0 and an out-of-range one is clamped
        # (upstream #142). Upstream used to be able to reject the whole checkpoint here.
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {
            k: clamp_temperature(v) for k, v in self.temperature_by_options_raw.items()
        }
        # Per-language calibration overrides (upstream #258), keyed by the primary subtag.
        self.lang_temperatures = {}
        for name, override in (lang_temperatures or {}).items():
            norm_name = name.split("-")[0].lower()
            raw = override.get("temperature", self.temperature_raw)
            if len(raw) != 3:
                raise ValueError(
                    "Language override %r temperature must be a list of 3 floats" % name
                )
            raw_buckets = override.get("temperature_by_options", {})
            self.lang_temperatures[norm_name] = {
                "temperature": [clamp_temperature(t) for t in raw],
                "temperature_by_options": {k: clamp_temperature(v) for k, v in raw_buckets.items()},
            }
        entries = [
            (k, v, self.temperature_by_options[k])
            for k, v in self.temperature_by_options_raw.items()
        ]
        entries += [
            ("temperature[%d]" % i, t, self.temperature[i])
            for i, t in enumerate(self.temperature_raw)
        ]
        rejected = []
        for name, raw, applied in entries:
            try:
                if float(raw) == applied:
                    continue
            except (TypeError, ValueError):
                # Invalid entries already have a neutral fallback; diagnostics must not
                # repeat the failed conversion or prevent the checkpoint from loading.
                pass
            rejected.append("%s=%r -> %g" % (name, raw, applied))
        if rejected:
            warnings.warn(
                "laya-mlx: this checkpoint ships invalid temperatures or values outside "
                "[%g, %g] which would distort confidence; clamping %s. Treat confidence from "
                "the affected entries as uncalibrated." % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
                RuntimeWarning,
                stacklevel=2,
            )
        self.tok = Tokenizer(self.model_dir / "tokenizer")
        with mx.stream(self.device):
            self.model = DecisionModel(enc_cfg, self.cfg)
            weights = sanitize_weights(mx.load(str(self.model_dir / "model.safetensors")))
            weights = {k: v.astype(self.dtype) for k, v in weights.items()}
            self.model.load_weights(list(weights.items()), strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())
        # Frozen inference instance: changing weights or module structure requires a new Agent.
        self._inference = mx.compile(self.model) if compile else self.model

    @staticmethod
    def _check_question(qid, qdef):
        """Reject a question that cannot be answered, naming it and what to fix.

        `render_options` reads `criteria` in the shape the question's type expects and the decision
        head needs at least one option, so a malformed definition used to surface as an exception
        that named neither the question nor the problem. Upstream #183/#156/#249.
        """
        if not isinstance(qdef, dict):
            raise ValueError(
                "question %r: definition must be a dict, got %s" % (qid, type(qdef).__name__)
            )
        t = qdef.get("type")
        if t not in QTYPES:
            raise ValueError(
                "question %r: unknown type %r; use one of %s" % (qid, t, sorted(QTYPES))
            )
        if "instructions" not in qdef:
            raise ValueError(
                "question %r: no 'instructions'; add the text the model should answer" % (qid,)
            )
        crit = qdef.get("criteria")
        if t == "choice":
            if not isinstance(crit, (dict, list)):
                raise ValueError(
                    "question %r: a choice question takes 'criteria' as a dict of "
                    "label -> description, or a list of labels" % (qid,)
                )
            if not crit:
                raise ValueError(
                    "question %r: a choice question needs at least one criterion" % (qid,)
                )
        elif t == "score":
            if not isinstance(crit, list):
                raise ValueError(
                    "question %r: a score question takes 'criteria' as a list of level "
                    "descriptions, index 0 first" % (qid,)
                )
            if not crit:
                raise ValueError("question %r: a score question needs at least one level" % (qid,))
        elif crit is not None and not isinstance(crit, dict):
            raise ValueError(
                "question %r: a noul question takes 'criteria' as a dict with optional "
                "'true'/'false' descriptions, or omits it" % (qid,)
            )
        elif isinstance(crit, dict):
            # `render_options` reads these two descriptions out by name -- `crit.get("false")` and
            # `crit.get("true")` -- so a dict keyed any other way is not a noul description at all.
            # It used to be substituted with the default pair without a word, so a caller saw their
            # descriptions accepted and never reach the model (#156). `labels` has rejected the
            # same mistake since #163; this is the same rule on the other parameter, and a noul is
            # a boolean question either way, so those are the only two keys it can have.
            keys = {str(k).lower() for k in crit}
            if not keys <= {"true", "false"}:
                raise ValueError(
                    "question %r: a noul question takes 'criteria' keyed only 'true'/'false' "
                    "(either or both, and omitted is fine), got %s. Those keys are the option "
                    "texts the model reads; any other key was silently dropped and replaced with "
                    "the defaults. If you want the answer worded differently, keep 'criteria' "
                    "keyed 'true'/'false' and set 'labels' instead." % (qid, sorted(keys))
                )
        if "labels" in qdef:
            if t != "noul":
                raise ValueError(
                    "question %r: 'labels' is only supported for noul questions" % (qid,)
                )
            try:
                _resolve_noul_labels(qdef["labels"])
            except ValueError as e:
                raise ValueError("question %r: %s" % (qid, e)) from e

    @staticmethod
    def _to_internal(qdef):
        if not isinstance(qdef, dict):
            raise ValueError("Each question must be a dictionary")
        kind = qdef.get("type")
        if kind not in QTYPES:
            raise ValueError(f"Unknown question type {kind!r}; expected choice, score, or noul")
        if "instructions" not in qdef:
            raise ValueError("Question is missing instructions")
        criteria = qdef.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(c, str) for c in criteria):
                    raise ValueError("Choice labels must be strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty dictionary or list")
            if not all(isinstance(k, str) for k in criteria):
                raise ValueError("Choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score criteria must be a nonempty list")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a dictionary with false/true descriptions")
        elif isinstance(criteria, dict):
            # Boolean literal keys are how JSON `true`/`false` arrive, and uppercase spellings are
            # a reasonable way to write them; both normalise to the string keys `render_options`
            # reads by name (upstream #146).
            criteria = {str(k).lower(): v for k, v in criteria.items()}
        instructions = qdef["instructions"]
        if not isinstance(instructions, str):
            # `ensure_ascii=False`, matching `serialize_state` and `render_criterion`: the
            # default escaped non-ASCII to literal `\uXXXX`, which the tokenizer then read as
            # escape text rather than characters (upstream #228).
            instructions = json.dumps(instructions, ensure_ascii=False)
        q = {"t": kind, "ins": instructions, "crit": criteria}
        # `labels` only rewords the noul options; the internal keys stay false/true (#163).
        if "labels" in qdef:
            q["labels"] = qdef["labels"]
        return q

    def prepare(self, state, questions):
        """Construct upstream-compatible CPU inputs, useful for parity and profiling."""
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        # Validate every question by name before anything is rendered or tokenized.
        for qid, definition in questions.items():
            self._check_question(qid, definition)
        if self._prefix_cache is not None:
            return self._prefix_cache.prepare(self, state, questions)
        if not questions:
            return [], []
        tok = self.tok
        max_len, head_max_len = self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192)
        # A chronological conversation list is serialized newest-last, so the default
        # right-truncation would silently drop the newest turn. Truncate from the left for lists
        # so the most recent intent survives; strings and dicts are unchanged (upstream #224).
        truncate_left = isinstance(state, list)
        # The state ids are identical for every question, so serialize and tokenize the shared
        # state once here and let `build_sequence` slice it per question (upstream #109).
        state_ids = tok(
            serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False
        )["input_ids"]
        items, internal = [], []
        for qid, definition in questions.items():
            q = self._to_internal(definition)
            ids, markers = build_sequence(
                tok,
                state,
                q,
                max_len,
                head_max_len,
                truncate_left=truncate_left,
                state_ids=state_ids,
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal

    def forward(self, batch):
        """Run one prepared batch and return evaluated MLX logits on this agent's device."""
        with mx.stream(self.device):
            tensors = {k: mx.array(v) for k, v in batch.items()}
            result = self._inference(**tensors)
            mx.eval(result)
        return result

    def _decode_answers(self, logits, act, items, ids, internal, offset, lang=None):
        """Turn one batch's logit rows (starting at `offset`) into typed answers.

        `internal` is either the list aligned with `ids`, or a dict keyed by question id.
        """
        answers = {}
        for j, qid in enumerate(ids):
            r = offset + j
            q = internal[qid] if isinstance(internal, dict) else internal[j]
            k = len(items[j]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            if lang:
                norm_lang = lang.split("-")[0].lower()
                if norm_lang in self.lang_temperatures:
                    l_cfg = self.lang_temperatures[norm_lang]
                    t_scale = l_cfg["temperature_by_options"].get(
                        temp_bucket(qt, k), l_cfg["temperature"][qt]
                    )
            z = logits[r, :k] / t_scale
            p = np.exp(z - z.max())
            p /= p.sum()

            # `confidence` means one thing for `noul` (max(p)) and another for `choice` and
            # `score` (normalized entropy), and only the first is the quantity temperature
            # scaling fits and ECE measures. Rather than change one underneath existing
            # callers, report both: `answer_confidence` is the calibrated one, on every
            # question type, so a caller can gate across types on a single number (#126).
            ans_conf = round(answer_confidence(p, k), 4)
            action = {"act_probability": round(float(act[r, 0]), 4)}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {key: round(float(v), 4) for key, v in zip(keys, p)},
                    # entropy is only meaningful for the types that report it (#293)
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "answer_confidence": ans_conf,
                    "action": action,
                }
            elif q["t"] == "score":
                answers[qid] = {
                    "type": "score",
                    "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): value for i, value in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "answer_confidence": ans_conf,
                    "action": action,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    # identical here: over two options max(p_true, 1 - p_true) is max(p)
                    "answer_confidence": ans_conf,
                    "action": action,
                }
        return answers

    def system_one(self, state, questions, lang=None):
        """Evaluate typed questions across one state in a single, batched forward pass.

        Returns `{"model", "answers", "usage"}`; each answer carries `type`, `confidence`,
        the calibrated `answer_confidence`, the typed value, and `action`. Empty questions
        return empty answers and zero usage without tokenizing or running inference (#144).
        """
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        if not questions:
            return {
                "model": "laya-rl-agent",
                "answers": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        items, internal = self.prepare(state, questions)
        answers = {}
        question_ids = list(questions)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            qids = question_ids[start : start + len(chunk)]
            batch = collate_items(
                chunk,
                self.tok.pad_token_id,
                pad_to_multiple=self.pad_to_multiple,
                max_length=self.cfg.get("max_len", 512),
            )
            logits, act = self.forward(batch)
            logits, act = np.asarray(logits), np.asarray(act)
            if not np.isfinite(logits).all() or not np.isfinite(act).all():
                raise FloatingPointError("Non-finite model outputs; retry with dtype='float32'")
            act = np.exp(act - act.max(axis=-1, keepdims=True))
            act /= act.sum(axis=-1, keepdims=True)
            answers.update(
                self._decode_answers(
                    logits, act, chunk, qids, internal[start : start + len(chunk)], 0, lang=lang
                )
            )
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": sum(len(item["ids"]) for item in items), "output_tokens": 0},
        }

    predict = system_one


RLAgent = Agent


def load(
    model_id_or_path="convaiinnovations/laya", device=None, token=None, subfolder=None, **kwargs
):
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder, **kwargs)
