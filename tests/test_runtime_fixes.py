"""Runtime regressions for the state-reuse, decoding, and calibration changes.

Ported from upstream ``laya/tests/test_runtime_fixes.py`` and the decoding section of
``laya/tests/test_batch.py`` (v0.3.20). Upstream's file also covers autocast, MPS row gating, and a
CUDA OOM fallback; none of those exist in the MLX runtime, which has no `torch.autocast`, no MPS
device type, and no device fallback path, so those sections are not ported.
"""

from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

import laya_mlx.agent as agent_mod
from laya_mlx import Agent
from laya_mlx.agent import build_sequence
from laya_mlx.common import serialize_state


class FakeTok:
    """Just enough of a tokenizer for `build_sequence` / `system_one`, with deterministic ids."""

    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        self.calls.append(text)
        ids = [10 + (len(w) % 90) for w in text.split() if w]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


class FakeModel:
    """A stand-in for the MLX decision model: one logit per marker row, no weights needed."""

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits = mx.where(
            marker_mask,
            mx.ones(marker_mask.shape, dtype=mx.float32),
            mx.full(marker_mask.shape, -1e4, dtype=mx.float32),
        )
        return logits, mx.zeros((input_ids.shape[0], 2), dtype=mx.float32)


def _bare_agent(model, tok=None):
    agent = object.__new__(Agent)
    agent.device = mx.cpu
    agent.dtype = mx.float32
    agent.cfg = {"max_len": 64, "head_max_len": 32}
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.lang_temperatures = {}
    agent.batch_size = 16
    agent.pad_to_multiple = None
    agent._prefix_cache = None
    agent._inference = model
    agent.tok = tok or FakeTok()
    return agent


# ------------------------------------------------------------------ state reused once


STATE = {"subject": "Duplicate charge", "body": "x" * 400}
QUESTION = {"t": "choice", "ins": "pick", "crit": {"a": "x", "b": "y"}}


def test_state_ids_are_equivalent_to_inline_tokenization():
    tok_ref = FakeTok()
    seq_ref, markers_ref = build_sequence(tok_ref, STATE, QUESTION, 64, 32)

    tok_shared = FakeTok()
    state_ids = tok_shared(serialize_state(STATE), add_special_tokens=False)["input_ids"]
    seq_shared, markers_shared = build_sequence(
        tok_shared, STATE, QUESTION, 64, 32, state_ids=state_ids
    )
    assert seq_shared == seq_ref
    assert markers_shared == markers_ref
    assert tok_shared.calls.count(serialize_state(STATE)) == 1


def test_system_one_tokenizes_the_state_once_for_two_questions():
    tok = FakeTok()
    agent = _bare_agent(FakeModel(), tok=tok)
    out = agent.system_one(
        "the customer was charged twice",
        {
            "department": {
                "type": "choice",
                "instructions": "which?",
                "criteria": {"billing": "x", "technical": "y"},
            },
            "urgent": {"type": "noul", "instructions": "is it urgent?"},
        },
    )
    state_text = serialize_state("the customer was charged twice").replace(tok.mask_token, " ")
    assert tok.calls.count(state_text) == 1
    assert sorted(out["answers"]) == ["department", "urgent"]


def test_option_segments_are_capped_for_long_descriptions():
    # Upstream #109 caps each option at the tokenizer (truncation=True, max_length=48). The port's
    # Tokenizer (tokenizer.py) has no truncation argument, so the equivalent first-48-token slice
    # is taken instead; the ids are identical, which is what this asserts.
    long_q = {"t": "choice", "ins": "pick", "crit": {"a": "x" * 400, "b": "y" * 400}}
    tok = FakeTok()
    _, markers = build_sequence(tok, "state", long_q, 300, 200)
    assert len(markers) == 2
    assert all(markers[i + 1] - markers[i] <= 49 for i in range(len(markers) - 1))


# ------------------------------------------------------------------ answer decoding


def test_decode_reports_calibrated_confidence_and_skips_noul_entropy():
    """Upstream #126 / #293: `answer_confidence` is max(p) on every type; noul needs no entropy."""
    decoder = object.__new__(Agent)
    decoder.temperature = {0: 1.0, 1: 2.0, 2: 3.0}
    decoder.temperature_by_options = {"choice:2": 0.5}
    decode_ids = ["pick", "level", "flag"]
    decode_internal = {
        "pick": {"t": "choice", "crit": {"left": "left", "right": "right"}},
        "level": {"t": "score", "crit": ["low", "high"]},
        "flag": {"t": "noul", "crit": None},
    }
    decode_items = [{"markers": [0, 1]} for _ in decode_ids]
    decode_logits = np.array(
        [
            [99.0, -99.0],  # preceding state: must not be decoded
            np.log([0.25, 0.75]) * 0.5,
            np.log([0.25, 0.75]) * 2.0,
            np.log([0.8, 0.2]) * 3.0,
        ]
    )
    decode_act = np.array([[0.0, 1.0], [0.125, 0.875], [0.25, 0.75], [0.75, 0.25]])
    with patch.object(
        agent_mod, "confidence_from_probs", wraps=agent_mod.confidence_from_probs
    ) as entropy:
        decoded = decoder._decode_answers(
            decode_logits, decode_act, decode_items, decode_ids, decode_internal, 1
        )
        assert decoded == {
            "pick": {
                "type": "choice",
                "choice": "right",
                "probabilities": {"left": 0.25, "right": 0.75},
                "confidence": 0.1887,
                "answer_confidence": 0.75,
                "action": {"act_probability": 0.125},
            },
            "level": {
                "type": "score",
                "score": 0.75,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.25, "1": 0.75},
                "confidence": 0.1887,
                "answer_confidence": 0.75,
                "action": {"act_probability": 0.25},
            },
            "flag": {
                "type": "noul",
                "noul": 0.2,
                "confidence": 0.8,
                "answer_confidence": 0.8,
                "action": {"act_probability": 0.75},
            },
        }
        assert entropy.call_count == 2
        entropy.reset_mock()
        for true_probability in (0.1, 0.5, 0.9):
            logits = np.log([[1.0 - true_probability, true_probability]]) * 3.0
            result = decoder._decode_answers(
                logits, decode_act[-1:], decode_items[-1:], ["flag"], decode_internal, 0
            )
            assert result["flag"] == {
                "type": "noul",
                "noul": true_probability,
                "confidence": max(true_probability, 1.0 - true_probability),
                "answer_confidence": max(true_probability, 1.0 - true_probability),
                "action": {"act_probability": 0.75},
            }
        assert entropy.call_count == 0


def test_decode_accepts_internal_keyed_by_question_id():
    decoder = object.__new__(Agent)
    decoder.temperature = [1.0, 1.0, 1.0]
    decoder.temperature_by_options = {}
    decoded = decoder._decode_answers(
        np.zeros((1, 2)),
        np.array([[0.5, 0.5]]),
        [{"markers": [0, 1]}],
        ["flag"],
        {"flag": {"t": "noul", "crit": None}},
        0,
    )
    assert decoded["flag"]["type"] == "noul"


# ------------------------------------------------------------------ language temperatures


def test_language_temperatures_override_calibration(tiny_checkpoint):
    agent = Agent(
        tiny_checkpoint,
        dtype="float32",
        lang_temperatures={"ZH-CN": {"temperature": [5.0, 5.0, 5.0]}},
    )
    # the language key is normalised to its primary subtag
    assert set(agent.lang_temperatures) == {"zh"}
    questions = {"noul": {"type": "noul", "instructions": "x"}}
    logits = np.array([[0.0, 1.0]], dtype=np.float32)
    with patch.object(agent, "forward", return_value=(mx.array(logits), mx.zeros((1, 2)))):
        default = agent.predict("hello", questions)["answers"]["noul"]["noul"]
        chinese = agent.predict("hello", questions, lang="zh-CN")["answers"]["noul"]["noul"]
        other = agent.predict("hello", questions, lang="fr")["answers"]["noul"]["noul"]
    # fixture temperature[2] is 2.0; the override is 5.0 and softens the answer
    assert default == pytest.approx(1.0 / (1.0 + np.exp(-0.5)), abs=1e-4)
    assert chinese == pytest.approx(1.0 / (1.0 + np.exp(-0.2)), abs=1e-4)
    assert chinese < default
    assert other == default


def test_language_temperature_must_be_three_values(tiny_checkpoint):
    with pytest.raises(ValueError, match="3 floats"):
        Agent(tiny_checkpoint, dtype="float32", lang_temperatures={"zh": {"temperature": [1.0]}})


def test_system_one_passes_lang_to_the_decoder(tiny_checkpoint):
    agent = Agent(tiny_checkpoint, dtype="float32")
    with patch.object(agent, "_decode_answers", wraps=agent._decode_answers) as decode:
        agent.system_one("hello", {"q": {"type": "noul", "instructions": "test"}}, lang="zh-CN")
    assert decode.call_args.kwargs["lang"] == "zh-CN"


# ------------------------------------------------------------------ download token


def test_empty_hf_token_is_no_token(monkeypatch):
    captured = {}

    def fake_snapshot(repo_id, **kwargs):
        captured.update(kwargs)
        return "/nonexistent/laya-mlx-test"

    monkeypatch.setattr(agent_mod, "snapshot_download", fake_snapshot)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(FileNotFoundError):
        agent_mod.resolve_model("test/nonexistent-repo", token="")
    assert captured["token"] is None

    monkeypatch.setenv("HF_TOKEN", "env-token")
    with pytest.raises(FileNotFoundError):
        agent_mod.resolve_model("test/nonexistent-repo", token="")
    assert captured["token"] == "env-token"

    with pytest.raises(FileNotFoundError):
        agent_mod.resolve_model("test/nonexistent-repo", token="explicit")
    assert captured["token"] == "explicit"
