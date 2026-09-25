"""Temperature loading regressions using a tiny local checkpoint; no training or downloads.

Ported from upstream ``laya/tests/test_temperature_loading.py`` and the inference half of
``laya/tests/test_calibration_persistence.py`` (v0.3.20). Upstream's checkpoint fixture is a torch
`DecisionModel` saved with safetensors; this port uses the tiny MLX checkpoint fixture.

Upstream #142: an entry that is not a number, or a value outside the clamp range, must not stop the
checkpoint from loading. It falls back to 1.0 / the clamp bound and warns once, naming every
affected entry. Upstream's warning starts with ``laya:``; this port's starts with ``laya-mlx:``.

The notebook-export half of ``test_calibration_persistence.py`` (upstream #139, clearing inherited
per-type buckets) is not ported: it executes the training notebook, which this port does not ship.
"""

import json
import warnings
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from laya_mlx import load

QUESTIONS = {
    "choice": {"type": "choice", "instructions": "Pick one", "criteria": ["a", "b"]},
    "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "mid", "high"]},
    "noul": {"type": "noul", "instructions": "True?"},
}


def _load_config(tiny_checkpoint, **temperatures):
    cfg_path = tiny_checkpoint / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("temperature", None)
    cfg.pop("temperature_by_options", None)
    cfg.update(temperatures)
    cfg_path.write_text(json.dumps(cfg))
    written = cfg_path.read_bytes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with patch(
            "laya_mlx.agent.snapshot_download", side_effect=AssertionError("unexpected download")
        ):
            agent = load(str(tiny_checkpoint), device="cpu")
    assert agent.device == mx.cpu
    # the checkpoint's own config is never rewritten by loading
    assert cfg_path.read_bytes() == written
    assert json.dumps(agent.cfg) == json.dumps(cfg)
    assert json.dumps(agent.temperature_raw) == json.dumps(cfg.get("temperature", [1.0] * 3))
    assert json.dumps(agent.temperature_by_options_raw) == json.dumps(
        cfg.get("temperature_by_options", {})
    )
    messages = [str(w.message) for w in caught if str(w.message).startswith("laya-mlx:")]
    assert all(
        issubclass(w.category, RuntimeWarning)
        for w in caught
        if str(w.message).startswith("laya-mlx:")
    )
    return agent, messages


def _assert_predictions(agent, choice, score, noul):
    # Loading is real; fixed logits make the expected scaling independent of random weights.
    logits = np.array([[0.0, 1.0, -1e4], [0.0, 1.0, 2.0], [0.0, 1.0, -1e4]], dtype=np.float32)
    act = np.zeros((3, 2), dtype=np.float32)
    with patch.object(agent, "forward", return_value=(mx.array(logits), mx.array(act))):
        answers = agent.predict("hello", QUESTIONS)["answers"]
    for name, temperature, k in (("choice", choice, 2), ("score", score, 3), ("noul", noul, 2)):
        z = np.arange(k, dtype=np.float64) / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        if name == "noul":
            assert answers[name]["noul"] == pytest.approx(p[1], abs=1e-4)
        else:
            actual = list(answers[name]["probabilities"].values())
            assert len(actual) == k
            for got, want in zip(actual, p.tolist()):
                assert got == pytest.approx(want, abs=1e-4)


@pytest.mark.parametrize("value", [None, "invalid", "", [], {}])
def test_invalid_type_entries_use_neutral_fallback_and_warn(tiny_checkpoint, value):
    agent, messages = _load_config(tiny_checkpoint, temperature=[value, 2.0, 3.0])
    assert agent.temperature == [1.0, 2.0, 3.0]
    assert len(messages) == 1
    assert "temperature[0]" in messages[0]
    assert repr(value) in messages[0]
    assert "uncalibrated" in messages[0]
    _assert_predictions(agent, choice=1.0, score=2.0, noul=3.0)


@pytest.mark.parametrize("value", [None, "invalid", "", [], {}])
def test_invalid_bucket_entries_use_neutral_fallback_and_keep_precedence(tiny_checkpoint, value):
    agent, messages = _load_config(
        tiny_checkpoint, temperature=[2.0, 3.0, 4.0], temperature_by_options={"choice:2": value}
    )
    assert agent.temperature_by_options == {"choice:2": 1.0}
    assert len(messages) == 1
    assert "choice:2" in messages[0]
    assert repr(value) in messages[0]
    _assert_predictions(agent, choice=1.0, score=3.0, noul=4.0)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", "-Infinity"]
)
def test_nonfinite_values_keep_neutral_fallback_and_warn(tiny_checkpoint, value):
    agent, messages = _load_config(
        tiny_checkpoint, temperature=[value, 2.0, 3.0], temperature_by_options={"noul:2": value}
    )
    assert agent.temperature == [1.0, 2.0, 3.0]
    assert agent.temperature_by_options == {"noul:2": 1.0}
    assert len(messages) == 1
    assert "temperature[0]" in messages[0]
    assert "noul:2" in messages[0]
    _assert_predictions(agent, choice=1.0, score=2.0, noul=1.0)


def test_out_of_range_values_keep_existing_clamps(tiny_checkpoint):
    agent, messages = _load_config(
        tiny_checkpoint,
        temperature=[0.1, 10, -1],
        temperature_by_options={"choice:2": 9, "score:3-5": 0},
    )
    assert agent.temperature == [0.5, 5.0, 0.5]
    assert agent.temperature_by_options == {"choice:2": 5.0, "score:3-5": 0.5}
    assert len(messages) == 1
    for entry in (
        "temperature[0]",
        "temperature[1]",
        "temperature[2]",
        "choice:2",
        "score:3-5",
    ):
        assert entry in messages[0]
    _assert_predictions(agent, choice=5.0, score=0.5, noul=0.5)


def test_valid_numbers_and_numeric_strings_do_not_warn(tiny_checkpoint):
    agent, messages = _load_config(
        tiny_checkpoint,
        temperature=[0.5, "2", 5],
        temperature_by_options={"choice:2": "1.5", "score:3-5": 2.5, "noul:2": "5.0"},
    )
    assert agent.temperature == [0.5, 2.0, 5.0]
    assert agent.temperature_by_options == {"choice:2": 1.5, "score:3-5": 2.5, "noul:2": 5.0}
    assert messages == []
    _assert_predictions(agent, choice=1.5, score=2.5, noul=5.0)


def test_missing_bucket_uses_corresponding_type(tiny_checkpoint):
    agent, messages = _load_config(
        tiny_checkpoint, temperature=[2.0, 3.0, 4.0], temperature_by_options={"choice:2": 1.5}
    )
    assert messages == []
    _assert_predictions(agent, choice=1.5, score=3.0, noul=4.0)


def test_missing_temperature_fields_keep_defaults(tiny_checkpoint):
    agent, messages = _load_config(tiny_checkpoint)
    assert agent.temperature == [1.0, 1.0, 1.0]
    assert agent.temperature_by_options == {}
    assert messages == []
    _assert_predictions(agent, choice=1.0, score=1.0, noul=1.0)


def test_existing_bucket_calibration_keeps_precedence(tiny_checkpoint):
    # from upstream test_calibration_persistence: a fitted bucket still wins over its type's
    # temperature, and the sizes map to the documented bucket edges.
    buckets = {"2": 0.75, "3-5": 1.25, "6-10": 1.5, "11+": 1.75}
    by_options = {}
    for qtype in ("choice", "score", "noul"):
        for bucket, value in buckets.items():
            by_options["%s:%s" % (qtype, bucket)] = value
    agent, messages = _load_config(
        tiny_checkpoint, temperature=[2.0, 3.0, 4.0], temperature_by_options=by_options
    )
    assert messages == []
    assert agent.temperature_by_options["choice:2"] == 0.75
    assert agent.temperature_by_options["score:6-10"] == 1.5
    assert agent.temperature_by_options["noul:11+"] == 1.75


def test_loading_does_not_rewrite_tokenizer_config(tiny_checkpoint):
    # upstream #107/#267 harden `_fix_tokenizer_config`; this port only ever reads the file, so
    # the guarantee to assert is the observable one: loading leaves it byte-identical.
    path = tiny_checkpoint / "tokenizer" / "tokenizer_config.json"
    before = path.read_bytes()
    load(str(tiny_checkpoint), device="cpu")
    assert path.read_bytes() == before
