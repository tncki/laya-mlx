"""The command-line contract other programs rely on.

``laya-decide`` is meant to be called by an agent harness or a shell script, so its argument
handling, its exit codes and its stdout shape are the interface. These tests stub the model so the
contract can be checked without loading a checkpoint.
"""

import json
import sys

import pytest

from laya_mlx import cli

FAKE_RESULT = {
    "model": "fake",
    "answers": {
        "intent": {"type": "choice", "choice": "refund", "confidence": 0.9},
        "is_urgent": {"type": "noul", "noul": 0.25},
    },
    "usage": {"input_tokens": 7, "output_tokens": 0},
}


class FakeAgent:
    """Stands in for Agent: records how it was constructed and what it was asked."""

    calls = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        FakeAgent.calls.append({"model": model, "kwargs": kwargs})

    def predict(self, state, questions):
        FakeAgent.calls[-1]["state"] = state
        FakeAgent.calls[-1]["questions"] = questions
        return FAKE_RESULT


class FakeStdin:
    def __init__(self, text=""):
        self.text = text

    def read(self):
        return self.text


@pytest.fixture(autouse=True)
def fake_agent(monkeypatch):
    FakeAgent.calls = []
    monkeypatch.setattr(cli, "Agent", FakeAgent)
    # decide_main falls back to stdin when given no positional text; pytest's captured stdin is
    # not readable, so give every test an empty one unless it supplies its own.
    monkeypatch.setattr(sys, "stdin", FakeStdin())
    return FakeAgent


def test_decide_reads_text_from_stdin_when_no_positional_text(monkeypatch, capsys):
    # stdin is the safe path for text containing quotes, newlines or non-ASCII characters.
    monkeypatch.setattr(sys, "stdin", FakeStdin("发票4411被重复扣款"))
    assert cli.decide_main(["--preset", "triage"]) == 0
    assert FakeAgent.calls[-1]["state"] == "发票4411被重复扣款"
    assert json.loads(capsys.readouterr().out) == FAKE_RESULT


def test_decide_joins_positional_text(capsys):
    cli.decide_main(["charged", "twice"])
    assert FakeAgent.calls[-1]["state"] == "charged twice"
    json.loads(capsys.readouterr().out)


def test_decide_state_file_is_read_as_json(tmp_path, capsys):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"message": "hi"}))
    cli.decide_main(["--state-file", str(path)])
    assert FakeAgent.calls[-1]["state"] == {"message": "hi"}
    capsys.readouterr()


def test_field_prints_a_single_value(capsys):
    cli.decide_main(["--field", "answers.intent.choice"])
    assert capsys.readouterr().out.strip() == "refund"


def test_field_prints_numbers_as_json(capsys):
    cli.decide_main(["--field", "answers.is_urgent.noul"])
    assert capsys.readouterr().out.strip() == "0.25"


def test_unknown_field_exits_2_with_one_sentence(capsys):
    # A harness parses stderr; it should not have to read a traceback.
    with pytest.raises(SystemExit) as exit_info:
        cli.decide_main(["--field", "answers.nope"])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "not in the result" in err
    assert "intent" in err and "is_urgent" in err
    assert "Traceback" not in err


def test_preset_defaults_to_router(monkeypatch, capsys):
    cli.decide_main([])
    asked = FakeAgent.calls[-1]["questions"]
    assert asked == cli.presets.router_questions()
    capsys.readouterr()


@pytest.mark.parametrize("name", sorted(cli.PRESETS))
def test_every_preset_name_is_accepted(name, capsys):
    cli.decide_main(["--preset", name])
    assert FakeAgent.calls[-1]["questions"] == cli.PRESETS[name]()
    capsys.readouterr()


def test_model_defaults_to_the_environment_then_the_hub(monkeypatch, capsys):
    monkeypatch.setenv("LAYA_MLX_MODEL", "/tmp/local-checkpoint")
    cli.decide_main([])
    assert FakeAgent.calls[-1]["model"] == "/tmp/local-checkpoint"
    capsys.readouterr()


def test_device_and_dtype_are_forwarded(tmp_path, capsys):
    cli.decide_main(["--device", "cpu", "--dtype", "float32", "--batch-size", "4"])
    assert FakeAgent.calls[-1]["kwargs"]["device"] == "cpu"
    assert FakeAgent.calls[-1]["kwargs"]["dtype"] == "float32"
    assert FakeAgent.calls[-1]["kwargs"]["batch_size"] == 4
    capsys.readouterr()


def test_predict_accepts_a_preset_instead_of_a_questions_file(capsys):
    # Upstream's CLI grew --preset; without it every harness call needs a questions file.
    assert cli.main(["predict", "--preset", "guard", "--state", "hi", "--device", "cpu"]) == 0
    assert FakeAgent.calls[-1]["questions"] == cli.presets.guard_questions()
    capsys.readouterr()


def test_predict_requires_a_questions_source():
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["predict", "--state", "hi"])
    assert exit_info.value.code == 2


def test_predict_rejects_a_questions_file_and_a_preset_together(tmp_path):
    path = tmp_path / "q.json"
    path.write_text("{}")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["predict", "--state", "hi", "--questions", str(path), "--preset", "guard"])
    assert exit_info.value.code == 2


def test_predict_reads_the_state_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", FakeStdin("from stdin"))
    cli.main(["predict", "--state-stdin", "--preset", "triage"])
    assert FakeAgent.calls[-1]["state"] == "from stdin"
    capsys.readouterr()
