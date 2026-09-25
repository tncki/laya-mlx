"""Empty-question inference regressions; tiny MLX fixture, no training or downloads.

Ported from upstream ``laya/tests/test_empty_questions.py`` (v0.3.20, #144): empty questions must
return empty answers with zero usage without tokenizing, batching, or running the model, and the
response must not share mutable containers between calls. Upstream builds a torch DecisionModel in
memory; this port uses the tiny MLX checkpoint fixture.
"""

import copy
from unittest.mock import Mock, patch

import pytest

from laya_mlx import Agent

EMPTY = {
    "model": "laya-rl-agent",
    "answers": {},
    "usage": {"input_tokens": 0, "output_tokens": 0},
}
STATES = ["hello", {"text": "hello"}, [{"role": "user", "content": "hello"}], "", {}, []]


def _bare_agent():
    """The loaded runtime attributes only; no checkpoint is needed for the empty path."""
    agent = object.__new__(Agent)
    agent.cfg = {"max_len": 64, "head_max_len": 32}
    return agent


@pytest.mark.parametrize("method", ["predict", "system_one"])
@pytest.mark.parametrize("state", STATES)
def test_both_methods_return_the_empty_response_without_touching_the_state(method, state):
    agent = _bare_agent()
    original = copy.deepcopy(state)
    questions = {}
    assert getattr(agent, method)(state, questions) == EMPTY
    assert state == original
    assert questions == {}


def test_empty_questions_skip_tokenization_batching_and_forward():
    agent = _bare_agent()
    agent.tok = Mock(side_effect=AssertionError("unexpected tokenization"))
    agent.model = Mock(side_effect=AssertionError("unexpected forward pass"))
    with (
        patch(
            "laya_mlx.agent.build_sequence", side_effect=AssertionError("unexpected encoding")
        ) as encode,
        patch(
            "laya_mlx.agent.collate_items", side_effect=AssertionError("unexpected batching")
        ) as collate,
    ):
        assert agent.predict("hello", {}) == EMPTY
        assert agent.system_one("hello", {}) == EMPTY
    encode.assert_not_called()
    collate.assert_not_called()
    agent.tok.assert_not_called()
    agent.model.assert_not_called()


def test_empty_responses_do_not_share_mutable_containers():
    agent = _bare_agent()
    result = agent.predict("hello", {})
    result["answers"]["changed"] = True
    result["usage"]["input_tokens"] = 7
    assert agent.system_one("hello", {}) == EMPTY


def test_nonempty_predictions_are_unchanged_after_an_empty_call(tiny_checkpoint):
    agent = Agent(tiny_checkpoint, dtype="float32")
    questions = {
        "choice": {"type": "choice", "instructions": "Pick one", "criteria": ["yes", "no"]},
        "score": {
            "type": "score",
            "instructions": "Rate it",
            "criteria": ["low", "medium", "high"],
        },
        "noul": {"type": "noul", "instructions": "Is it true?"},
    }
    original = copy.deepcopy(questions)
    with patch.object(agent, "forward", wraps=agent.forward) as forward:
        before = agent.predict("hello", questions)
        assert agent.predict("hello", {}) == EMPTY
        after = agent.system_one("hello", questions)
    assert forward.call_count == 2
    assert before == after
    assert questions == original
    assert before["model"] == "laya-rl-agent"
    assert before["usage"]["input_tokens"] > 0
    assert before["usage"]["output_tokens"] == 0
    answers = before["answers"]
    assert set(answers) == set(questions)
    for qid in ("choice", "score"):
        assert answers[qid]["type"] == qid
        assert sum(answers[qid]["probabilities"].values()) == pytest.approx(1.0, abs=2e-4)
    assert answers["choice"]["choice"] in ("yes", "no")
    assert 0 <= answers["score"]["score"] <= 2
    assert answers["noul"]["type"] == "noul"
    assert 0 <= answers["noul"]["noul"] <= 1


def test_prepare_with_empty_questions_is_a_no_op(tiny_checkpoint):
    agent = Agent(tiny_checkpoint, dtype="float32")
    assert agent.prepare("hello", {}) == ([], [])
