"""Question normalization: what ``Agent._to_internal`` turns a public question into.

Ported from upstream ``laya/tests/test_criteria_normalization.py`` (v0.3.20), plus the labels
shipped through unchanged (upstream #163).
"""

from laya_mlx.agent import Agent
from laya_mlx.common import render_options


def test_noul_criteria_boolean_keys():
    qdef = {
        "type": "noul",
        "instructions": "Is this a refund?",
        "criteria": {True: "User requests money back", False: "User does not ask for money back"},
    }
    internal = Agent._to_internal(qdef)
    assert "true" in internal["crit"]
    assert "false" in internal["crit"]
    opts = render_options(internal)
    assert len(opts) == 2
    assert opts[0].startswith("false: User does not ask for money back")
    assert opts[1].startswith("true: User requests money back")


def test_choice_criteria_list_expansion():
    qdef = {
        "type": "choice",
        "instructions": "Select urgency",
        "criteria": ["low", "medium", "high"],
    }
    internal = Agent._to_internal(qdef)
    assert list(internal["crit"].keys()) == ["low", "medium", "high"]


def test_choice_criteria_list_expansion_keeps_values_none():
    internal = Agent._to_internal(
        {"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]}
    )
    assert internal["crit"] == {"a": None, "b": None}


def test_uppercase_noul_criteria_keys_normalize():
    internal = Agent._to_internal(
        {"type": "noul", "instructions": "Is it spam?", "criteria": {"TRUE": "yes", "FALSE": "no"}}
    )
    assert internal["crit"] == {"true": "yes", "false": "no"}


def test_noul_labels_are_carried_through_untouched():
    labels = {"true": "A", "false": "B"}
    internal = Agent._to_internal({"type": "noul", "instructions": "Is it true?", "labels": labels})
    assert internal["labels"] == labels
    assert "labels" not in Agent._to_internal({"type": "noul", "instructions": "Is it true?"})
