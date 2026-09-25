"""Criteria rendering: structured values must not crash or leak Python reprs.

Regression tests ported from upstream (``laya/tests/test_criteria.py``, bug reported in PR #2 by
trocker): a ``noul`` question whose criteria values were dicts raised
``TypeError: can only concatenate str (not "dict") to str``, and ``choice``/``score``
stringified dicts as Python reprs instead of JSON.

Upstream's final section -- source inspection of ``laya.agent.Agent.__init__`` for a torch
CUDA-fallback warning -- is intentionally dropped: the MLX port has no torch fallback path.
"""

import json

import pytest

from laya_mlx.common import render_criterion, render_options


def _question(t, crit):
    """A minimal internal question; only ``t`` and ``crit`` matter for option rendering."""
    return {"t": t, "ins": "x", "crit": crit}


# ------------------------------------------------------------------------- render_criterion


@pytest.mark.parametrize(
    "value,want",
    [
        ("phishing or scam", "phishing or scam"),
        ({"desc": "phishing"}, '{"desc": "phishing"}'),
        (["a", "b"], '["a", "b"]'),
        (3, "3"),
        (False, "false"),
        (True, "true"),
        # non-ASCII must survive rather than be \u-escaped
        ({"d": "münchen"}, '{"d": "münchen"}'),
    ],
)
def test_render_criterion(value, want):
    assert render_criterion(value) == want


def test_render_criterion_unserialisable_falls_back_to_str():
    rendered = render_criterion({"o": object()})
    assert isinstance(rendered, str)
    assert "'" not in rendered  # a Python repr leaked instead of JSON


# --------------------------------------------------------------- the reported crash: noul+dict


def test_noul_with_dict_criteria_does_not_crash():
    out = render_options(
        {
            "t": "noul",
            "ins": "Is this phishing?",
            "crit": {
                "true": {"desc": "phishing, scam or fraud"},
                "false": {"desc": "legitimate"},
            },
        }
    )
    assert len(out) == 2
    assert out[0] == 'false: {"desc": "legitimate"}'
    assert out[1] == 'true: {"desc": "phishing, scam or fraud"}'
    assert "'" not in "".join(out)


# ------------------------------------------------------------------------- choice and score


def test_choice_renders_dict_none_and_empty_string():
    out = render_options(
        _question("choice", {"billing": {"desc": "payments"}, "tech": None, "sales": ""})
    )
    assert out[0] == 'billing: {"desc": "payments"}'
    assert out[1] == "tech"
    assert out[2] == "sales"
    assert "{'" not in "".join(out)


def test_choice_keeps_zero_and_false_as_real_values():
    # 0 and False are legitimate criteria, not "missing"
    out = render_options(_question("choice", {"zero": 0, "no": False}))
    assert out == ["zero: 0", "no: false"]


def test_score_renders_levels_in_criterion_order():
    out = render_options(_question("score", [{"d": "low"}, "high", 2]))
    assert out == ['level 0: {"d": "low"}', "level 1: high", "level 2: 2"]


# --------------------------------------------------------------------- unchanged behaviour


def test_noul_default_texts():
    out = render_options(_question("noul", None))
    assert out[0] == "false: no, the statement does not hold"
    assert out[1] == "true: yes, the statement holds"


@pytest.mark.parametrize(
    "t,crit,want",
    [
        ("noul", {"true": "yes it is", "false": "no"}, ["false: no", "true: yes it is"]),
        ("choice", {"a": "first", "b": None}, ["a: first", "b"]),
        ("score", ["low", "high"], ["level 0: low", "level 1: high"]),
    ],
)
def test_plain_string_criteria_are_unchanged(t, crit, want):
    assert render_options(_question(t, crit)) == want


# ------------------------------------------------------------- type safety and json validity


@pytest.mark.parametrize(
    "t,crit",
    [
        ("choice", {"a": {"n": 1}, "b": [1, 2], "c": 3.5}),
        ("score", [{"a": 1}, [2], None]),
        ("noul", {"true": [1], "false": {"z": 0}}),
    ],
)
def test_every_rendered_option_is_a_str(t, crit):
    assert all(isinstance(o, str) for o in render_options(_question(t, crit)))


def test_emitted_json_round_trips_through_json_loads():
    out = render_options(_question("noul", {"true": {"a": 1}, "false": {"b": 2}}))
    assert json.loads(out[1].split("true: ", 1)[1]) == {"a": 1}
    assert json.loads(out[0].split("false: ", 1)[1]) == {"b": 2}
