"""Criteria rendering: structured values must not crash or leak Python reprs.

Regression tests ported from upstream (``laya/tests/test_criteria.py`` at v0.3.20, bug reported in
PR #2 by trocker): a ``noul`` question whose criteria values were dicts raised
``TypeError: can only concatenate str (not "dict") to str``, and ``choice``/``score``
stringified dicts as Python reprs instead of JSON.

Also ported here: the calibration-boundary checks upstream's file carries (upstream #39),
custom ``noul`` labels (upstream #163), the criteria-key guard and malformed-question shapes that
``Agent.system_one`` must reject by name (upstream #182/#249/#156), non-ASCII instructions
(upstream #228), and the ``build_sequence`` left-truncation fix (upstream #112).

Upstream's final section -- source inspection of ``laya.agent.Agent.__init__`` for a torch
CUDA-fallback warning -- is intentionally dropped: the MLX port has no torch fallback path.
"""

import json

import numpy as np
import pytest

from laya_mlx.agent import Agent
from laya_mlx.common import (
    answer_confidence,
    build_sequence,
    confidence_from_probs,
    ece_score,
    render_criterion,
    render_options,
)


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


def test_choice_boolean_word_labels_are_not_rewritten():
    # normalization is only for `noul` criteria; a choice label named "true" stays "true"
    assert render_options(_question("choice", {"true": "yes", "false": "no"})) == [
        "true: yes",
        "false: no",
    ]


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


# --------------------------------------------------------------- calibration boundaries (#39)
# The first bin must include 0.0, otherwise a zero-confidence answer falls outside every bin and
# ECE silently ignores it.


def test_ece_includes_zero_confidence():
    assert ece_score(np.array([0.0]), np.array([1.0])) == 1.0


def test_ece_zero_confidence_has_its_proper_weight():
    assert ece_score(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_ece_empty_is_nan():
    assert np.isnan(ece_score(np.array([]), np.array([])))


# --------------------------------------------------------- answer_confidence is max(p) (#126)


@pytest.mark.parametrize(
    "probs", [[0.5, 0.5], [0.1, 0.9], [0.7, 0.2, 0.1], [0.25] * 4, [1.0, 0, 0]]
)
def test_answer_confidence_is_max_probability(probs):
    p = np.array(probs)
    assert answer_confidence(p, len(probs)) == max(probs)


def test_answer_confidence_only_counts_the_first_k_entries():
    assert answer_confidence(np.array([0.4, 0.6, 0.99]), 2) == 0.6


def test_answer_confidence_k_edges():
    assert answer_confidence(np.array([1.0]), 1) == 1.0
    assert answer_confidence(np.array([]), 0) == 1.0


def test_answer_confidence_matches_noul_confidence():
    # noul reports max(p_true, 1 - p_true); over two options that is exactly max(p)
    for p_true in (0.0, 0.05, 0.3, 0.5, 0.62, 0.9, 1.0):
        p = np.array([1.0 - p_true, p_true])
        assert answer_confidence(p, 2) == max(p_true, 1.0 - p_true)


def test_answer_confidence_differs_from_entropy_confidence():
    # the two scales must not be compared against one threshold
    p = np.array([0.1, 0.9])
    assert confidence_from_probs(p, 2) < 0.55 < answer_confidence(p, 2)
    assert answer_confidence(np.array([0.5, 0.5]), 2) == 0.5
    assert confidence_from_probs(np.array([0.5, 0.5]), 2) == 0.0


def test_answer_confidence_is_exported_at_top_level():
    import laya_mlx

    assert "answer_confidence" in laya_mlx.__all__
    assert hasattr(laya_mlx, "answer_confidence")
    assert "answer_confidence" in dir(laya_mlx)


# --------------------------------------------------------------- noul labels (#163)

_LABEL_ERROR = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"


def test_noul_explicit_none_labels_use_defaults():
    assert render_options(_question("noul", None) | {"labels": None}) == [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]


def test_noul_custom_labels_preserve_false_then_true_semantics():
    custom = {"true": " A ", "false": " B "}
    out = render_options({"t": "noul", "ins": "x", "crit": None, "labels": custom})
    assert out == ["B: no, the statement does not hold", "A: yes, the statement holds"]
    # the caller's dict is not mutated by stripping
    assert custom == {"true": " A ", "false": " B "}


@pytest.mark.parametrize(
    "labels",
    [
        ["negative", "positive"],
        {"false": "negative"},
        {"false": "negative", "true": "positive", "other": "x"},
        {"false": " ", "true": "positive"},
        {"false": "same", "true": "same"},
        {"false": 0, "true": "positive"},
        {"false": "", "true": "positive"},
    ],
)
def test_noul_invalid_labels_are_rejected(labels):
    with pytest.raises(ValueError) as excinfo:
        render_options({"t": "noul", "ins": "x", "labels": labels})
    assert str(excinfo.value) == _LABEL_ERROR


@pytest.mark.parametrize(
    "t,crit",
    [("choice", {"a": None, "b": None}), ("score", ["low", "high"])],
)
def test_labels_are_rejected_outside_noul(t, crit):
    with pytest.raises(ValueError) as excinfo:
        render_options({"t": t, "ins": "x", "crit": crit, "labels": {"false": "B", "true": "A"}})
    assert str(excinfo.value) == "labels is only supported for noul questions"


# ------------------------------------- labels reach the renderer through Agent._to_internal


def test_agent_forwards_noul_labels_without_mutating_the_caller_question():
    public_labels = {"true": "A", "false": "B"}
    public_question = {"type": "noul", "instructions": "Is this true?", "labels": public_labels}
    internal = Agent._to_internal(public_question)
    assert internal["labels"] == public_labels
    assert render_options(internal) == [
        "B: no, the statement does not hold",
        "A: yes, the statement holds",
    ]
    assert public_question == {
        "type": "noul",
        "instructions": "Is this true?",
        "labels": {"true": "A", "false": "B"},
    }


def test_agent_normalizes_boolean_noul_criteria_and_keeps_input_unchanged():
    boolean_criteria = {True: "yes", False: "no"}
    internal = Agent._to_internal(
        {
            "type": "noul",
            "instructions": "Is this true?",
            "criteria": boolean_criteria,
            "labels": {"false": "B", "true": "A"},
        }
    )
    assert internal["crit"] == {"true": "yes", "false": "no"}
    assert render_options(internal) == ["B: no", "A: yes"]
    assert boolean_criteria == {True: "yes", False: "no"}


# ----------------------------------------------------- non-ASCII instructions (#228)


def test_non_ascii_instructions_keep_their_characters():
    internal = Agent._to_internal(
        {
            "type": "noul",
            "instructions": {"frage": "Bittet um eine R\u00fcckerstattung?"},
            "criteria": None,
        }
    )
    assert internal["ins"] == '{"frage": "Bittet um eine R\u00fcckerstattung?"}'
    assert "\\u" not in internal["ins"]
    assert (
        Agent._to_internal(
            {"type": "noul", "instructions": {"asks": "for a refund"}, "criteria": None}
        )["ins"]
        == '{"asks": "for a refund"}'
    )
    assert (
        Agent._to_internal(
            {
                "type": "noul",
                "instructions": "Bittet der Kunde um eine R\u00fcckerstattung?",
                "criteria": None,
            }
        )["ins"]
        == "Bittet der Kunde um eine R\u00fcckerstattung?"
    )
    assert (
        Agent._to_internal({"type": "noul", "instructions": ["a", "b"], "criteria": None})["ins"]
        == '["a", "b"]'
    )


# ------------------------------------------------- noul criteria keys (#249 / #156)

_DEFAULT_FALSE_TEXT = "false: no, the statement does not hold"
_DEFAULT_TRUE_TEXT = "true: yes, the statement holds"


@pytest.mark.parametrize(
    "crit,want",
    [
        (
            {"true": "the review is positive", "false": "the review is negative"},
            ["false: the review is negative", "true: the review is positive"],
        ),
        # uppercase keys are normalised by _to_internal
        (
            {"TRUE": "the review is positive", "FALSE": "the review is negative"},
            ["false: the review is negative", "true: the review is positive"],
        ),
        # Python bool keys are how JSON true/false arrive
        (
            {True: "the review is positive", False: "the review is negative"},
            ["false: the review is negative", "true: the review is positive"],
        ),
        ({"true": "the review is positive"}, [_DEFAULT_FALSE_TEXT, "true: the review is positive"]),
        ({}, [_DEFAULT_FALSE_TEXT, _DEFAULT_TRUE_TEXT]),
        (None, [_DEFAULT_FALSE_TEXT, _DEFAULT_TRUE_TEXT]),
        (
            {"true": "yes, the statement holds", "false": "no, the statement does not hold"},
            [_DEFAULT_FALSE_TEXT, _DEFAULT_TRUE_TEXT],
        ),
    ],
)
def test_noul_criteria_spellings_that_keep_working(crit, want):
    qdef = {"type": "noul", "instructions": "Is the review positive?"}
    if crit is not None:
        qdef["criteria"] = crit
    assert render_options(Agent._to_internal(qdef)) == want


def test_noul_criteria_text_survives_alongside_labels():
    mixed = Agent._to_internal(
        {
            "type": "noul",
            "instructions": "Is the review positive?",
            "criteria": {"true": "the review is positive", "false": "the review is negative"},
            "labels": {"true": "positive", "false": "negative"},
        }
    )
    assert render_options(mixed) == [
        "negative: the review is negative",
        "positive: the review is positive",
    ]
    assert mixed["labels"] == {"true": "positive", "false": "negative"}
    assert render_options(mixed)[0].startswith("negative:")
    # `labels` changes only the prefix, never the criteria text behind it
    plain = Agent._to_internal(
        {
            "type": "noul",
            "instructions": "Is the review positive?",
            "criteria": {"true": "the review is positive", "false": "the review is negative"},
        }
    )
    assert [o.split(": ", 1)[1] for o in render_options(mixed)] == [
        o.split(": ", 1)[1] for o in render_options(plain)
    ]


# ------------------------------------------------- malformed question shapes (#182/#249)

_STATE = {"body": "I was charged twice for invoice 4411 and want a refund"}

REJECTED_QUESTIONS = [
    ("choice without criteria", {"type": "choice", "instructions": "Which team?"}),
    (
        "choice with criteria None",
        {"type": "choice", "instructions": "Which team?", "criteria": None},
    ),
    (
        "choice with empty criteria",
        {"type": "choice", "instructions": "Which team?", "criteria": {}},
    ),
    (
        "choice with a tuple of labels",
        {"type": "choice", "instructions": "Which team?", "criteria": ("billing", "tech")},
    ),
    ("score without criteria", {"type": "score", "instructions": "How urgent?"}),
    ("score with an empty list", {"type": "score", "instructions": "How urgent?", "criteria": []}),
    (
        "score with a dict of levels",
        {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": {"low": "no pressure", "high": "blocking"},
        },
    ),
    (
        "choice with labels",
        {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": ["billing", "tech"],
            "labels": {"false": "B", "true": "A"},
        },
    ),
    (
        "score with labels",
        {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "high"],
            "labels": {"false": "B", "true": "A"},
        },
    ),
    (
        "noul with list criteria",
        {"type": "noul", "instructions": "Is it spam?", "criteria": ["a", "b"]},
    ),
    (
        "noul with string criteria",
        {"type": "noul", "instructions": "Is it spam?", "criteria": "spam?"},
    ),
    (
        "noul with incomplete labels",
        {"type": "noul", "instructions": "Is it spam?", "labels": {"true": "A"}},
    ),
    (
        "noul with duplicate labels",
        {"type": "noul", "instructions": "Is it spam?", "labels": {"false": "A", "true": "A"}},
    ),
    # `render_options` reads the two noul descriptions by name, so any other key used to be
    # dropped and replaced with the defaults without a word (#156).
    (
        "noul with yes/no criteria",
        {
            "type": "noul",
            "instructions": "Is it spam?",
            "criteria": {"yes": "it is spam", "no": "it is not"},
        },
    ),
    (
        "noul with neutral keys",
        {
            "type": "noul",
            "instructions": "Is it spam?",
            "criteria": {"spam": "it is spam", "ham": "it is not"},
        },
    ),
    (
        "noul with alpha/beta criteria",
        {"type": "noul", "instructions": "Is it spam?", "criteria": {"alpha": "yes", "beta": "no"}},
    ),
    (
        "noul with a typo'd key",
        {"type": "noul", "instructions": "Is it spam?", "criteria": {"ture": "yes", "false": "no"}},
    ),
    (
        "noul with an extra key",
        {
            "type": "noul",
            "instructions": "Is it spam?",
            "criteria": {"true": "y", "false": "n", "maybe": "?"},
        },
    ),
    ("unknown type", {"type": "bool", "instructions": "Is it spam?"}),
    ("missing type", {"instructions": "Is it spam?"}),
    ("no instructions", {"type": "noul"}),
]


@pytest.mark.parametrize("label,qdef", REJECTED_QUESTIONS, ids=[c[0] for c in REJECTED_QUESTIONS])
def test_malformed_questions_are_rejected_by_name(label, qdef):
    agent = object.__new__(Agent)
    agent._prefix_cache = None
    with pytest.raises(ValueError) as excinfo:
        agent.system_one(_STATE, {"q": qdef})
    message = str(excinfo.value)
    assert "'q'" in message, message
    assert len(message) > 40, message


def test_every_question_is_validated_not_only_the_first():
    agent = object.__new__(Agent)
    agent._prefix_cache = None
    with pytest.raises(ValueError) as excinfo:
        agent.system_one(
            _STATE,
            {
                "ok": {"type": "noul", "instructions": "Is it urgent?"},
                "broken": {"type": "choice", "instructions": "Which team?"},
            },
        )
    assert "'broken'" in str(excinfo.value)


GOOD_QUESTIONS = {
    "choice": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "invoices and refunds", "tech": "bugs"},
    },
    "choice as a list": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": ["billing", "tech"],
    },
    "score": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["no pressure", "soon", "blocking"],
    },
    "noul": {"type": "noul", "instructions": "Does the sender want a reply?"},
    "noul with criteria": {
        "type": "noul",
        "instructions": "Is it phishing?",
        "criteria": {"true": "phishing", "false": "legitimate"},
    },
    "noul with labels": {
        "type": "noul",
        "instructions": "Is it phishing?",
        "criteria": {"true": "phishing", "false": "legitimate"},
        "labels": {"false": "B", "true": "A"},
    },
    "non-string instructions": {"type": "noul", "instructions": {"asks": "for a refund"}},
}


def test_valid_question_shapes_still_answer(tiny_checkpoint):
    # ...so the reject path above is not validation-only coverage
    agent = Agent(tiny_checkpoint, dtype="float32")
    out = agent.system_one(_STATE, GOOD_QUESTIONS)
    answers = out["answers"]
    assert sorted(answers) == sorted(GOOD_QUESTIONS)
    assert answers["choice"]["choice"] in ("billing", "tech")
    assert answers["choice as a list"]["choice"] in ("billing", "tech")
    assert round(sum(answers["choice"]["probabilities"].values()), 3) == 1.0
    assert 0.0 <= answers["score"]["score"] <= 2.0
    assert answers["score"]["legend"] == {"0": "no pressure", "1": "soon", "2": "blocking"}
    for qid in ("noul", "noul with criteria", "noul with labels"):
        assert 0.0 <= answers[qid]["noul"] <= 1.0
    assert out["usage"]["output_tokens"] == 0
    assert out["usage"]["input_tokens"] > 0


def test_router_surfaces_the_rejected_question_name(tiny_checkpoint):
    from laya_mlx import Router

    router = Router()
    router.attach("english", Agent(tiny_checkpoint, dtype="float32"))
    with pytest.raises(ValueError) as excinfo:
        router.predict(_STATE, {"q": {"type": "choice", "instructions": "x"}}, model="english")
    assert "'q'" in str(excinfo.value)


# --------------------------------------------------------------- build_sequence left truncation
# With no room left for the state, `st[-0:]` kept all of it: the closing [SEP] was replaced by the
# *first* state token, i.e. the wrong end of the state and an unterminated sequence (#112).


class _SeqTok:
    mask_token, mask_token_id, cls_token_id, sep_token_id = "[MASK]", 1, 2, 3

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [self.vocab.setdefault(w, 100 + len(self.vocab)) for w in text.split()]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


@pytest.mark.parametrize(
    "room,kept", [(0, []), (2, ["two", "three"]), (10, ["one", "two", "three"])]
)
def test_truncate_left_keeps_the_tail(room, kept):
    tok, q = _SeqTok(), {"t": "noul", "ins": "Is it urgent?", "crit": None}
    full = len(build_sequence(tok, "", q, 10**6)[0])  # prompt + closing [SEP], no state
    ids = build_sequence(tok, "one two three", q, full + room, truncate_left=True)[0]
    assert ids[full - 1 :] == [tok.vocab[w] for w in kept] + [tok.sep_token_id]


def test_truncate_left_default_is_unchanged():
    tok = _SeqTok()
    state = "one two three"
    seq_default, _ = build_sequence(
        tok, state, {"t": "noul", "ins": "Is it urgent?", "crit": None}, 8, 32
    )
    seq_explicit, _ = build_sequence(
        tok, state, {"t": "noul", "ins": "Is it urgent?", "crit": None}, 8, 32, truncate_left=False
    )
    assert seq_default == seq_explicit
