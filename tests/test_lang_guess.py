"""Caller-supplied language hint (`lang_guess`) and blank-`lang` routing, from upstream v0.3.20.

Spec sources: ``laya/tests/test_lang_guess.py`` (#211) and
``laya/tests/test_blank_lang_routing.py`` (#292). No weights are loaded: ``Router.route`` is pure
and ``predict`` runs against a stub Agent.
"""

import inspect

import pytest

from laya_mlx.router import Router, _english_from_code

# The state the maintainer used on #35: a short Romanian request the heuristic cannot place.
ROMANIAN = "Care este ora in Tokyo?"
GENERIC = {"intent": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}
ENGLISH = "I was charged twice for invoice 4411"
GERMAN = "Mein Konto wurde zweimal belastet, bitte erstatten Sie"


# ------------------------------------------------------------------ the baseline
def test_baseline_romanian_reaches_english():
    decision = Router().route(ROMANIAN, GENERIC)
    # Pinned, not "either checkpoint": this defect is what the hint exists to fix.
    assert decision["detection"]["language"] == "en"
    assert decision.model == "english"


# ------------------------------------------------------------------ codes
@pytest.mark.parametrize(
    "code,want",
    [
        ("ro", "multilingual"),
        ("en", "english"),
        ("en_US", "english"),
        ("en_US.UTF-8", "english"),
        ("de-DE", "multilingual"),
        ("RO", "multilingual"),
        ("  en  ", "english"),
        ("qq", "multilingual"),
    ],
)
def test_code_hint_routes(code, want):
    assert Router().route(ROMANIAN, GENERIC, lang_guess=code).model == want


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_abstaining_hint_falls_through_to_detection(empty):
    assert Router().route(ROMANIAN, GENERIC, lang_guess=empty)["detection"] is not None


# ------------------------------------------------------------------ callables
def test_callable_hint_is_used():
    assert Router().route(ROMANIAN, GENERIC, lang_guess=lambda s: "ro").model == "multilingual"
    assert (
        Router()
        .route(ROMANIAN, GENERIC, lang_guess=lambda s: "en" if "Tokyo" in str(s) else "ro")
        .model
        == "english"
    )


@pytest.mark.parametrize("result", [None, ""])
def test_callable_abstaining_falls_through(result):
    assert Router().route(ROMANIAN, GENERIC, lang_guess=lambda s: result)["detection"] is not None


def test_abstaining_callable_does_not_change_the_default_route():
    router = Router()
    assert (
        router.route(ROMANIAN, GENERIC, lang_guess=lambda s: None).model
        == router.route(ROMANIAN, GENERIC).model
    )


# ------------------------------------------------------------------ installed on the Router
def test_installed_hint_applies_and_is_overridable():
    installed = Router(lang_guess="ro")
    assert installed.route(ROMANIAN, GENERIC).model == "multilingual"
    assert installed.route(ROMANIAN, GENERIC, lang_guess="en").model == "english"
    assert Router(lang_guess=lambda s: "ro").route(ROMANIAN, GENERIC).model == "multilingual"
    assert Router().lang_guess is None
    assert Router(lang_guess=lambda s: None).route(ROMANIAN, GENERIC)["detection"] is not None


# ------------------------------------------------------------------ precedence
def test_precedence_explicit_arguments_beat_the_hint():
    router = Router()
    assert router.route(ROMANIAN, GENERIC, model="english", lang_guess="ro").model == "english"
    assert (
        router.route(ROMANIAN, GENERIC, task="typed_decisions", lang_guess="ro").model
        == "typed-decisions"
    )
    explicit = router.route(ROMANIAN, GENERIC, lang="en", lang_guess="ro")
    assert explicit.model == "english"
    assert "explicit lang" in explicit.reason


# ------------------------------------------------------------------ the decision payload
def test_hint_decision_payload():
    decision = Router().route(ROMANIAN, GENERIC, lang_guess="ro")
    assert decision["model"] == "multilingual"
    assert isinstance(decision["repo"], str)
    assert "lang_guess" in decision["reason"]
    assert decision["detection"] is None
    assert "Router(lang_guess=...)" in Router(lang_guess="ro").route(ROMANIAN, GENERIC)["reason"]
    assert "convaiinnovations/laya" in decision["repo"]


def test_standalone_repos_are_used_by_the_hint():
    decision = Router(standalone_repos=True, lang_guess="ro").route(ROMANIAN, GENERIC)
    assert decision["repo"] == "convaiinnovations/laya-multilingual"


# ------------------------------------------------------------------ predict forwards it
class _FakeAgent:
    """Stands in for a loaded checkpoint (no `lang` parameter, as the port's Agent has none)."""

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


class _LangAwareAgent:
    """An Agent-like object that does accept `lang`, to prove predict forwards it."""

    def __init__(self):
        self.langs = []

    def system_one(self, state, questions, lang=None):
        self.langs.append(lang)
        return {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


def _router_with_stub(agent, **kwargs):
    router = Router(**kwargs)
    router.load = lambda name: agent
    return router


def test_predict_uses_and_forwards_the_hint():
    stub = _FakeAgent()
    router = _router_with_stub(stub, lang_guess="ro")
    out = router.predict(ROMANIAN, GENERIC)
    assert out["routing"]["model"] == "multilingual"
    assert router.predict(ROMANIAN, GENERIC, lang_guess="en")["routing"]["model"] == "english"
    assert len(stub.calls) == 2
    assert stub.calls[0][0] == ROMANIAN
    assert set(out["routing"]) >= {"model", "repo", "reason", "detection", "workflow"}


def test_predict_forwards_an_agent_that_accepts_lang():
    agent = _LangAwareAgent()
    router = _router_with_stub(agent)
    # The detected language reaches the agent, not just the routing block.
    router.predict(GERMAN, GENERIC)
    assert agent.langs == ["de"]
    # An explicit `lang` wins over detection.
    router.predict(GERMAN, GENERIC, lang="zh-CN")
    assert agent.langs[-1] == "zh-CN"


def test_predict_works_when_the_agent_rejects_lang():
    # The port's own Agent.system_one(state, questions) has no `lang`; predict must still work.
    stub = _FakeAgent()
    assert "routing" in _router_with_stub(stub).predict(ENGLISH, GENERIC)


# ------------------------------------------------------------------ the helper itself
@pytest.mark.parametrize(
    "value,want",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("en", True),
        ("en_US", True),
        ("zh_CN", False),
        ("en.UTF-8", True),
        (".", None),
    ],
)
def test_english_from_code(value, want):
    assert _english_from_code(value) is want


def test_without_a_hint_nothing_moves():
    router = Router()
    for label, state, want in [
        ("plain english", "I was charged twice and want a refund", "english"),
        (
            "German with umlauts",
            "Mein Konto wurde zweimal belastet, bitte erstatten Sie",
            "multilingual",
        ),
        ("Hindi", "यह एक हिंदी वाक्य है", "multilingual"),
        ("empty", "", "english"),
        ("digits", "12345", "english"),
    ]:
        assert router.route(state, GENERIC).model == want, label


# ------------------------------------------------------------------ blank `lang` (#292)
def test_blank_lang_falls_through_to_detection():
    router = Router()
    for value in ("", "   ", None):
        decision = router.route(ENGLISH, GENERIC, lang=value)
        assert decision.model == "english"
        assert "explicit lang=" not in decision.reason
        assert decision["detection"] is not None


def test_real_codes_still_win():
    router = Router()
    assert router.route(GERMAN, GENERIC, lang="en").model == "english"
    assert router.route(ENGLISH, GENERIC, lang="de").model == "multilingual"
    assert "explicit lang=" in router.route(GERMAN, GENERIC, lang="en").reason
    assert "explicit lang=" in router.route(ENGLISH, GENERIC, lang="de").reason


@pytest.mark.parametrize("code", ["en", "EN", "en-US", "en_US", "en_US.UTF-8"])
def test_english_codes(code):
    assert Router().route(ENGLISH, GENERIC, lang=code).model == "english"


@pytest.mark.parametrize("code", ["de", "fr", "zh_CN", "pt-BR"])
def test_non_english_codes(code):
    assert Router().route(ENGLISH, GENERIC, lang=code).model == "multilingual"


def test_blank_lang_does_not_mask_an_installed_hint():
    router = Router(lang_guess="de")
    assert router.route(GERMAN, GENERIC, lang="").model == "multilingual"
    assert router.route(GERMAN, GENERIC).model == "multilingual"


def test_baseline_no_lang_emits_no_explicit_reason():
    assert "explicit lang=" not in Router().route(ENGLISH, GENERIC).reason


# ------------------------------------------------------------------ upstream test_system_one_lang.py
def test_agent_system_one_accepts_lang_for_router_forwarding():
    """Port of upstream ``tests/test_system_one_lang.py``.

    Upstream proves ``Agent.system_one(state, q, lang=...)`` forwards ``lang`` to
    ``Agent.predict_batch``. The port's Agent has no ``predict_batch`` (batch inference is
    excluded from this port), so the contract that can hold here is the public parameter the
    Router forwards into -- verified end-to-end for Agent-like objects by
    ``test_predict_forwards_an_agent_that_accepts_lang``. If ``laya_mlx/agent.py`` drops ``lang``,
    this fails instead of silently skipping the capability.
    """
    from laya_mlx.agent import Agent

    params = inspect.signature(Agent.system_one).parameters
    assert "lang" in params, "Agent.system_one must accept `lang` for Router to forward it"
    assert params["lang"].default is None
    # The per-language calibration upstream passes it to (#258) must be constructible.
    assert "lang_temperatures" in inspect.signature(Agent.__init__).parameters
