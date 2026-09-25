"""Routing, language-detection and temperature-clamp tests ported from upstream v0.3.20.

No model weights are loaded: `Router.route` is pure. Language-only cases live in
`tests/test_lang.py`; caller hints and blank `lang` in `tests/test_lang_guess.py`.
"""

import threading
import time
from unittest.mock import patch

import pytest

from laya_mlx import Router
from laya_mlx.common import QTYPES, TEMP_MAX, TEMP_MIN, clamp_temperature, temp_bucket
from laya_mlx.lang import analyse, detect_script, guess_latin_language, is_english, state_text
from laya_mlx.router import (
    BUNDLE_REPO,
    DEFAULT_MODELS,
    STANDALONE_MODELS,
    _repo_str,
    match_typed_decisions_workflow,
    normalise_name,
)

# --------------------------------------------------------------------- script detection


@pytest.mark.parametrize(
    "text,want",
    [
        ("The customer was charged twice and wants a refund.", "latin"),
        ("Հայերեն", "armenian"),
        ("ՀԱՅԵՐԵՆ", "armenian"),
        ("։֊", "unknown"),
        ("Le client a été facturé deux fois et demande un remboursement.", "latin"),
        ("ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"),
        ("お客様は二重に請求されたため返金を希望しています。", "kana"),
        ("客户被重复扣款要求退款", "han"),
        ("고객이 두 번 청구되어 환불을 원합니다", "hangul"),
        ("تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"),
        ("С клиента дважды сняли деньги и он хочет возврат", "cyrillic"),
        ("Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"),
        ("הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"),
        ("", "unknown"),
        ("12345 6789", "unknown"),
    ],
)
def test_detect_script(text, want):
    assert detect_script(text) == want


# --------------------------------------------------------------------- english vs not


@pytest.mark.parametrize(
    "text,want",
    [
        ("Please refund the duplicate charge on invoice 4411 today.", True),
        ("Հայերեն", False),
        ("refund me", True),
        ("ग्राहक से दो बार शुल्क लिया गया", False),
        ("お客様は二重に請求されました", False),
        (
            "Le client a été facturé deux fois et il demande un remboursement pour la "
            "facture qui a été payée le mois dernier avec la carte de crédit",
            False,
        ),
        (
            "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die "
            "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde",
            False,
        ),
        # Latin-script languages with no stopword list of their own: reported upstream in #35,
        # where Romanian states were handed to the English checkpoint instead of the
        # multilingual one. An unidentified language must never be assumed English.
        ("Gătește-mi o rețetă de sarmale de post pentru mâine.", False),
        ("Am fost taxat de două ori pentru factura din luna martie și vreau banii", False),
        ("Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę", False),
        ("Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz", False),
        ("Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım", False),
        ("Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay", False),
        # English with the odd loanword must not tip over into the multilingual checkpoint
        (
            "We visited a cafe in Zurich and the naive assumption about the "
            "invoice was wrong, so please refund the duplicate charge",
            True,
        ),
    ],
)
def test_is_english(text, want):
    assert is_english(text) is want


def test_undecided_latin_is_not_dressed_up_as_a_detection():
    # A single shared function word used to name a language ("para" in Turkish text was
    # called Spanish). Undecided must be reported as undecided.
    det = analyse("Müşteriden iki kez ücret alındı ve para iadesi istiyor")
    assert det["language_undecided"] is True
    assert det["language"] is None
    assert (
        analyse("Please refund the duplicate charge on the invoice")["language_undecided"] is False
    )
    assert analyse("Gătește-mi o rețetă de sarmale")["diacritic_rate"] > 0.02
    assert analyse("Please refund the duplicate charge today")["diacritic_rate"] == 0.0


def test_analyse_reports_the_same_keys_from_every_branch():
    keys = {
        "script",
        "script_profile",
        "language",
        "is_english",
        "language_undecided",
        "diacritic_rate",
        "non_latin_fraction",
    }
    for text in (
        "Please refund the duplicate charge",
        "ग्राहक से दो बार",
        "Gătește-mi o rețetă de sarmale",
        "12345 ???",
    ):
        assert set(analyse(text)) == keys


def test_zero_stopword_tie_invents_no_language():
    assert guess_latin_language("Cât e ora acum la Tokyo") is None


def test_known_gap_romanian_without_diacritics_still_reads_english():
    # Kept visible on purpose, as upstream does: short Romanian with no diacritics and an
    # English function word ("in") still reads as English. A real LID model is the fix.
    assert is_english("Care este ora in Tokyo?") is True


@pytest.mark.parametrize(
    "text,want",
    [
        ("The customer was charged twice and wants a refund for this invoice", "en"),
        ("Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr"),
        (
            "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung",
            "de",
        ),
        (
            "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura",
            "es",
        ),
        ("refund", None),
    ],
)
def test_guess_latin_language(text, want):
    assert guess_latin_language(text) == want


def test_state_text_flattens_and_ignores_keys():
    assert "charged twice" in state_text({"body": "charged twice", "n": 3})
    assert "deep" in state_text({"a": {"b": ["deep"]}})
    assert "x" in state_text(["x", {"y": "z"}])
    assert state_text(None) == ""
    det = analyse({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})
    assert det["is_english"] is False


def test_script_profile_armenian():
    assert analyse("Հայերեն")["script_profile"] == {"armenian": 1.0}
    assert analyse("Հայերեն abc")["non_latin_fraction"] == 0.7


# --------------------------------------------------------------------- routing decisions

Q_GENERIC = {
    "dept": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": None, "tech": None},
    }
}


@pytest.mark.parametrize(
    "text",
    [
        "Gătește-mi o rețetă de sarmale de post pentru mâine.",
        "Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez.",
        "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę",
        "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım",
    ],
)
def test_unidentified_latin_routes_to_multilingual(text):
    assert Router().route(text).model == "multilingual"


def test_unidentified_latin_reason_says_what_it_routed_on():
    reason = Router().route("Müşteriden iki kez ücret alındı ve para iadesi istiyor").reason
    assert "not identified" in reason


def test_identified_non_english_reason_names_the_language():
    reason = (
        Router()
        .route(
            "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die "
            "Rechnung die nicht korrekt ist"
        )
        .reason
    )
    assert "'de'" in reason


def test_english_routing_unchanged():
    router = Router()
    assert (
        router.route("Please refund the duplicate charge on invoice 4411 today.").model == "english"
    )
    assert router.route("refund me").model == "english"
    assert router.route("Հայերեն", model="english").model == "english"
    assert normalise_name("ML") == "multilingual"


# --------------------------------------------------------------------- temperature clamp (#35)


@pytest.mark.parametrize(
    "value,want",
    [
        (0.1006, 0.5),  # pathological sharpening
        (0.10058280825614929, TEMP_MIN),  # the shipped choice:11+ bucket
        (1.7601518630981445, 1.7601518630981445),  # legitimate value untouched
        (1.0, 1.0),
        (9.0, TEMP_MAX),
        (0.0, TEMP_MIN),
        (-3.0, TEMP_MIN),
        (None, 1.0),
        ("x", 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
    ],
)
def test_clamp_temperature(value, want):
    assert clamp_temperature(value) == want


def test_clamp_bounds_are_sane_and_bucket_matches_reported_case():
    assert TEMP_MIN <= 1.0 <= TEMP_MAX
    # 13 options is the bucket the reported skill-router landed in
    assert temp_bucket(QTYPES["choice"], 13) == "choice:11+"


def test_agent_clamps_shipped_temperatures_and_keeps_raw(tiny_checkpoint):
    import json

    import laya_mlx.agent as agent_mod

    cfg_path = tiny_checkpoint / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["temperature_by_options"]["choice:11+"] = 0.10058280825614929
    cfg_path.write_text(json.dumps(cfg))
    with pytest.warns(RuntimeWarning, match="clamping"):
        agent = agent_mod.Agent(tiny_checkpoint)
    assert agent.temperature_by_options["choice:11+"] == TEMP_MIN
    assert agent.temperature_by_options_raw["choice:11+"] == 0.10058280825614929
    assert agent.temperature_by_options["choice:2"] == 1.7  # legitimate value untouched


# --------------------------------------------------------------------- thread safety (#95)


def test_concurrent_loads_share_one_agent(monkeypatch):
    import laya_mlx.agent as agent_mod

    constructions = []
    cl = threading.Lock()

    class SlowAgent:
        def __init__(self, *args, **kwargs):
            time.sleep(0.05)  # widen the check-then-build window
            with cl:
                constructions.append(1)

    monkeypatch.setattr(agent_mod, "Agent", SlowAgent)
    router = Router()
    got = []

    def worker():
        got.append(router.load("english"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(x) for x in got}) == 1
    assert len(constructions) == 1
    assert len(router._order) == 1
    assert sorted(router._agents) == ["english"]


def test_concurrent_hotpath_keeps_lru_consistent(monkeypatch):
    import laya_mlx.agent as agent_mod

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(agent_mod, "Agent", FakeAgent)
    router = Router(max_loaded=3)
    router.load("english")  # warm the cache

    threads = [threading.Thread(target=lambda: router.load("english")) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(router._order) == 1
    assert len(router._agents) == 1
    assert router._order == ["english"]


# --------------------------------------------------------------------- routing decisions

Q_TD = {
    key: {"type": "noul", "instructions": "x"}
    for key in ("action", "category", "churn_risk", "needs_human", "urgency")
}

ROUTE_CASES = [
    ("english text", {"body": "I was charged twice, please refund."}, Q_GENERIC, {}, "english"),
    ("armenian text", {"body": "Հայերեն"}, Q_GENERIC, {}, "multilingual"),
    ("armenian explicit override", {"body": "Հայերեն"}, Q_GENERIC, {"model": "english"}, "english"),
    (
        "azerbaijani no diacritics",
        {"body": "Sifarisim gelmedi ve pulum geri qaytarilmadi, zehmet olmasa yoxlayin"},
        Q_GENERIC,
        {},
        "multilingual",
    ),
    (
        "azerbaijani explicit override",
        {"body": "Müştəridən iki dəfə pul alınıb"},
        Q_GENERIC,
        {"model": "english"},
        "english",
    ),
    ("hindi text", {"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC, {}, "multilingual"),
    ("japanese text", {"body": "二重に請求されました"}, Q_GENERIC, {}, "multilingual"),
    ("korean text", {"body": "두 번 청구되었습니다"}, Q_GENERIC, {}, "multilingual"),
    ("arabic text", {"body": "تم خصم المبلغ مرتين"}, Q_GENERIC, {}, "multilingual"),
    # Latin brand names are the letter plurality here, but the request itself is CJK (#113)
    (
        "chinese with a brand",
        {"body": "我的 iPhone 15 Pro Max 订单还没到"},
        Q_GENERIC,
        {},
        "multilingual",
    ),
    (
        "japanese with brands",
        {"body": "Amazonで買ったiPhoneが届かない"},
        Q_GENERIC,
        {},
        "multilingual",
    ),
    (
        "korean with a brand",
        {"body": "Samsung Galaxy 주문이 아직 안 왔어요"},
        Q_GENERIC,
        {},
        "multilingual",
    ),
    (
        "english with a han name",
        {"body": "My name is 王小明 and my order is late"},
        Q_GENERIC,
        {},
        "english",
    ),
    # an English wrapper dilutes the share, but the request is still CJK
    (
        "chinese in a ticket",
        {
            "ticket_id": "TCK-88213",
            "channel": "web chat",
            "agent_notes": "Customer asked about a delayed order. Please check shipping status.",
            "message": "我的订单已经两个星期了还没有到",
        },
        Q_GENERIC,
        {},
        "multilingual",
    ),
    (
        "korean after english turns",
        [
            {"role": "agent", "text": "Hello! Thanks for contacting support."},
            {"role": "agent", "text": "Could you share your order number please?"},
            {"role": "user", "text": "주문번호는 5521이고 아직 배송이 안 됐어요"},
        ],
        Q_GENERIC,
        {},
        "multilingual",
    ),
    (
        "english with greek symbols",
        {"request": "Compute the mean μ and variance σ of X, then P(|X-μ| > 2σ)."},
        Q_GENERIC,
        {},
        "english",
    ),
    # English prose that names someone in their own script: the name is annotation, not a request
    (
        "english prose, russian name",
        {"body": "Anton Pavlovich Chekhov (Russian: Антон Павлович Чехов) was a playwright."},
        Q_GENERIC,
        {},
        "english",
    ),
    (
        "english prose, name with IPA",
        {
            "body": "Vladimir Nabokov (Russian: Влади́мир Набо́ков [vlɐˈdʲimʲɪr nɐˈbokəf]) wrote "
            "Lolita and taught literature at Cornell for more than a decade."
        },
        Q_GENERIC,
        {},
        "english",
    ),
    (
        "english prose, greek name",
        {"body": "Eleftherios Venizelos (Greek: Ελευθέριος Βενιζέλος) served as prime minister."},
        Q_GENERIC,
        {},
        "english",
    ),
    (
        "english prose, hebrew name",
        {
            "body": "Amos Oz (Hebrew: עמוס עוז), born Amos Klausner, was an Israeli writer and "
            "professor of literature at Ben-Gurion University of the Negev in Beersheba."
        },
        Q_GENERIC,
        {},
        "english",
    ),
    (
        "german text",
        {
            "body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die "
            "Rechnung die nicht korrekt ist"
        },
        Q_GENERIC,
        {},
        "multilingual",
    ),
    ("explicit model", {"body": "anything"}, Q_GENERIC, {"model": "multilingual"}, "multilingual"),
    (
        "explicit model overrides script",
        {"body": "मुझसे दो बार"},
        Q_GENERIC,
        {"model": "english"},
        "english",
    ),
    ("explicit task", {"body": "x"}, Q_GENERIC, {"task": "typed_decisions"}, "typed-decisions"),
    ("explicit lang en", {"body": "मुझसे दो बार"}, Q_GENERIC, {"lang": "en"}, "english"),
    ("explicit lang de", {"body": "hello there"}, Q_GENERIC, {"lang": "de"}, "multilingual"),
    ("td workflow, auto OFF", {"body": "I was charged twice"}, Q_TD, {}, "english"),
    ("empty state", {}, Q_GENERIC, {}, "english"),
    ("none state", None, Q_GENERIC, {}, "english"),
]


@pytest.mark.parametrize(
    "state,questions,kwargs,want",
    [case[1:] for case in ROUTE_CASES],
    ids=[case[0] for case in ROUTE_CASES],
)
def test_route_decisions(state, questions, kwargs, want):
    assert Router().route(state, questions, **kwargs)["model"] == want


def test_auto_task_detection_is_opt_in():
    auto = Router(auto_task_detection=True)
    assert auto.route({"body": "I was charged twice"}, Q_TD)["model"] == "typed-decisions"
    assert auto.route({"body": "I was charged twice"}, Q_GENERIC)["model"] == "english"
    # explicit model still beats an auto-detected workflow
    assert auto.route({"body": "x"}, Q_TD, model="multilingual")["model"] == "multilingual"


def test_auto_workflow_reports_repo_as_a_string():
    # It used to hand back the raw (repo, subfolder) spec, which serialises to a JSON list (#205).
    auto = Router(auto_task_detection=True)
    decision = auto.route({"body": "I was charged twice"}, Q_TD)
    assert decision["repo"] == "convaiinnovations/laya/typed-decisions"
    assert (
        decision["repo"]
        == auto.route({"body": "I was charged twice"}, Q_TD, task="typed_decisions")["repo"]
    )
    standalone = Router(auto_task_detection=True, standalone_repos=True)
    assert standalone.route({"body": "x"}, Q_TD)["repo"] == "convaiinnovations/laya-typed-decisions"


def test_decision_payload_shape():
    decision = Router().route({"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC)
    assert decision["repo"] == "convaiinnovations/laya/multilingual"
    assert isinstance(decision["reason"], str) and decision["reason"]
    assert decision["detection"]["script"] == "devanagari"
    assert decision.model == "multilingual"
    assert isinstance(decision, dict)


def test_custom_default_is_used_for_letterless_states():
    assert Router(default="multilingual").route("12345", Q_GENERIC)["model"] == "multilingual"


# --------------------------------------------------------------------- unknown-Latin routing (#35, #203)


@pytest.mark.parametrize(
    "text",
    [
        "Gătește-mi o rețetă de sarmale de post pentru mâine.",
        "Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez.",
        "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę",
        "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım",
    ],
)
def test_unknown_latin_routes_to_multilingual(text):
    assert Router().route(text).model == "multilingual"


def test_undecided_reason_says_what_it_routed_on():
    reason = Router().route("Müşteriden iki kez ücret alındı ve para iadesi istiyor").reason
    assert "not identified" in reason


def test_short_english_and_identified_english_still_route_english():
    router = Router()
    assert (
        router.route("Please refund the duplicate charge on invoice 4411 today.").model == "english"
    )
    assert router.route("refund me").model == "english"
    assert (
        router.route("Please refund the duplicate charge on invoice 4411 today.").reason
        == "English Latin text"
    )


def test_undecided_latin_follows_default():
    stock = Router()
    multi = Router(default="multilingual")
    for text in [
        "Quero cancelar",
        "Esqueci minha senha",
        "Fui cobrado duas vezes",
        "Produto veio quebrado, quero trocar",
        "refund me",
    ]:
        assert multi.route(text).model == "multilingual", text
        assert stock.route(text).model == "english", text
    assert "using default (multilingual)" in multi.route("Esqueci minha senha").reason


def test_identified_english_ignores_a_non_english_default():
    multi = Router(default="multilingual")
    for text in [
        "Please refund the duplicate charge",
        "Please refund the duplicate charge on invoice 4411 today.",
    ]:
        assert multi.route(text).model == "english", text


# --------------------------------------------------------------------- unlisted scripts (#169)


@pytest.mark.parametrize(
    "text",
    [
        "ｱﾘｶﾞﾄｳ",
        "ㄆㄇㄈㄉ",
        "\U0001b000\U0001b001",
        "\U00020000\U00020001",
        "\ua960\ua961",
        "ᏣᎳᎩ",
        "ᠮᠣᠩᠭᠣᠯ",
        "ܫܠܡܐ",
        "ދިވެހި",
        "ⵜⴰⵎⴰⵣⵉⵖⵜ",
        "ꆈꌠ",
    ],
)
def test_unlisted_scripts_route_to_multilingual(text):
    assert Router().route(text)["model"] == "multilingual"


def test_letterless_states_keep_the_default():
    for text in ("", "12345 67890", "😀😀😀"):
        assert Router().route(text)["model"] == "english"


# --------------------------------------------------------------------- workflow signatures


def test_workflow_signatures():
    workflows = {
        "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
        "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
        "invoice_processing": [
            "discrepancy_severity",
            "disposition",
            "duplicate",
            "matches_order",
            "urgency",
        ],
        "security_incidents": [
            "credential_compromise",
            "disposition",
            "severity",
            "true_positive",
            "urgency",
        ],
    }
    for name, ids in workflows.items():
        assert match_typed_decisions_workflow({i: {} for i in ids}) == name
    assert match_typed_decisions_workflow({"urgency": {}, "category": {}}) is None
    assert (
        match_typed_decisions_workflow({**{i: {} for i in workflows["customer_service"]}, "x": {}})
        is None
    )
    assert match_typed_decisions_workflow({}) is None


# --------------------------------------------------------------------- name normalisation


@pytest.mark.parametrize(
    "alias,want",
    [
        ("en", "english"),
        ("laya", "english"),
        ("multi", "multilingual"),
        ("ML", "multilingual"),
        ("typed", "typed-decisions"),
        ("typed_decisions", "typed-decisions"),
        ("English", "english"),
        ("convaiinnovations/laya", "english"),
    ],
)
def test_name_aliases(alias, want):
    # "convaiinnovations/laya" ends in a known alias only after the last path segment stripping?
    # No: the alias table is keyed on the bare name, so the full id must be given as its tail.
    assert normalise_name(alias.split("/")[-1]) == want


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        normalise_name("nope")


# --------------------------------------------------------------------- bundle vs standalone


def test_bundle_and_standalone_maps():
    assert DEFAULT_MODELS["english"] == (BUNDLE_REPO, None)
    assert DEFAULT_MODELS["multilingual"] == (BUNDLE_REPO, "multilingual")
    assert DEFAULT_MODELS["typed-decisions"] == (BUNDLE_REPO, "typed-decisions")
    assert _repo_str((BUNDLE_REPO, None)) == "convaiinnovations/laya"
    assert _repo_str((BUNDLE_REPO, "multilingual")) == "convaiinnovations/laya/multilingual"
    assert _repo_str("some/repo") == "some/repo"
    assert sorted(STANDALONE_MODELS) == sorted(DEFAULT_MODELS)


def test_bundle_and_standalone_repos_in_decisions():
    bundle = Router()
    alone = Router(standalone_repos=True)
    assert (
        bundle.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"]
        == "convaiinnovations/laya/multilingual"
    )
    assert (
        alone.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"] == "convaiinnovations/laya-multilingual"
    )
    assert alone.route({"m": "I was charged twice"}, Q_GENERIC)["repo"] == "convaiinnovations/laya"


def test_local_path_override_is_kept():
    router = Router(models={"english": "/tmp/en", "multilingual": "/tmp/ml"})
    assert router.route({"m": "मुझसे दो बार"}, Q_GENERIC)["repo"] == "/tmp/ml"


# --------------------------------------------------------------------- LRU bookkeeping


class _Stub:
    def __init__(self, name):
        self.name = name

    def system_one(self, state, questions):
        return {"model": self.name, "answers": {}, "usage": {}}


def _load_stub(router, name):
    key = normalise_name(name)
    if key in router._agents:
        router._touch(key)
        return router._agents[key]
    router._agents[key] = _Stub(key)
    router._order.append(key)
    router._evict()
    return router._agents[key]


def _stubbed_router(max_loaded):
    router = Router(max_loaded=max_loaded)
    router.load = lambda n, _r=router: _load_stub(_r, n)
    return router


def test_lru_bookkeeping():
    router = _stubbed_router(1)
    router.load("english")
    router.load("multilingual")
    assert router.loaded == ["multilingual"]
    assert sorted(router._agents) == ["multilingual"]

    router = _stubbed_router(2)
    router.load("english")
    router.load("multilingual")
    router.load("typed-decisions")
    assert router.loaded == ["multilingual", "typed-decisions"]

    router = _stubbed_router(2)
    router.load("english")
    router.load("multilingual")
    router.load("english")  # touch english
    router.load("typed-decisions")
    assert sorted(router.loaded) == ["english", "typed-decisions"]

    router.unload("english")
    assert "english" not in router.loaded
    router.unload()
    assert router.loaded == []


# --------------------------------------------------------------------- default cap (#180)


def _counting_router(cap=None):
    """Router whose loader records which checkpoints it had to build."""
    router = Router() if cap is None else Router(max_loaded=cap)
    built = []

    def load(name, _r=router, _built=built):
        key = normalise_name(name)
        if key in _r._agents:
            _r._touch(key)
            return _r._agents[key]
        _built.append(key)
        _r._agents[key] = _Stub(key)
        _r._order.append(key)
        _r._evict()
        return _r._agents[key]

    router.load = load
    return router, built


def test_default_cap_is_two():
    assert Router().max_loaded == 2


def test_alternating_traffic_builds_per_cap():
    english = {"body": "I was charged twice for invoice 4411, please refund."}
    multilingual = {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung"}
    for cap, want_built in ((1, 20), (2, 2)):
        router, built = _counting_router(cap)
        for _ in range(10):  # the reported alternating workload
            router.predict(english, Q_GENERIC)
            router.predict(multilingual, Q_GENERIC)
        assert len(built) == want_built, "cap=%d" % cap


def test_single_language_traffic_builds_one_checkpoint():
    router, built = _counting_router()
    for _ in range(5):
        router.predict({"body": "I was charged twice for invoice 4411, please refund."}, Q_GENERIC)
    assert built == ["english"]


# --------------------------------------------------------------------- preload (#240, #27)


def test_preload_keeps_models_resident_and_honours_an_empty_selection():
    with patch("laya_mlx.agent.Agent", side_effect=lambda repo, **kw: _Stub(repo)) as build:
        router = Router(preload=True)
        assert sorted(router.loaded) == ["english", "multilingual", "typed-decisions"]
        assert router.max_loaded == 3
        assert build.call_count == 3
        assert router.preload() is router
        assert build.call_count == 3

        empty = Router()
        empty.preload([])
        assert empty.loaded == []
        assert build.call_count == 3

        subset = Router()
        subset.preload(["english", "multilingual"])
        assert sorted(subset.loaded) == ["english", "multilingual"]
        assert subset.max_loaded == 2
        subset.load("english")
        assert sorted(subset.loaded) == ["english", "multilingual"]
        assert build.call_count == 5

        incremental = Router()
        incremental.preload(["english"])
        english = incremental.load("english")
        incremental.preload(["multilingual"])
        assert incremental.loaded == ["english", "multilingual"]
        assert incremental.max_loaded == 2
        assert incremental.load("english") is english
        assert build.call_count == 7

        incremental.preload(["en", "english", "multi", "ml"])
        assert incremental.max_loaded == 2
        assert build.call_count == 7
        incremental.preload(["english", "typed-decisions"])
        assert sorted(incremental.loaded) == ["english", "multilingual", "typed-decisions"]
        assert incremental.max_loaded == 3
        for name in DEFAULT_MODELS:
            incremental.predict("hello", Q_GENERIC, model=name)
        assert build.call_count == 8

        attached = Router()
        original = _Stub("already-built")
        attached.attach("english", original)
        attached.preload(["multilingual"])
        assert attached.load("english") is original
        assert sorted(attached.loaded) == ["english", "multilingual"]
        assert attached.max_loaded == 2
        assert build.call_count == 9

        roomy = Router(max_loaded=5)
        roomy.preload(["en", "english", "multi"])
        assert roomy.max_loaded == 5
        assert build.call_count == 11


# --------------------------------------------------------------------- attach


def test_attach_registers_and_raises_the_cap_only_when_needed():
    router = _stubbed_router(1)
    sentinel = _Stub("already-built")
    router.attach("english", sentinel)
    assert router._agents["english"] is sentinel
    assert "english" in router.loaded
    assert router.max_loaded == 1
    router.attach("multilingual", _Stub("second"))
    assert router.max_loaded == 2
    router.unload("multilingual")
    # attaching then loading another must not evict the attached one
    _load_stub(router, "multilingual")
    assert sorted(router.loaded) == ["english", "multilingual"]
    assert router._agents["english"] is sentinel
    assert _stubbed_router(1).attach("en", _Stub("x")) is not None


# --------------------------------------------------------------------- language families route


@pytest.mark.parametrize(
    "text",
    [
        # plain-ASCII Romance (#178)
        "El pedido llego roto y nadie responde cuando escribo al soporte",
        "Il cliente e stato addebitato due volte e vuole un rimborso",
        "La fattura contiene un errore nell importo totale",
        "O cliente foi cobrado duas vezes e quer o dinheiro de volta",
        "Le client a ete facture deux fois et demande un remboursement",
        # accented Romance still routes on the diacritic rate
        "La facturación tiene un error y necesito una corrección urgente",
        "La fattura è sbagliata, devo avere un rimborso per il pagamento",
        "La commande est arrivée cassée et personne ne répond au support",
        # Brazilian support text (#202)
        "Boa tarde, gostaria de cancelar o plano",
        "Voce pode me mandar a nota fiscal?",
        "Deu erro 500 no endpoint de login depois do update",
        # romanized Bangla (#187)
        "amar kach theke duibar taka kata hoyeche, doya kore ferot din",
        "ami invoice er jonno duibar charge peyechi, refund chai",
        # Bengali script
        "আমার কাছ থেকে দুইবার টাকা কাটা হয়েছে",
        # plain-ASCII German (#130)
        "trage diesen termin in meinen kalender ein",
        "wie lautet die temperatur in fulda in hessen",
        "was ist die aktuelle zeit",
    ],
)
def test_non_english_families_route_to_multilingual(text):
    assert Router().route(text)["model"] == "multilingual"


@pytest.mark.parametrize(
    "text",
    [
        "The report by Smith et al. shows the de facto standard, e.g. the LA office and Rio",
        "Our MI5 and UN contacts discussed the DOS attack in LA last month",
        "Our Sao Paulo office still has not received the invoice",
        "My VC asked for the cap table and the invoice",
        "Nossa Cafe charged my card twice this month",
        "The Boa Vista branch reported an outage this morning",
        "The pra team will review the claim tomorrow",
        "Chai latte order was charged twice, please refund the extra amount",
        "The AR team says the ETA for the fix is Friday",
        "Ami Patel from the Koto office sent the invoice to Kore Ltd",
        "I would like to book a flight to Berlin tomorrow",
        "turn off smart lamp in den",
    ],
)
def test_english_controls_route_to_english(text):
    assert Router().route(text)["model"] == "english"


def test_dotted_tokens_route_to_english():
    router = Router()
    assert router.route({"url": "github.com", "email": "user@acme.com"}).model == "english"
    assert router.route("github.com acme.com").model == "english"
    assert (
        router.route({"body": "Please check example.com and acme.com for the invoice"}).model
        == "english"
    )
    assert router.route("build 1.2.3 on 12.30 with ratio 0.5").model == "english"
