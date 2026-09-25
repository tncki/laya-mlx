"""Language/script detection tests ported from upstream v0.3.20.

Spec source: ``laya/tests/test_router.py`` (script/English/latin-guess/dotted-token/unlisted
script/plain-ASCII sections) plus the commit-level regressions it covers (#130, #169, #177,
#178, #187, #202, #113). No model weights are loaded: ``laya_mlx.lang`` is dependency-free.
"""

import pytest

from laya_mlx.lang import (
    _STOP,
    analyse,
    detect_script,
    guess_latin_language,
    is_english,
    state_text,
)

# --------------------------------------------------------------------- script detection


@pytest.mark.parametrize(
    "text,want",
    [
        ("The customer was charged twice and wants a refund.", "latin"),
        ("Հայերեն", "armenian"),
        ("ՀԱՅԵՐԵՆ", "armenian"),
        ("։֊", "unknown"),
        # Azerbaijani schwa (U+0259) is Latin, not an unlisted script (#36)
        ("ə", "latin"),
        ("MÜŞTƏRİ İLƏ ƏLAQƏ SAXLAYIN", "latin"),
        ("Le client a été facturé deux fois et demande un remboursement.", "latin"),
        ("ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"),
        ("お客様は二重に請求されたため返金を希望しています。", "kana"),
        ("客户被重复扣款要求退款", "han"),
        ("고객이 두 번 청구되어 환불을 원합니다", "hangul"),
        ("تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"),
        ("வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"),
        ("С клиента дважды сняли деньги и он хочет возврат", "cyrillic"),
        ("ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"),
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
        # Azerbaijani: with and without diacritics (#36)
        ("Sifarisim gelmedi ve pulum geri qaytarilmadi, zehmet olmasa yoxlayin", False),
        ("Mən sizin xidmətinizdən razı deyiləm və pulumu geri istəyirəm", False),
        ("refund me", True),
        ("ग्राहक से दो बार शुल्क लिया गया", False),
        ("お客様は二重に請求されました", False),
        ("С клиента дважды сняли деньги", False),
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
        # Latin-script languages with no stopword list of their own (#35)
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


def test_known_gap_romanian_without_diacritics_still_reads_english():
    # Kept visible on purpose, as upstream does: short Romanian with no diacritics and an
    # English function word ("in") still reads as English. A real LID model is the fix.
    assert is_english("Care este ora in Tokyo?") is True


# --------------------------------------------------------------------- Latin language guess


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
        (
            "Zəhmət olmasa, sifarişim üçün pulu geri qaytarın, çünki məhsul gəlmədi",
            "az",
        ),
        # one shared function word is not enough to name a language
        (
            "Müştəridən iki dəfə pul alınıb və o, geri qaytarılmasını istəyir",
            None,
        ),
        # dotted capital I in Azerbaijani lowercases to i + a combining dot (#36)
        ("MÜŞTƏRİ İLƏ ƏLAQƏ SAXLAYIN VƏ PULU GERİ QAYTARIN", "az"),
        ("refund", None),
        ("Cât e ora acum la Tokyo", None),
    ],
)
def test_guess_latin_language(text, want):
    assert guess_latin_language(text) == want


def test_long_english_stays_en():
    assert (
        guess_latin_language(
            "Please refund the duplicate charge on invoice 4411 today because "
            "we have been waiting for three days and nobody has replied to us"
        )
        == "en"
    )


def test_shared_words_alone_name_nothing_but_do_not_block_a_distinctive_word():
    # `la`, `e`, `o` are claimed by several lists, so overlap alone names no language (#178).
    assert analyse("Cât e ora acum la Tokyo")["language"] is None
    assert guess_latin_language("La fattura contiene un errore nell importo totale") == "it"
    assert guess_latin_language("La factura tiene un error en el importe total") == "es"


# --------------------------------------------------------------------- dotted tokens (#177)


@pytest.mark.parametrize(
    "state",
    [
        {"url": "github.com", "email": "user@acme.com"},
        "github.com acme.com",
        {"links": ["example.com", "example.co.uk", "docs.readthedocs.io"]},
    ],
)
def test_dotted_tokens_are_identifiers_not_prose(state):
    # `com` is Portuguese ("with") and `o` its article, so domains used to score Portuguese.
    assert analyse(state)["language"] is None
    assert is_english(state) is True


@pytest.mark.parametrize(
    "text",
    [
        "build 1.2.3 on 12.30 with ratio 0.5",
        "Report by Smith et al., e.g. the U.S.A. office",
    ],
)
def test_dotted_versions_and_abbreviations_are_identifiers(text):
    assert is_english(text) is True


def test_masking_identifiers_keeps_the_prose_around_them():
    # A full stop ends a sentence rather than joining an identifier: the word keeps its letters.
    assert (
        guess_latin_language(
            "O cliente nao recebeu o produto, mas quer o dinheiro para a conta, veja example.com"
        )
        == "pt"
    )
    assert (
        guess_latin_language("O cliente nao recebeu o produto, mas quer o dinheiro para a conta.")
        == "pt"
    )


# --------------------------------------------------------------------- state flattening


def test_state_text_flattens_and_ignores_keys():
    assert "charged twice" in state_text({"body": "charged twice", "n": 3})
    assert "deep" in state_text({"a": {"b": ["deep"]}})
    assert "x" in state_text(["x", {"y": "z"}])
    assert state_text(None) == ""
    # keys must not drive detection: English keys around Hindi content stay non-English
    assert (
        analyse({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})["is_english"] is False
    )


def test_script_profile_shapes():
    assert analyse("Հայերեն")["script_profile"] == {"armenian": 1.0}
    assert analyse("Հայերեն abc")["non_latin_fraction"] == 0.7
    assert analyse("ələ")["script_profile"] == {"latin": 1.0}


# --------------------------------------------------------------------- CJK with Latin brands (#113)


@pytest.mark.parametrize(
    "text",
    [
        "我的 iPhone 15 Pro Max 订单还没到",
        "Amazonで買ったiPhoneが届かない",
        "Samsung Galaxy 주문이 아직 안 왔어요",
    ],
)
def test_cjk_with_latin_brand_names_is_not_english(text):
    assert is_english(text) is False
    assert analyse(text)["script"] != "latin"


def test_english_with_a_han_name_stays_english():
    # A capitalised run or a lone symbol is annotation, not the request itself.
    assert is_english("My name is 王小明 and my order is late") is True


@pytest.mark.parametrize(
    "text",
    [
        "Compute the mean μ and variance σ of X, then P(|X-μ| > 2σ).",
        "Anton Pavlovich Chekhov (Russian: Антон Павлович Чехов) was a playwright.",
        (
            "Vladimir Nabokov (Russian: Влади́мир Набо́ков [vlɐˈdʲimʲɪr nɐˈbokəf]) wrote Lolita "
            "and taught literature at Cornell for more than a decade."
        ),
        "Eleftherios Venizelos (Greek: Ελευθέριος Βενιζέλος) served as prime minister.",
        (
            "Amos Oz (Hebrew: עמוס עוז), born Amos Klausner, was an Israeli writer and professor "
            "of literature at Ben-Gurion University of the Negev in Beersheba."
        ),
    ],
)
def test_english_prose_with_foreign_names_stays_english(text):
    assert is_english(text) is True


# --------------------------------------------------------------------- unlisted scripts (#169)


@pytest.mark.parametrize(
    "text",
    [
        "ｱﾘｶﾞﾄｳ",  # halfwidth katakana
        "ㄆㄇㄈㄉ",  # bopomofo
        "\U0001b000\U0001b001",  # kana supplement
        "\U00020000\U00020001",  # CJK Ext-B
        "\ua960\ua961",  # hangul jamo ext-A
        "ᏣᎳᎩ",  # Cherokee
        "ᠮᠣᠩᠭᠣᠯ",  # Mongolian
        "ܫܠܡܐ",  # Syriac
        "ދިވެހި",  # Thaana
        "ⵜⴰⵎⴰⵣⵉⵖⵜ",  # Tifinagh
        "ꆈꌠ",  # Yi
    ],
)
def test_unclaimed_scripts_are_not_called_english(text):
    assert analyse(text)["is_english"] is False


def test_fullwidth_latin_is_latin_not_unlisted():
    assert detect_script("ＨＥＬＬＯ") == "latin"


@pytest.mark.parametrize("text", ["", "12345 67890", "😀😀😀"])
def test_letterless_states_are_still_unknown(text):
    assert analyse(text)["script"] == "unknown"
    assert is_english(text) is True


@pytest.mark.parametrize(
    "text,script",
    [
        ("please cancel my subscription", "latin"),
        ("Mein Konto wurde zweimal belastet, bitte erstatten Sie den Betrag", "latin"),
        ("यह एक हिंदी वाक्य है", "devanagari"),
        ("请取消我的订阅", "han"),
        ("ありがとう", "kana"),
        ("감사합니다", "hangul"),
        ("Мой аккаунт был списан дважды", "cyrillic"),
        ("تم خصم حسابي مرتين", "arabic"),
        ("Η χρέωση έγινε δύο φορές", "greek"),
        ("Իմ հաշիվը գանձվել է երկու անգամ", "armenian"),
    ],
)
def test_named_scripts_are_untouched(text, script):
    assert detect_script(text) == script


# --------------------------------------------------------------------- plain-ASCII Romance (#178)


@pytest.mark.parametrize(
    "lang,text",
    [
        ("es", "El pedido llego roto y nadie responde cuando escribo al soporte"),
        ("es", "Quiero cancelar mi plan y pedir un reembolso"),
        ("es", "La factura tiene un error en el importe total"),
        ("es", "Necesito que me devuelvan el dinero de la compra duplicada"),
        ("it", "Il cliente e stato addebitato due volte e vuole un rimborso"),
        ("it", "Voglio cancellare il mio abbonamento e chiedere un rimborso"),
        ("it", "La fattura contiene un errore nell importo totale"),
        ("pt", "O cliente foi cobrado duas vezes e quer o dinheiro de volta"),
        ("fr", "Le client a ete facture deux fois et demande un remboursement"),
        ("fr", "Je ne peux pas acceder a mon compte et j ai besoin d aide"),
    ],
)
def test_plain_ascii_romance_is_detected(lang, text):
    assert guess_latin_language(text) == lang
    assert is_english(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "The customer was charged twice and wants a refund for this invoice",
        "Please cancel my subscription and refund the duplicate charge today",
        "The report by Smith et al. shows the de facto standard, e.g. the LA office and Rio",
        "Our MI5 and UN contacts discussed the DOS attack in LA last month",
        "No refund was issued, so I am writing to you again about invoice 4411",
        "no refund no reply",
        "The son of the director filed a complaint about the duplicate invoice",
    ],
)
def test_english_must_not_move_for_plain_ascii_romance(text):
    assert is_english(text) is True


# --------------------------------------------------------------------- Brazilian Portuguese (#202)


@pytest.mark.parametrize(
    "text",
    [
        "Boa tarde, gostaria de cancelar o plano",
        "Voce pode me mandar a nota fiscal?",
        "Você pode me mandar a nota fiscal?",
        "Nao consigo fazer login no app",
        "Pix nao caiu na conta",
        "Gostaria de saber o prazo de entrega",
        "Estou esperando faz uma semana",
        "Vc pode cancelar pra mim?",
        "Deu erro 500 no endpoint de login depois do update",
        "Depois da atualizacao ninguem consegue logar",
        "Antes funcionava, agora deu pau",
        "Estava tudo certo ate a migracao",
        "Entao o sistema travou de novo",
    ],
)
def test_brazilian_support_text_is_detected(text):
    assert guess_latin_language(text) == "pt"


@pytest.mark.parametrize(
    "text",
    [
        "Our Sao Paulo office still has not received the invoice",
        "My VC asked for the cap table and the invoice",
        "Nossa Cafe charged my card twice this month",
        "The Boa Vista branch reported an outage this morning",
        "The pra team will review the claim tomorrow",
    ],
)
def test_portuguese_words_shared_with_english_do_not_move_english(text):
    assert is_english(text) is True


# --------------------------------------------------------------------- romanized Bangla (#187)


@pytest.mark.parametrize(
    "text",
    [
        "amar kach theke duibar taka kata hoyeche, doya kore ferot din",
        "ami invoice er jonno duibar charge peyechi, refund chai",
        "Ami ei product ta niye khub hotash, ekhon e cancel korte chai",
        "apnara keno amar call dhorchen na? ajke kichu ekta korun",
        "bhai amar account e login korte parchi na",
        "taka ekhono ferot paini, kobe pabo?",
        "order ta kobe asbe bolte parben?",
    ],
)
def test_romanized_bangla_is_detected(text):
    assert guess_latin_language(text) == "bn"
    assert is_english(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Chai latte order was charged twice, please refund the extra amount",
        "The AR team says the ETA for the fix is Friday",
        "Ami Patel from the Koto office sent the invoice to Kore Ltd",
        "Our AR and VR demo in Oi Bahia went well, the client wants a quote",
        "Take the age of the account into account before you refund",
    ],
)
def test_banglish_words_shared_with_english_do_not_move_english(text):
    assert is_english(text) is True


def test_bn_list_shares_no_word_with_another_list():
    shared = sorted(
        w for w in _STOP.get("bn", ()) for lg, words in _STOP.items() if lg != "bn" and w in words
    )
    assert shared == []


def test_romanian_with_ei_stays_romanian():
    assert guess_latin_language("Ei nu sunt de acord cu factura, vreau o corecție") == "ro"


# --------------------------------------------------------------------- plain-ASCII German (#130)


@pytest.mark.parametrize(
    "text",
    [
        "trage diesen termin in meinen kalender ein",
        "wie lautet die temperatur in fulda in hessen",
        "schalte das licht im wohnzimmer aus",
        "was ist die aktuelle zeit",
    ],
)
def test_plain_ascii_german_is_detected(text):
    assert guess_latin_language(text) == "de"


@pytest.mark.parametrize(
    "text",
    [
        "I would like to book a flight to Berlin tomorrow",
        "turn off smart lamp in den",
        "im so sorry, am an hour late, stuck in traffic",
    ],
)
def test_english_sharing_german_words_stays_english(text):
    assert is_english(text) is True


def test_spanish_and_french_distinctive_words_stay_evidence():
    assert guess_latin_language("que hora es en australia") == "es"
    assert guess_latin_language("baisse le volume du haut-parleur") == "fr"
