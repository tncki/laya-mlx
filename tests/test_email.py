"""Email cleaning: a disclaimer footer must not delete the sender's actual request.

Ported from upstream v0.3.20 ``tests/test_email.py`` (which is a script suite). `_DISCLAIMER`
used to be applied to whole paragraphs, so any paragraph that merely *mentioned* boilerplate was
deleted outright. When the footer ran on without a blank line, the request went with it:

    clean_email_body("My account is locked.\\nThis email is confidential...\\nPlease unlock it.")
    # before: ''      <- the whole body, request included, was deleted
    # after:  'My account is locked. Please unlock it.'

Dropping the request is silent and severe; leaving one boilerplate line behind is neither, so the
cleaning errs towards keeping text.

The same asymmetry drives the sign-off tests below. `_SIGNATURE_MARKERS` allowed a whole
sentence behind the closing word, so an ordinary body line beginning with one of them was read
as the start of a signature and everything after it was cut.

The Portuguese/Spanish block (upstream #200) covers Gmail's `Em ... escreveu:`, Outlook's
`-----Mensagem original-----`, the `Atenciosamente` sign-off and the confidentiality footer.

The last block guards a different kind of silence. `email_questions` was defined twice, here and
in `presets.py`, and the package `__init__` re-exports the one from `presets`. Editing the copy
in `email.py` moved `laya_mlx.email.email_questions` and left `laya_mlx.email_questions` where it
was, with no test and no lint failing.
"""

import pytest

import laya_mlx
from laya_mlx import email as email_module
from laya_mlx import presets
from laya_mlx.email import clean_email_body, email_state

DISCLAIMER = "This email is confidential and intended solely for the named addressee."

# --------------------------------------------------------------- the request survives the footer
KEEP_THE_REQUEST = [
    (
        "My account is locked.\n%s\nPlease unlock it." % DISCLAIMER,
        "My account is locked. Please unlock it.",
    ),
    (
        "My account is locked\n%s\nPlease unlock it." % DISCLAIMER,
        "My account is locked Please unlock it.",
    ),
    ("Please unlock my account\n%s" % DISCLAIMER, "Please unlock my account"),
    (
        "RMA 5521 is still pending\n%s\nPlease advise." % DISCLAIMER,
        "RMA 5521 is still pending Please advise.",
    ),
    # Keep-ward trade-off, documented on purpose: a capitalised continuation of a
    # boilerplate sentence can be a real fragment ("...in error,\nPlease delete it."),
    # so it stays. Leaving one boilerplate line behind is harmless; dropping the
    # request is not.
    (
        "If you have received this message in error,\nPlease delete it.",
        "Please delete it.",
    ),
    # Documented boundary: an all-lowercase fused request ("locked\nthis email ...")
    # is indistinguishable from a wrapped boilerplate footer, so the first line is
    # still lost. Only a newline followed by an uppercase letter splits.
    (
        "my account is locked\n"
        "this email is confidential and intended solely for the named addressee.\n"
        "please unlock it.",
        "please unlock it.",
    ),
    ("My account is locked. %s" % DISCLAIMER, "My account is locked."),
]

# --------------------------------------------------------------- a pure footer is still removed
PURE_FOOTER_REMOVED = [
    ("My account is locked.\n\n%s" % DISCLAIMER, "My account is locked."),
    (
        "My account is locked.\n\nThis email and any files transmitted with it are\n"
        "confidential and intended solely for the named addressee.",
        "My account is locked.",
    ),
    (
        "Please reopen ticket 4411.\n\nIf you have received this message in error, delete it.",
        "Please reopen ticket 4411.",
    ),
]

# --------------------------------------------------------------- unrelated cleaning is unchanged
UNRELATED_CLEANING = [
    (
        "Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text",
        "Thanks for the update.",
    ),
    (
        "Hi team,\nCan you confirm the refund?\nRegards,\nAlice",
        "Hi team,\nCan you confirm the refund?",
    ),
    ("", ""),
]

# --------------------------------------------------------------- Portuguese and Spanish mail
# With English-only markers none of this was removed, and the quoted history below (a cancellation)
# reached the model next to the new message (a refund request).
PT_EMAIL = """Olá equipe,

Fomos cobrados duas vezes na fatura de março. Por favor, estornem a cobrança duplicada hoje.

Atenciosamente,
João Silva
Financeiro - ACME Ltda

Enviado do meu iPhone

Esta mensagem pode conter informações confidenciais. Se você recebeu esta mensagem por engano, favor apagá-la.

Em seg., 22 de set. de 2026 às 10:14, Suporte <suporte@x.com> escreveu:
> Olá João, recebemos seu chamado de cancelamento do plano Enterprise.
"""

PT_ES_MAIL = [
    (
        PT_EMAIL,
        "Olá equipe,\n\nFomos cobrados duas vezes na fatura de março. "
        "Por favor, estornem a cobrança duplicada hoje.",
    ),
    (
        "Segue o comprovante do pagamento.\n\n-----Mensagem original-----\n"
        "De: Maria <maria@acme.com>\nAssunto: cancelar contrato\nQueremos cancelar o contrato.",
        "Segue o comprovante do pagamento.",
    ),
    (
        "Segue o comprovante.\n\nDe: Maria <maria@acme.com>\nEnviado: segunda-feira\n"
        "Assunto: cancelar contrato\nQueremos cancelar o contrato.",
        "Segue o comprovante.",
    ),
    (
        "O acesso voltou, obrigado.\n\nEm seg., 22 de set. de 2026 às 10:14, Suporte Técnico <\n"
        "suporte@acme.com> escreveu:\n> texto antigo",
        "O acesso voltou, obrigado.",
    ),
    (
        "Bom dia,\nO boleto de março não chegou.\nObrigado,\nAna",
        "Bom dia,\nO boleto de março não chegou.",
    ),
    (
        "Hola,\nNo puedo acceder a mi cuenta desde ayer.\nSaludos,\nCarlos\n\n"
        "El lun, 22 sept 2026 a las 10:14, Soporte <soporte@x.com> escribió:\n> texto anterior",
        "Hola,\nNo puedo acceder a mi cuenta desde ayer.",
    ),
    (
        "Necesito la factura de marzo.\n\nSi usted ha recibido este mensaje por error, bórrelo.",
        "Necesito la factura de marzo.",
    ),
]

# --------------------------------------------------------------- ...without eating the request
PT_ES_KEPT = [
    "Preciso do contrato confidencial assinado até sexta.",
    "Oi,\nRecebi a resposta.\nObrigado pelo retorno, mas continua\nsem funcionar.",
    "Oi,\nEm resposta ao que você escreveu:\no pedido 4411 ainda não chegou.",
    "O valor é destinado exclusivamente ao pagamento do boleto. Podem confirmar?",
    "Preciso das férias.\nDe: 10/09 a 15/09\nPode aprovar?",
]

PT_ES_MORE = [
    (
        "Favor reenviar a nota fiscal.\n\nCaso tenha recebido esta mensagem por engano, "
        "notifique o remetente.",
        "Favor reenviar a nota fiscal.",
    ),
    (
        "Resolvido, pode fechar.\n\nEm qua., 24 de set. de 2026 às 09:02, Suporte\n"
        "Técnico <suporte@acme.com> escreveu:\n> texto antigo",
        "Resolvido, pode fechar.",
    ),
    (
        "Fixed, thanks.\n\nOn Wed, Sep 24, 2026 at 9:02 AM Support Team <\n"
        "support@acme.com> wrote:\n> old text",
        "Fixed, thanks.",
    ),
]

# --------------------------------------------------------------- Brazilian clients and footers
BOLETO = "Preciso da segunda via do boleto."
DEVICE_FOOTERS = [
    "Enviado do meu Galaxy",
    "Enviado do meu smartphone Samsung Galaxy.",
    "Enviado do Outlook para iOS",
    "Obter o Outlook para Android",
    "Enviado do Email para Windows",
    "Enviado do Yahoo Mail no Android",
    "Antes de imprimir, pense em sua responsabilidade e compromisso com o MEIO AMBIENTE.",
    "Pense no meio ambiente antes de imprimir este e-mail.",
]

OUTLOOK_HEADER_NO_ADDRESS = [
    "Enviado: sexta-feira, 19 de setembro de 2026 10:02",
    "Data: sexta-feira, 19 de setembro de 2026 10:02",
]

# ...and none of them may take the request with it
DEVICE_KEPT = [
    "Oi,\nSegue o pedido.\nEnviado do meu celular o comprovante ontem.",
    "Preciso das férias.\nDe: 10/09\nPara: 15/09\nPode aprovar?",
    "Relatório do evento.\nDe: João\nData: amanhã cedo\nPode confirmar?",
    "Antes de imprimir o boleto, confira o valor. Está errado.",
    "Hi,\nThe box is at the front desk.\nGet mail",
]

# ------------------------------------------- a sign-off word inside the body is not a sign-off
SHORT = "Hi,\n\nThanks for the quick reply.\nCould you refund invoice 4411 as well?"
SIGNOFF_WORD_KEPT = [
    SHORT,
    "Hello,\n\nWe were billed twice in March.\nThanks for looking into it.\n"
    "The duplicate is 49 EUR on invoice 4411.",
    "Hi team,\n\nOur account is locked.\nBest practice would be a manual unlock.\n"
    "Please unlock account 88213 today.",
]

# ------------------------------------------- real sign-offs are still cut (positive controls)
SIGNOFF_BODY = "Hi,\n\nPlease refund invoice 4411."
SIGNOFF_TAILS = [
    "Thanks,\nAnna",
    "Thanks!",
    "Best regards,\nAnna",
    "Kind regards",
    "Cheers, Anna",
    "Many thanks,\nAnna Meier",
    "Thank you,",
    "Thanks in advance,",
    "Sincerely,\nA. Meier",
    "Sent from my iPhone",
    "--\nAnna Meier\nSupport",
]

# ------------------------------------------- closings the case rule did not reach (#245)
# These were cut before #132 and are not now: `warmest` is not in the alternation, `and regards`
# is not one of its continuations, and `[A-Z]` is ASCII, so a name in any other script reads as a
# sentence. Each leaves the signature block in the body that the sign-off rule exists to remove.
SIGNOFF_TAILS_WIDER = [
    "Thanks and regards,\nAnna",
    "Thanks & Regards,\nAnna",
    "Warmest regards,\nAnna",
    "Warmest wishes,",
    "Regards, Łukasz",
    "Thanks, José",
    "Regards, Дмитрий",
]

# ...and the wider closing must not swallow a sentence that merely starts the same way
SIGNOFF_WIDER_KEPT = [
    "Hi,\n\nPlease refund 4411.\nThanks and the team will confirm it today.",
    "Hi,\n\nThe room is cold.\nWarmest setting still reads 18 degrees.",
]

# ------------------------------------------------- the word, without the disclaimer (#227)
# `confidential` was a bare substring of `_DISCLAIMER`, so any sentence that merely
# mentioned it was dropped. A one-sentence body that mentions it was deleted whole and
# the model was then scored on an empty state, silently. The Portuguese branches beside
# it were already tied to disclaimer phrasing for this reason; English now is too.
WORD_ONLY_KEPT = [
    "Is this confidential?",
    "What is your confidentiality policy?",
    "Please keep this confidential but process my refund.",
    "Please treat this as confidential.",
    "This is confidential - can you help?",
    "Is the attached document confidential?",
    "Confidential: I need a refund.",
    "What is the information policy for contractors?",
]

# ...while the real footers those branches exist for are still dropped
WORD_ONLY_DROPPED = [
    "This email is confidential and intended solely for the named addressee.",
    "This message is confidential and intended solely for the use of the individual to whom "
    "it is addressed.",
    "The information in this email is confidential and may be privileged.",
    "This email and any files transmitted with it are\n"
    "confidential and intended solely for the named addressee.",
]


@pytest.mark.parametrize("body,want", KEEP_THE_REQUEST + UNRELATED_CLEANING)
def test_clean_email_body(body, want):
    assert clean_email_body(body) == want


@pytest.mark.parametrize("body,want", PURE_FOOTER_REMOVED)
def test_pure_footer_is_removed_but_the_request_stays(body, want):
    assert clean_email_body(body) == want


@pytest.mark.parametrize("body", WORD_ONLY_DROPPED)
def test_standalone_footer_is_dropped(body):
    assert clean_email_body(body).strip() == ""


def test_body_is_never_emptied():
    assert clean_email_body("My account is locked. %s" % DISCLAIMER).strip() != ""
    assert clean_email_body("Is this confidential?").strip() != ""


def test_email_state_body_keeps_the_request():
    state = email_state("Locked out", "My account is locked. %s" % DISCLAIMER)
    assert state["body"] == "My account is locked."
    assert email_state("Duplicate charge", SHORT)["body"] == SHORT
    assert email_state("Question", "Is this confidential?")["body"] == "Is this confidential?"


@pytest.mark.parametrize("body,want", PT_ES_MAIL + PT_ES_MORE)
def test_portuguese_and_spanish_mail(body, want):
    assert clean_email_body(body) == want


@pytest.mark.parametrize("body", PT_ES_KEPT)
def test_portuguese_and_spanish_requests_are_kept(body):
    assert clean_email_body(body) == body


@pytest.mark.parametrize("footer", DEVICE_FOOTERS)
def test_device_footer_is_removed(footer):
    assert clean_email_body(BOLETO + "\n\n" + footer) == BOLETO


def test_outlook_download_line_is_removed():
    assert clean_email_body("Please resend the invoice.\n\nGet Outlook for iOS") == (
        "Please resend the invoice."
    )


@pytest.mark.parametrize("second", OUTLOOK_HEADER_NO_ADDRESS)
def test_outlook_header_without_address_is_cut(second):
    cleaned = clean_email_body(
        "Segue o comprovante.\n\nDe: Maria Souza\n%s\nPara: Suporte\n"
        "Assunto: cancelar contrato\n\nQueremos cancelar o contrato." % second
    )
    assert cleaned == "Segue o comprovante."


@pytest.mark.parametrize("body", DEVICE_KEPT)
def test_device_words_do_not_eat_the_request(body):
    assert clean_email_body(body) == body


@pytest.mark.parametrize("body", SIGNOFF_WORD_KEPT)
def test_signoff_word_in_the_body_keeps_the_request(body):
    assert clean_email_body(body) == body


@pytest.mark.parametrize("tail", SIGNOFF_TAILS + SIGNOFF_TAILS_WIDER)
def test_real_signoff_is_cut(tail):
    assert clean_email_body("%s\n\n%s" % (SIGNOFF_BODY, tail)) == SIGNOFF_BODY


@pytest.mark.parametrize("body", SIGNOFF_WIDER_KEPT)
def test_wider_closing_does_not_swallow_a_sentence(body):
    assert clean_email_body(body) == body


@pytest.mark.parametrize("body", WORD_ONLY_KEPT)
def test_the_word_confidential_is_not_a_disclaimer(body):
    assert clean_email_body(body) == body


def test_request_around_a_footer_survives():
    assert (
        clean_email_body(
            "My account is locked.\n"
            "This email is confidential and intended solely for the named addressee.\n"
            "Please unlock it."
        )
        == "My account is locked. Please unlock it."
    )
    assert (
        clean_email_body(
            "Please unlock it. This email is confidential and intended solely "
            "for the named addressee."
        )
        == "Please unlock it."
    )


def test_long_body_is_bounded_before_matching():
    # Upstream #253 bounds regex work at max_chars * 4 and still returns at most max_chars.
    huge = ("Please refund invoice 4411. " * 5000) + "x" * 100
    cleaned = clean_email_body(huge, max_chars=100)
    assert len(cleaned) <= 100
    assert "Please refund invoice 4411." in cleaned


# ------------------------------------------- email_questions has exactly one definition (#136)
def test_email_questions_has_one_definition():
    assert email_module.email_questions is presets.email_questions
    assert laya_mlx.email_questions is presets.email_questions
    assert email_module.email_questions() == laya_mlx.email_questions()
    assert (
        email_module.email_questions({"legal": "contracts"})["category"]["criteria"]
        == laya_mlx.email_questions({"legal": "contracts"})["category"]["criteria"]
    )
