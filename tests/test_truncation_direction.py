"""Regression: Agent.system_one preserves the newest conversation turn.

Ported from upstream ``laya/tests/test_truncation_direction.py`` (v0.3.20, #224). The public Agent
API advertises `state` as accepting a chronological conversation-turn list. `build_sequence`
defaults to `truncate_left=False` (`st[:room]`), which preserves the head and drops the tail -- for
a chronological list that means the newest turn is silently lost. For list-shaped state the agent
truncates from the left so the most recent intent survives. Strings and dicts are unaffected.

Upstream drives a torch `DecisionModel` through `object.__new__`; this port loads the tiny MLX
checkpoint instead, and its `build_sequence` has the same `truncate_left` / `state_ids` surface.
"""

from unittest.mock import patch

from laya_mlx import Agent
from laya_mlx.agent import build_sequence
from laya_mlx.common import serialize_state
from laya_mlx.prepared import PrefixCache


class _FakeTok:
    """A deterministic tokenizer: the same word always maps to the same id."""

    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 4, 2
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (len(w) % 90) for w in text.split() if w]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


class _RecordingTok(_FakeTok):
    """`_FakeTok` that remembers every text it was asked to encode."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        self.calls.append(text)
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        )


class _VocabTok(_FakeTok):
    """A tokenizer whose ids decode back to the words, for end-to-end prompt inspection."""

    def __init__(self):
        self.words = {}

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = []
        for word in text.split():
            if word not in self.words:
                self.words[word] = 10 + len(self.words)
            ids.append(self.words[word])
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def decode(self, ids):
        # The prompt carries special-token ids that never came from words, so decoding has to name
        # them instead of raising KeyError. Without this the helper crashes before the assertion
        # runs, which is what made this test look like a truncation failure.
        inverse = {value: word for word, value in self.words.items()}
        for token, value in (
            ("[PAD]", self.pad_token_id),
            ("[CLS]", self.cls_token_id),
            ("[SEP]", self.sep_token_id),
            ("[MASK]", self.mask_token_id),
        ):
            inverse.setdefault(value, token)
        return [inverse.get(i, "[?%d]" % i) for i in ids]


def _short_agent(tiny_checkpoint):
    """The tiny checkpoint with a small head/sequence budget, so truncation actually happens."""
    agent = Agent(tiny_checkpoint, dtype="float32")
    agent.cfg["max_len"] = 30
    agent.cfg["head_max_len"] = 12
    return agent


Q = {
    "t": "choice",
    "ins": "What action?",
    "crit": {"refund": "money back", "escalate": "manager", "hold": "wait"},
}
QUESTIONS = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}


def _state_prefix_len(tok, q, max_len, head_max_len):
    ref, _ = build_sequence(tok, "", q, max_len, head_max_len)
    return len(ref) - 1


def test_agent_list_vs_string_truncation_direction(tiny_checkpoint):
    """Agent must truncate left for lists, right for strings (via call inspection)."""
    agent = _short_agent(tiny_checkpoint)
    list_capture = {}
    string_capture = {}

    def fake_build(tok, state, q, max_len, head_max_len, truncate_left=False, state_ids=None):
        if isinstance(state, list):
            list_capture["truncate_left"] = truncate_left
        else:
            string_capture["truncate_left"] = truncate_left
        return build_sequence(
            tok,
            state,
            q,
            max_len,
            head_max_len,
            truncate_left=truncate_left,
            state_ids=state_ids,
        )

    with patch("laya_mlx.agent.build_sequence", side_effect=fake_build):
        agent.system_one(
            [{"role": "user", "content": "hi"}, {"role": "user", "content": "newest"}], QUESTIONS
        )
        agent.system_one("string state", QUESTIONS)

    assert list_capture["truncate_left"] is True
    assert string_capture["truncate_left"] is False


def test_agent_system_one_passes_truncate_left_for_list(tiny_checkpoint):
    agent = _short_agent(tiny_checkpoint)
    conversation = [{"role": "user", "content": "hello"}, {"role": "user", "content": "NEWEST"}]
    captured = {}

    def fake_build(tok, state, q, max_len, head_max_len, truncate_left=False, state_ids=None):
        captured["truncate_left"] = truncate_left
        captured["state"] = state
        return build_sequence(
            tok,
            state,
            q,
            max_len,
            head_max_len,
            truncate_left=truncate_left,
            state_ids=state_ids,
        )

    with patch("laya_mlx.agent.build_sequence", side_effect=fake_build):
        agent.system_one(conversation, QUESTIONS)

    assert captured["truncate_left"] is True, "list state must trigger truncate_left=True"
    assert isinstance(captured["state"], list)


def test_agent_system_one_string_state_default_truncation(tiny_checkpoint):
    agent = _short_agent(tiny_checkpoint)
    captured = {}

    def fake_build(tok, state, q, max_len, head_max_len, truncate_left=False, state_ids=None):
        captured["truncate_left"] = truncate_left
        return build_sequence(
            tok,
            state,
            q,
            max_len,
            head_max_len,
            truncate_left=truncate_left,
            state_ids=state_ids,
        )

    with patch("laya_mlx.agent.build_sequence", side_effect=fake_build):
        agent.system_one("just a string state", QUESTIONS)

    assert captured["truncate_left"] is False, "string state must keep default truncation"


def test_string_state_preserves_head():
    """A string state must still preserve the head (backward compatible)."""
    tok = _FakeTok()
    state = "HEADMARKERWORD " + "filler " * 20 + " TAILMARKER"
    state_ids = tok(state.replace("[MASK]", " "), add_special_tokens=False)["input_ids"]
    head_id = state_ids[0]
    tail_id = state_ids[-1]
    assert head_id != tail_id, "head and tail must have different ids"

    seq, _ = build_sequence(tok, state, Q, 30, 12)
    prefix = _state_prefix_len(tok, Q, 30, 12)
    kept = seq[prefix:-1]

    assert head_id in kept, "string state must preserve the head (default mode)"
    assert tail_id not in kept, "string state must drop the tail (default mode)"


def test_list_state_preserves_the_newest_turn():
    """The complement: for a chronological list the tail is what must survive."""
    tok = _VocabTok()
    state = [
        {"role": "user", "content": "OLDTURN " + "filler " * 20},
        {"role": "user", "content": "NEWESTTURN"},
    ]
    seq, _ = build_sequence(tok, state, Q, 30, 12, truncate_left=True)
    prefix = _state_prefix_len(tok, Q, 30, 12)
    kept = tok.decode(seq[prefix:-1])

    assert any("NEWESTTURN" in word for word in kept), kept
    assert not any("OLDTURN" in word for word in kept), kept


def test_cached_prefix_path_also_preserves_the_newest_turn(tiny_checkpoint):
    agent = _short_agent(tiny_checkpoint)
    agent._prefix_cache = PrefixCache()
    agent.tok = _VocabTok()
    state = [
        {"role": "user", "content": "OLDTURN " + "filler " * 20},
        {"role": "user", "content": "NEWESTTURN"},
    ]
    items, _ = agent.prepare(state, QUESTIONS)
    kept = agent.tok.decode(items[0]["ids"])

    assert any("NEWESTTURN" in word for word in kept), kept


def test_build_sequence_default_unchanged():
    tok = _FakeTok()
    state = "OLDFRONT " + "filler " * 20 + " NEWBACK"
    seq_default, _ = build_sequence(tok, state, Q, 30, 12)
    seq_explicit, _ = build_sequence(tok, state, Q, 30, 12, truncate_left=False)
    assert seq_default == seq_explicit, "default must remain truncate_left=False"


def test_state_ids_reuse_matches_inline_tokenization():
    tok_ref = _RecordingTok()
    state = {"subject": "Duplicate charge", "body": "x" * 400}
    seq_ref, markers_ref = build_sequence(tok_ref, state, Q, 300, 200)

    tok_shared = _RecordingTok()
    state_ids = tok_shared(serialize_state(state), add_special_tokens=False)["input_ids"]
    seq_shared, markers_shared = build_sequence(tok_shared, state, Q, 300, 200, state_ids=state_ids)

    assert seq_shared == seq_ref
    assert markers_shared == markers_ref
    assert tok_shared.calls.count(serialize_state(state)) == 1
