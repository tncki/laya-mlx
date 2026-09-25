"""Regression tests for DecisionModel with a single valid option.

Ported from upstream `laya` (tests/test_decision_model.py, issues #96/#103): a `choice`
question carrying exactly one criterion used to raise "selected index k out of range"
inside the act head, because topk(2) has nothing to select for the second slot when only
one marker is valid.

Unlike the upstream file these tests need no PyTorch reference model, so they run in a
plain `pip install -e '.[dev]'` environment.
"""

import mlx.core as mx
import numpy as np
import pytest

from laya_mlx.model import DecisionModel, EncoderConfig

ENCODER = {
    "model_type": "modernbert",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 1,
    "local_attention": 16,
    "max_position_embeddings": 256,
}


def tiny_model(seed=0, head_layers=1, act_costs=None):
    mx.random.seed(seed)
    model = DecisionModel(
        EncoderConfig.from_dict(ENCODER),
        {"head_layers": head_layers, "act_costs": act_costs or {"escalate": 0.5}},
    )
    model.eval()
    return model


def forward(model, *, batch=1, seq=8, marker_mask=None, marker_pos=None, seed=0):
    """Run one batch; marker_pos defaults to the leading `marker_mask` slots."""
    mask = np.asarray(marker_mask)
    if marker_pos is None:
        marker_pos = np.tile(np.arange(mask.shape[1], dtype=np.int32), (batch, 1))
    rng = np.random.default_rng(seed)
    logits, action = model(
        mx.array(rng.integers(0, 128, (batch, seq)).astype(np.int32)),
        mx.array(np.ones((batch, seq), np.int32)),
        mx.array(np.asarray(marker_pos, dtype=np.int32)),
        mx.array(mask),
        mx.array(np.zeros(batch, np.int32)),
    )
    mx.eval(logits, action)
    return np.asarray(logits), np.asarray(action)


def test_single_option_question_does_not_crash():
    logits, action = forward(tiny_model(), marker_mask=np.ones((1, 1), bool))
    assert logits.shape == (1, 1)
    assert action.shape == (1, 2)
    assert np.isfinite(logits).all()
    assert np.isfinite(action).all()


def test_single_option_matches_the_equivalent_padded_two_slot_call():
    # Softmax over a single valid logit is 1.0 regardless of its value, so a one-marker
    # question must drive the act head exactly like the same question padded to two slots
    # with the second masked out: top1 == 1.0 and top1 - top2 == 1.0, the "fully decided"
    # signal of an unambiguous choice.
    model = tiny_model(seed=1)
    _, single = forward(model, marker_mask=np.ones((1, 1), bool))
    _, padded = forward(
        model,
        marker_mask=np.array([[True, False]]),
        marker_pos=np.array([[0, 0]], np.int32),
    )
    np.testing.assert_array_equal(single, padded)


def test_single_marker_softmax_is_exactly_one():
    logits, _ = forward(tiny_model(seed=2), marker_mask=np.ones((1, 1), bool))
    np.testing.assert_allclose(np.asarray(mx.softmax(mx.array(logits), axis=-1)), 1.0)


@pytest.mark.parametrize("head_layers", [0, 2])
def test_multi_option_question_is_unaffected(head_layers):
    logits, action = forward(
        tiny_model(seed=3, head_layers=head_layers),
        batch=2,
        seq=10,
        marker_mask=np.array([[1, 1, 1, 1], [1, 1, 0, 0]], bool),
    )
    assert logits.shape == (2, 4)
    assert np.isfinite(logits).all()
    assert np.isfinite(action).all()
