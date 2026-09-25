"""Block-tiled exact local (sliding-window) attention for MLX ModernBERT.

Hypothesis
----------
``laya_mlx/model.py::attention_masks`` builds one dense ``[B, 1, L, L]`` boolean
sliding mask (``|i - j| <= local_attention // 2``, inclusive) and hands it to
``mx.fast.scaled_dot_product_attention`` over the FULL length. For a query block
``[start, stop)`` the only unmasked keys lie in ``[start - half, stop + half)``,
so slicing K/V (and the mask) to that overlap is mathematically identical while
doing far less work.

This module contains
1. ``BlockedLocalAttention`` -- a drop-in ``EncoderAttention`` replacement that
   only ever sees the dense boolean mask, so it preserves the shipped mask
   semantics exactly (including padded-query rows).
2. ``blocked_local_model`` -- builds a *second* module graph that shares the
   eager model's parameter arrays (no weight copy, no per-call branch on the
   eager path).
3. a serial isolated-op microbenchmark (``python -m
   experiments.engineering.local_blocked``) that reports dense vs blocked
   timings plus a bit-exactness check against dense SDPA.

Nothing under ``laya_mlx/`` is modified; this is an experiment only.
"""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from benchmarks.common import environment, save_json
from laya_mlx.model import DecisionModel, EncoderConfig

from .run_variants import measured

HOST_NOTE = (
    "This host is an Apple M2 Pro (16 GB, applegpu_g14s). Every number checked "
    "into benchmarks/results/ and experiments/engineering/ was produced on an "
    "M3 Max (128 GB, applegpu_g15s). Absolute latencies here are NOT comparable "
    "to the published tables and no published figure has been edited."
)


class BlockedLocalAttention(nn.Module):
    """Sliding-window attention that slices each query block's K/V overlap.

    ``source`` is the original ``EncoderAttention`` whose ``Wqkv``/``Wo``
    modules are reused by reference (no copy).
    """

    def __init__(self, source, window, block):
        super().__init__()
        if block < 1:
            raise ValueError("block must be a positive integer")
        self.num_heads = source.num_heads
        self.head_dim = source.head_dim
        self.base = source.base
        self.Wqkv = source.Wqkv
        self.Wo = source.Wo
        self.half = window // 2
        self.block = block

    def __call__(self, x, mask):
        b, length, _ = x.shape
        qkv = self.Wqkv(x).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=0)
        scale = self.head_dim**-0.5
        outputs = []
        for start in range(0, length, self.block):
            stop = min(length, start + self.block)
            low = max(0, start - self.half)
            high = min(length, stop + self.half)
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q[:, :, start:stop],
                    k[:, :, low:high],
                    v[:, :, low:high],
                    scale=scale,
                    mask=mask[:, :, start:stop, low:high],
                )
            )
        out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)
        return self.Wo(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


def blocked_local_model(agent, block=128, window=None):
    """Return a module graph sharing ``agent.model`` weights but with every
    sliding layer replaced by :class:`BlockedLocalAttention`."""
    enc_cfg = EncoderConfig.from_dict(agent.encoder_cfg)
    model = DecisionModel(enc_cfg, agent.cfg)
    model.load_weights(tree_flatten(agent.model.parameters()))
    window = enc_cfg.local_attention if window is None else window
    replaced = 0
    for layer in model.encoder.layers:
        if layer.attention_type == "sliding_attention":
            layer.attn = BlockedLocalAttention(layer.attn, window, block)
            replaced += 1
    model.eval()
    mx.eval(model.parameters())
    return model, replaced


def _isolated_case(batch, heads, head_dim, length, window, block, iterations, seed):
    mx.random.seed(seed)
    x = mx.random.normal((batch, heads, length, head_dim)).astype(mx.float16)
    positions = mx.arange(length)
    dense_mask = (mx.abs(positions[:, None] - positions[None, :]) <= window // 2)[
        None, None
    ].astype(mx.bool_)
    mx.eval(x, dense_mask)
    scale = head_dim**-0.5

    def dense():
        return mx.fast.scaled_dot_product_attention(x, x, x, scale=scale, mask=dense_mask)

    def blocked():
        outputs = []
        for start in range(0, length, block):
            stop = min(length, start + block)
            low = max(0, start - window // 2)
            high = min(length, stop + window // 2)
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    x[:, :, start:stop],
                    x[:, :, low:high],
                    x[:, :, low:high],
                    scale=scale,
                    mask=dense_mask[:, :, start:stop, low:high],
                )
            )
        return mx.concatenate(outputs, axis=2)

    reference = np.asarray(dense()).astype(np.float32)
    actual = np.asarray(blocked()).astype(np.float32)
    for _ in range(5):
        mx.eval(dense())
        mx.eval(blocked())
    dense_timing = measured(dense, iterations)
    blocked_timing = measured(blocked, iterations)
    return {
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "dtype": "float16",
        "length": length,
        "local_window": window,
        "block": block,
        "keys_per_block": block + 2 * (window // 2),
        "dense": dense_timing,
        "blocked": blocked_timing,
        "speedup_p50": dense_timing["p50_ms"] / blocked_timing["p50_ms"],
        "max_abs_diff": float(np.max(np.abs(actual - reference))),
        "bit_identical": bool(np.array_equal(actual, reference)),
        "elements": int(reference.size),
    }


def isolated_main():
    parser = argparse.ArgumentParser(description="isolated dense vs blocked sliding SDPA")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/engineering/local-blocked-microbench.json")
    )
    args = parser.parse_args()
    report = {
        "environment": environment(),
        "host_note": HOST_NOTE,
        "machine_facts": {
            "chip": "Apple M2 Pro",
            "unified_memory_gb": 16,
            "os": "macOS 26.6.2",
            "gpu_arch": "applegpu_g14s",
            "published_host": "Apple M3 Max / 128 GB / applegpu_g15s",
        },
        "op": "mx.fast.scaled_dot_product_attention, FP16, B=1, window=128 (inclusive half-window 64 -> 129 keys/query)",
        "method": "synchronized per-call wall time (p50), 5 warmups, fixed tensors per config",
        "results": [],
    }
    configs = [(16, 64), (12, 64)]  # (heads, head_dim): probe config, multilingual config
    for heads, head_dim in configs:
        for length in (512, 1024):
            for block in (64, 128, 256):
                row = _isolated_case(
                    batch=1,
                    heads=heads,
                    head_dim=head_dim,
                    length=length,
                    window=128,
                    block=block,
                    iterations=args.iterations,
                    seed=20260919,
                )
                report["results"].append(row)
                print(
                    f"H{heads} D{head_dim} L{length} block={block}: "
                    f"dense={row['dense']['p50_ms']:.4f} blocked={row['blocked']['p50_ms']:.4f} "
                    f"speedup={row['speedup_p50']:.3f} maxdiff={row['max_abs_diff']:.3e} "
                    f"bit={row['bit_identical']}",
                    flush=True,
                )
    save_json(args.output, report)
    print("wrote", args.output)


if __name__ == "__main__":
    isolated_main()
