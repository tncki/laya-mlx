# Changelog

`laya-mlx` keeps its own release line. It is an independent port, not an official Convai
Innovations release; the upstream Laya release each version tracks is recorded in
`[tool.laya-mlx]` in `pyproject.toml` and exposed as `laya_mlx.UPSTREAM_VERSION` /
`laya_mlx.UPSTREAM_COMMIT`.

## 0.3.0

Tracks upstream Laya **v0.3.5** (`573e5b62696ba441230cd6be71d593331b5d23af`).

### Fixed

- **Single-option decisions could crash the model.** Upstream guards `topk(2)` inside
  `DecisionModel.forward` for a question with exactly one valid option; this port relied only on
  the collate step padding every question to at least two marker slots, so calling
  `DecisionModel` directly with one marker raised `Cannot squeeze axis 1 with size 0`. The model
  now pads the missing second slot itself, giving the act head the same `top1 - top2 == 1.0`
  "fully decided" signal upstream produces, and matching the equivalent two-slot call exactly.
- **The replay path no longer drags optional dependencies into pure data validation.**
  `laya_mlx.snake.replay.load_record` imported Pillow and rich at module scope, so validating a
  recording required the `demo` extra. Both are now imported lazily, and the render path raises an
  actionable message naming the extra instead of a bare `ModuleNotFoundError`.
- **`pip install -e '.[dev]'` now runs the whole test suite.** Previously seven tests failed on a
  dev-only install because the rendering dependencies were missing. The tests that genuinely need
  Pillow now skip instead of failing when the `demo` extra is absent.
- **Still-frame exports now carry provenance.** `laya-snake export --output poster.png` wrote no
  sidecar, while the MP4 path wrote one, so a poster could not be checked against its recording.
  Both output kinds now share one sidecar builder; the still records its exact source offset and
  the video records its frame rate, frame count and GIF length.
- **`python -m laya_mlx.snake.replay` silently did nothing.** The module had no `__main__` guard,
  so it defined its functions and exited 0. It now runs, matching `laya-snake export`.
- **The installed `laya-snake` script could not run at all on a plain install.** `rich` and Pillow
  live in the `demo` extra, but the console script is installed by `pip install laya-mlx`; the CLI
  imported `rich` at module scope, so even `laya-snake --help` died with a bare
  `ModuleNotFoundError` traceback. The demo dependencies are now imported on demand through
  `laya_mlx.snake.deps`, which also consolidates the three copies of that error message. Argument
  parsing works without the extra, and running the demo reports
  `pip install 'laya-mlx[demo]'`. The same fix covers `laya-snake benchmark` and `laya-snake export`.
- **`--gif` was silently dropped for still exports.** `laya-snake export --output poster.png --gif
  clip.gif` exited 0 and produced no GIF, because a GIF is re-encoded from the rendered MP4. It is
  now rejected with an explanation, and a `--gif` path that is not a `.gif` file is rejected too.

### Added

- Upstream test parity: `tests/test_criteria.py`, `tests/test_decision_model.py`,
  `tests/test_download.py`, `tests/test_packaging.py` and `tests/test_local_e2e.py` are ported
  from upstream and rewritten as pytest tests.
- `tests/test_local_e2e.py` runs the real published checkpoints. It is marked `integration` and
  skips unless the weights are already in the local Hugging Face cache, so it never downloads
  during a normal test run.
- `tests/test_packaging.py` fails if the tracked upstream release drifts from the commit pinned in
  CI, from `README.md`, or from the constants in `laya_mlx`.
- Python version classifiers matching the declared `requires-python` floor.
- Block-tiled exact local attention, measured: `experiments/engineering/local_blocked.py` plus its
  isolated microbenchmark and paired complete-model runs. 1.10–1.14x end-to-end on 1024-token
  inputs with exact parity (0 logit error, 0 probability error, 16/16 argmax), independently
  replicated. Recorded in `docs/ENGINEERING_10X_RESEARCH.md` as a measured candidate; **not**
  promoted into the shipped runtime, because that is an opt-in product decision, not because it is
  unverified.

### Changed

- The replay rasterizer gained a smoke test that runs wherever the `demo` extra is installed, and
  the export sidecar schema is pinned by tests that need no ffmpeg.
- `docs/PERFORMANCE_RESEARCH.md` and `docs/MATH_10X_RESEARCH.md` carry correction notes: their
  latency tables were captured while the benchmark process was still writing artifacts, so their
  MLX figures (Laya FP16 short-1 13.421 ms) disagree with the committed JSON (17.746 ms). Their
  historical rows are kept and labelled superseded rather than rewritten.
- `tests/test_packaging.py` now fails if `BENCHMARKS.md`'s headline latency table drifts from the
  committed `benchmarks/results/*.json`, which is how that staleness went unnoticed.

## 0.2.0

Synced with upstream v0.3.5: temperature clamping, language routing fixes, a thread-safe router,
and opt-in embedding shortlist. Added the Snake demo, opt-in runtime optimizations, and PyPI
packaging.
