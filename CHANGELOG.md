# Changelog

`laya-mlx` keeps its own release line. It is an independent port, not an official Convai
Innovations release; the upstream Laya release each version tracks is recorded in
`[tool.laya-mlx]` in `pyproject.toml` and exposed as `laya_mlx.UPSTREAM_VERSION` /
`laya_mlx.UPSTREAM_COMMIT`.

## 0.4.0

Tracks upstream Laya **v0.3.20** (`23a17522aa4942da6cce53a995a275760320b691`), up from v0.3.5.

Upstream added a great deal between those releases — prediction hooks, `predict_batch`, an HTTP
server, an MCP stdio server, ONNX Runtime, LangChain/LangGraph integrations, structured
schema-driven decisions, a TileLang fast path, a TypeScript port and Docker packaging. Those are
new product surfaces rather than inference behaviour, and this port only mirrors inference, so they
are deliberately **not** ported. What is ported is the behaviour that changes what gets asked,
what gets answered, and how a request is routed.

### Changed

- The tracked upstream release moved from v0.3.5 to v0.3.20 across every place that records it:
  `[tool.laya-mlx]`, `laya_mlx.UPSTREAM_VERSION` / `UPSTREAM_COMMIT`, `README.md`, `README.zh-CN.md`,
  `BENCHMARKS.md`, CI's checkout ref, and `benchmarks/common.py:UPSTREAM_REVISION`. The test suite
  fails if those ever disagree.
- `laya-mlx`'s own version moves to 0.4.0; it stays on its own release line and does not reuse
  upstream's number.

### Added

- `laya-mlx predict --preset NAME` answers one of the ready-made question sets (`email`, `guard`,
  `moderation`, `router`, `triage`) without needing a questions file, matching upstream's CLI
  (#303). `--state-stdin` reads the state text from stdin.
- **`laya-decide`**, a second console script meant to be called *by another program*. It reads the
  request from stdin when given no positional text, prints one JSON object on stdout, and can print
  a single field with `--field answers.intent.choice` so a caller does not parse a nested payload.
  Usage errors exit `2` with one sentence on stderr and nothing but the payload on stdout, which is
  what a shell-based agent harness needs. See [docs/INTEGRATION.md](docs/INTEGRATION.md).
- `docs/INTEGRATION.md`: the integration contract, the preset reference, and the measured cost of
  each routing approach.
- **`laya-serve`**, a resident local HTTP service speaking TypeSafe Jev's `/v1/systemone` protocol,
  ported from upstream's `serve.py` (bumped from `serve` extra: `pip install 'laya-mlx[serve]'`).
  A process-per-decision caller pays ~1.3 s reloading the checkpoint; with the checkpoint resident
  the same decision answers in **77 ms** measured over eight consecutive requests. Same paths,
  status codes, request caps and bearer-token behaviour as upstream, with three documented
  differences: it binds `127.0.0.1` rather than `0.0.0.0`, it adds `LAYA_MODEL_DIR` so converted
  checkpoints can be served from disk instead of fetched from the Hub, and `LAYA_THREADS` is
  accepted but reported as ignored because MLX has no torch-style thread cap.

### Fixed

Behaviour fixes ported from upstream and verified against upstream's own tests for each module,
plus a cross-module differential comparison (165 checks, no differences).

**Email cleaning** (`laya_mlx/email.py`)

- Portuguese and Spanish replies, signatures and footers are cleaned (#200): `Em`/`El … escreveu:`
  quote headers gated on a date, `Mensagem original`/`encaminhada` and `Mensaje…` separators, `De:`
  only when it carries an address, Exchange `De: Name` plus `Enviado:`/dated `Data:` pairs, PT/ES
  sign-offs, and device/eco footers.
- A sign-off word inside the body is not a signature (#132, #245): the marker is now a closing word
  plus punctuation plus at most three capitalised name tokens, so `Thanks for the quick reply.`
  survives. Adds `warmest`, `thanks/and|& regards`, and non-ASCII/Cyrillic names.
- `confidential` is no longer matched as a bare substring (#227): the disclaimer branches are tied
  to a disclaimer noun and tail, so `Is this confidential?` survives while real footers still drop.
- A request fused into the same paragraph as a disclaimer is recovered (#115).
- `email_questions` has a single definition (#136): it now lives in `presets` and is re-exported, so
  `email.email_questions is presets.email_questions`.
- Regex and join work is bounded before truncation (#253).

**Shortlist** (`laya_mlx/shortlist.py`)

- Cosine similarities are clipped to `[-1, 1]` and an empty document matrix is guarded (#158).
- The embedding device is re-read on each call instead of captured at construction, so a device
  change is picked up (#145).

**Routing and language** (`laya_mlx/lang.py`, `laya_mlx/router.py`)

- Plain-ASCII German is identified as German instead of English (#130), and plain-ASCII Romance
  text routes to the multilingual checkpoint (#178) — `latin_profile` gained a shared-word evidence
  rule so a word claimed by several stop lists may count as evidence but cannot name a winner.
- Brazilian Portuguese support text is recognised (#202), romanized Bangla routes to multilingual
  (#187), and Azerbaijani is recognised with `ki` deliberately left out of its stop list (#36).
- CJK text carrying Latin brand names routes to the multilingual checkpoint (#113), and letters no
  script range claims are counted rather than treated as English (#169); fullwidth Latin and IPA
  count as Latin.
- A dotted token is treated as an identifier, not as prose (#179), and the text flattening work is
  bounded before truncation (#253).
- `Router` accepts a caller-supplied language hint (`lang_guess=`), which may be a code or a
  callable that abstains (#211); a blank or whitespace `lang` falls through to detection (#292);
  undecided Latin text goes to the router's `default` rather than always English (#203); and an
  auto-detected workflow reports its `repo` as a string (#205).
- Two checkpoints stay resident by default instead of one (#180), an explicit empty `preload`
  selection stays empty (#240), and an incremental `preload` preserves the models already resident
  (#27).
- `Router.predict` now passes the effective language down to `Agent.system_one(lang=...)`, with a
  fallback for agent-like objects that do not accept it.

**Prompts, criteria and results** (`laya_mlx/common.py`, `laya_mlx/agent.py`)

- A `noul` question accepts caller-supplied labels instead of the fixed `false`/`true` wording
  (#163).
- Boolean and upper-case criteria keys are normalised, so `{"True": …, "FALSE": …}` behaves like
  `{"true": …, "false": …}` (#146).
- A `noul` question ignores criteria it cannot read rather than including them (#249), and a
  question whose definition cannot be answered now names the question in the error (#183).
- Non-ASCII characters survive in non-string instructions instead of being escaped (#228).
- `truncate_left` no longer keeps the entire state when no room is left (#112), and a
  chronological conversation list keeps its newest turn (#224).
- The shared state is tokenized once per call and reused across questions (#109), and the unused
  `noul` entropy is not computed (#293).
- Every answer reports `answer_confidence` alongside the existing `confidence`, so a caller can
  gate across question types on one number (#126); zero-confidence entries are included in the ECE
  calculation instead of being dropped (#39).
- Invalid temperature entries now fall back to a neutral value with a warning instead of failing
  the load (#142), and per-language temperature overrides are supported (#258).
- An empty `HF_TOKEN` is treated as no token rather than sending an empty bearer header (#264).

Known intentional difference: this port still rejects a base calibration list that is not exactly
three values with a clear `ValueError`, where upstream proceeds and fails later with an
`IndexError`.

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
