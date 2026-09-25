# Calling laya-mlx from another program

`laya-decide` is the contract to integrate against: one request in, one JSON object out. It exists
because the other entry point (`laya-mlx predict`) needs a questions file and a long argument list,
which is awkward to build from a shell-based agent harness.

```bash
laya-decide --preset triage --field answers.intent.choice < request.txt
refund
```

## The interface

```
laya-decide [TEXT ...] [--preset NAME] [--state-file FILE] [--field PATH]
            [--pretty] [--model M] [--device gpu|cpu] [--dtype ...] [--batch-size N]
```

| Argument | Behaviour |
| --- | --- |
| *(no text)* | reads the request from **stdin**. Prefer this: it avoids shell quoting problems with quotes, newlines and non-ASCII text. |
| `TEXT ...` | positional words, joined with spaces. |
| `--preset` | which ready-made question set to answer; default `router`. |
| `--state-file` | read the state as JSON from a file, for structured states (tickets, payloads). |
| `--field` | print only that dotted field, e.g. `answers.intent.choice`. Without it the whole result is printed as one JSON object. |
| `--model` | defaults to `$LAYA_MLX_MODEL`, then the Hub id. Point it at a local checkpoint to stay offline. |

Exit status is `0` on success and `2` on a usage error (including an unknown `--field`), with a
one-sentence message on stderr. stdout carries only the payload, so a caller can pipe it straight
into a JSON parser.

## Presets

| Preset | Answers |
| --- | --- |
| `triage` | `intent`, `is_urgent`, `frustration`, `refund_requested`, `churn_risk` |
| `email` | `category`, `is_spam`, `is_phishing`, `urgency`, `needs_reply` |
| `guard` | `jailbreak`, `prompt_injection`, `sensitive_data`, `harm_severity`, `topic` |
| `moderation` | `toxic`, `harassment`, `threat`, `spam`, `severity` |
| `router` | `difficulty`, `domain`, `needs_tools`, `is_sensitive` |

The same names work on the full CLI: `laya-mlx predict --preset guard --state "..."`.

## Three ways to route, with very different costs

Measured on an Apple M2 Pro with the multilingual checkpoint, per call:

| Approach | Cost | What it can decide |
| --- | --- | --- |
| `detect_language` / `detect_script` (pure Python) | **0.36 s**, of which 0.35 s is `import mlx`; the analysis itself is ~10 ms | script and language, so "which language is this" routing |
| `laya-decide --preset router` | **1.3 s**, of which ~1.2 s is loading the checkpoint and ~95 ms is the decision | `difficulty`, `domain`, `needs_tools`, `is_sensitive` |
| `POST /v1/systemone` to `laya-serve` | **77 ms** measured, checkpoint resident | the same, without paying the load per request |

So a language/script rule needs no model at all:

```bash
python -c "
import sys, laya_mlx as laya
d = laya.detect_language(sys.stdin.read())
print('multilingual' if not d['is_english'] else 'english', d['script'], d['language'])
"
```

Anything that needs the model pays the load on **every process start**, which is fine for a tool an
agent consults occasionally and wrong for a router on the request path. `laya-serve` is the fix:
the checkpoint stays resident and each request costs ~77 ms, measured on eight consecutive calls
(76–78 ms).

## Wiring it into a shell-capable harness

Both harnesses described here can only run shell commands, so the whole integration is one command
plus two environment variables. Set them once where the harness picks up its environment:

```bash
export LAYA_MLX_MODEL=/Users/jack/workspace/laya-mlx/models/laya-multilingual-mlx
export HF_HOME=/Users/jack/workspace/.hf-cache
export PATH="/Users/jack/workspace/.venvs/laya-mlx/bin:$PATH"
```

Use the absolute path to the entry point if the harness does not inherit `PATH`:

```bash
/Users/jack/workspace/.venvs/laya-mlx/bin/laya-decide --device cpu --preset guard --field answers.jailbreak.noul
```

Then the two integration shapes:

**laya as a tool** — the agent decides to consult it:

```bash
echo "$REQUEST" | laya-decide --device cpu --preset triage --field answers.intent.choice
```

**laya as a router** — every request passes through it:

```bash
DECISION=$(echo "$REQUEST" | laya-decide --device cpu --preset router)
SCORE=$(echo "$DECISION" | python -c "import json,sys; print(json.load(sys.stdin)['answers']['difficulty']['score'])")
if python -c "import sys; sys.exit(0 if float('$SCORE') >= 2 else 1)"; then
  call_strong_model "$REQUEST"     # DeepSeek
else
  call_cheap_model "$REQUEST"      # the other target
fi
```

`--device cpu` is deliberate for a server or harness: it keeps results identical to the validation
runs and avoids competing with anything else using the GPU.

## The resident HTTP service

`laya-decide` spawns a process per call, so it reloads the checkpoint every time — 1.3 s, of which
about 1.2 s is loading. `laya-serve` keeps the checkpoint resident and answers in **77 ms**
(measured, eight consecutive requests). Use it when laya is on the request path rather than
consulted occasionally.

It speaks TypeSafe Jev's `/v1/systemone` wire protocol, matching upstream's server: a client
written against Jev can point its `baseUrl` at this and keep working.

**Start it** (needs the `serve` extra: `pip install 'laya-mlx[serve]'`):

```bash
HF_HOME=/Users/jack/workspace/.hf-cache \
LAYA_MODEL_DIR=/Users/jack/workspace/laya-mlx/models \
LAYA_MODELS=multilingual LAYA_DEVICE=cpu LAYA_PORT=8123 \
/Users/jack/workspace/.venvs/laya-mlx/bin/laya-serve
```

**Check it, then use it:**

```bash
curl -s http://127.0.0.1:8123/health
# {"status":"ok","loaded":["multilingual"],"device":"cpu"}

curl -s -X POST http://127.0.0.1:8123/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state": {"message": "发票4411被重复扣款，请今天退款。"},
       "questions": {"dept": {"type": "choice", "instructions": "Which team?",
                              "criteria": {"billing": "refunds", "technical": "bugs"}}}}'
```

The response is the usual payload — `answers` (each with `confidence` and `answer_confidence`) plus
`usage`. A `model` field in the request may name a checkpoint (`english`, `multilingual`,
`typed-decisions`, or a published id such as `aac6fef/laya-multilingual-mlx`); anything else, such
as a Jev model id, is ignored and the router auto-selects.

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `LAYA_HOST` | bind address | `127.0.0.1` |
| `LAYA_PORT` | bind port | `8000` |
| `LAYA_MODEL_DIR` | directory of converted checkpoints to serve from disk (`<dir>/<name>`) | unset: Hub |
| `LAYA_MODELS` | comma list to preload (`english,multilingual,typed-decisions`); empty = all | all |
| `LAYA_PRELOAD` | build the checkpoints at startup rather than lazily | `1` |
| `LAYA_DEVICE` | MLX device (`cpu` or `gpu`) | auto |
| `LAYA_AUTO_TASK` | auto-route to the typed-decisions checkpoint | `0` |
| `LAYA_API_KEY` | if set, require `Authorization: Bearer <key>` | none |
| `LAYA_LOG_LEVEL` | uvicorn log level | `info` |
| `LAYA_THREADS` | accepted and ignored — it caps torch threads and this runs on MLX | n/a |

Requests are capped before tokenization: at most 64 questions, 50 000 state characters and 2 MiB of
body. `400` means the body was not a JSON object with a `questions` field, `401` a bad bearer
token, `413` a request over a cap, `422` a question that cannot be answered (the message names the
question), `500` an inference failure with no internal detail leaked.

### Three deliberate differences from upstream's server

- **`LAYA_HOST` defaults to `127.0.0.1`, not `0.0.0.0`.** Binding every interface publishes a
  decision endpoint to the whole network, and without `LAYA_API_KEY` it is unauthenticated. Serving
  a LAN is an explicit `LAYA_HOST=0.0.0.0` away.
- **`LAYA_MODEL_DIR` is new.** Upstream can only fetch checkpoints by name from the Hub — for the
  multilingual checkpoint that is the 1.3 GB PyTorch export. If a converted checkpoint already sits
  under `models/<name>`, this serves it from disk and stays offline.
- **`LAYA_THREADS` is accepted but ignored**, and says so with a `RuntimeWarning` at startup rather
  than silently doing nothing to an unchanged service file.

## Before you trust it

`difficulty` and `is_sensitive` are the model's own estimates, not ground truth. Measure them on a
sample of your real requests before routing production traffic by them, and remember that the
multilingual checkpoint is the weaker one on English-only suites — use the English checkpoint for
English workloads.
