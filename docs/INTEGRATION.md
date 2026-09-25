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

## Two ways to route, with very different costs

Measured on an Apple M2 Pro with the multilingual checkpoint, per call:

| Approach | Cost | What it can decide |
| --- | --- | --- |
| `detect_language` / `detect_script` (pure Python) | **0.36 s**, of which 0.35 s is `import mlx`; the analysis itself is ~10 ms | script and language, so "which language is this" routing |
| `laya-decide --preset router` | **1.3 s**, of which ~1.2 s is loading the checkpoint and ~95 ms is the decision | `difficulty`, `domain`, `needs_tools`, `is_sensitive` |

So a language/script rule needs no model at all:

```bash
python -c "
import sys, laya_mlx as laya
d = laya.detect_language(sys.stdin.read())
print('multilingual' if not d['is_english'] else 'english', d['script'], d['language'])
"
```

Anything that needs the model pays the load on **every process start**. That is fine for a tool an
agent consults occasionally and wrong for a router on the request path. The fix is a long-lived
process: either keep laya in-process in your own service, or ask for the HTTP service to be ported
upstream (`serve.py`) so the checkpoint stays resident and each request costs ~95 ms instead.

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

## Before you trust it

`difficulty` and `is_sensitive` are the model's own estimates, not ground truth. Measure them on a
sample of your real requests before routing production traffic by them, and remember that the
multilingual checkpoint is the weaker one on English-only suites — use the English checkpoint for
English workloads.
