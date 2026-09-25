"""Command-line prediction, checkpoint conversion, and harness-friendly decisions.

Two entry points are installed:

``laya-mlx``    the full CLI (``predict`` / ``convert``), for people driving the port directly.
``laya-decide`` a narrow, stable contract meant to be called *by another program* — an agent
                harness, a shell script, a hook. It reads the request text from stdin when no
                positional text is given, prints one JSON object on stdout, and can print a single
                field with ``--field`` so a caller does not have to parse a nested payload.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, presets
from .agent import DTYPES, Agent

# Ready-made question presets, matching the names upstream's CLI uses.
PRESETS = {
    "email": presets.email_questions,
    "guard": presets.guard_questions,
    "moderation": presets.moderation_questions,
    "router": presets.router_questions,
    "triage": presets.triage_questions,
}

DEFAULT_MODEL = "convaiinnovations/laya"


def _load_questions(args):
    if args.preset:
        return PRESETS[args.preset]()
    return json.loads(args.questions.read_text())


def _add_model_arguments(parser):
    parser.add_argument("--model", default=os.environ.get("LAYA_MLX_MODEL", DEFAULT_MODEL))
    parser.add_argument("--subfolder")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=16)


def _dig(payload, path):
    """Follow a dotted path such as ``answers.intent.choice`` into the result payload."""
    value = payload
    for part in path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            if part not in value:
                raise KeyError(
                    "%r is not in the result; available here: %s"
                    % (part, ", ".join(sorted(value)) or "(nothing)")
                )
            value = value[part]
        else:
            raise KeyError("cannot descend into %r with %r" % (value, part))
    return value


def decide_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="laya-decide",
        description="Answer a question preset about one piece of text and print JSON.",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="the request text; omit it to read from stdin (safer for quotes and non-ASCII)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="router",
        help="which ready-made question set to answer (default: router)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="read the state as JSON from this file instead of passing text",
    )
    parser.add_argument(
        "--field",
        help="print only this dotted field of the result, e.g. answers.intent.choice",
    )
    parser.add_argument("--pretty", action="store_true", help="indent the JSON output")
    parser.add_argument("--version", action="version", version=__version__)
    _add_model_arguments(parser)
    args = parser.parse_args(argv)

    if args.state_file is not None:
        state = json.loads(args.state_file.read_text())
    elif args.text:
        state = " ".join(args.text)
    else:
        state = sys.stdin.read()

    agent = Agent(
        args.model,
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        subfolder=args.subfolder,
        batch_size=args.batch_size,
    )
    result = agent.predict(state, PRESETS[args.preset]())
    if args.field:
        try:
            value = _dig(result, args.field)
        except KeyError as error:
            # A caller parsing stderr should get one sentence, not a traceback.
            parser.error(str(error).strip("'\""))
        print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    else:
        print(
            json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None),
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-mlx", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("predict", "convert"):
        sub = commands.add_parser(command)
        sub.add_argument("--model", default=DEFAULT_MODEL)
        sub.add_argument("--subfolder")
        sub.add_argument("--revision")
        sub.add_argument("--dtype", choices=DTYPES, default="float16")
        if command == "predict":
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument("--state", help="Plain text input")
            source.add_argument("--state-file", type=Path, help="JSON state file")
            source.add_argument(
                "--state-stdin", action="store_true", help="Read the state text from stdin"
            )
            questions = sub.add_mutually_exclusive_group(required=True)
            questions.add_argument("--questions", type=Path, help="JSON question definitions")
            questions.add_argument(
                "--preset",
                choices=sorted(PRESETS),
                help="answer a ready-made preset instead of a questions file",
            )
            sub.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
            sub.add_argument("--batch-size", type=int, default=16)
        else:
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "convert":
        from .convert import convert

        result = convert(
            args.model,
            args.output,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
        )
        print(json.dumps({"output": str(result), "dtype": args.dtype}))
    else:
        if args.state_stdin:
            state = sys.stdin.read()
        elif args.state is not None:
            state = args.state
        else:
            state = json.loads(args.state_file.read_text())
        questions = _load_questions(args)
        agent = Agent(
            args.model,
            device=args.device,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
            batch_size=args.batch_size,
        )
        print(json.dumps(agent.predict(state, questions), ensure_ascii=False, indent=2))
    return 0
