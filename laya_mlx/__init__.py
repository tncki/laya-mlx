"""Laya typed decisions on Apple silicon with MLX."""

from .agent import Agent, RLAgent, load
from .email import clean_email_body, email_state
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .router import DEFAULT_MODELS, RouteDecision, Router
from .shortlist import embed_fn_from_agent, predict_shortlist, shortlist_choice

__version__ = "0.3.0"
# The upstream Laya release this port tracks. laya-mlx keeps its own release line; these record
# compatibility, and tests/test_packaging.py keeps them in sync with pyproject.toml, README.md
# and the commit pinned in CI.
UPSTREAM_VERSION = "0.3.5"
UPSTREAM_COMMIT = "573e5b62696ba441230cd6be71d593331b5d23af"
__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    "shortlist_choice",
    "predict_shortlist",
    "embed_fn_from_agent",
    "detect_language",
    "detect_script",
    "is_english",
    "clean_email_body",
    "email_state",
    "email_questions",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "UPSTREAM_VERSION",
    "UPSTREAM_COMMIT",
]
