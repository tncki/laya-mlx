"""Laya typed decisions on Apple silicon with MLX."""

from .agent import Agent, RLAgent, load
from .common import (
    QTYPE_NAMES,
    QTYPES,
    answer_confidence,
    confidence_from_probs,
    ece_score,
    render_options,
)
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

__version__ = "0.4.0"
# The upstream Laya release this port tracks. laya-mlx keeps its own release line; these record
# compatibility, and tests/test_packaging.py keeps them in sync with pyproject.toml, README.md
# and the commit pinned in CI.
UPSTREAM_VERSION = "0.3.20"
UPSTREAM_COMMIT = "23a17522aa4942da6cce53a995a275760320b691"
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
    # Prompt and calibration helpers that upstream also exports. The training-only helpers
    # (proper_reward, td_lambda_targets) and the new product surfaces (hooks, serve, MCP, ONNX,
    # LangChain integrations, structured decide) are deliberately not part of this port.
    "QTYPES",
    "QTYPE_NAMES",
    "render_options",
    "answer_confidence",
    "confidence_from_probs",
    "ece_score",
    "UPSTREAM_VERSION",
    "UPSTREAM_COMMIT",
]
