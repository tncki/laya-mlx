# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-mlx.
"""HTTP server exposing Laya over TypeSafe Jev's ``/v1/systemone`` wire protocol.

Laya's ``predict()`` output is already schema-compatible with the Jev decision API -- ``choice`` /
``score`` / ``noul`` answers and a ``{input_tokens, output_tokens}`` usage block -- so a client
written against Jev can point its ``baseUrl`` at this server and keep working unchanged. All this
module adds is the HTTP surface Laya itself does not ship: a ``POST /v1/systemone`` route, an
optional bearer check, and a health probe.

It exists so a caller does not pay the checkpoint load on every request. Spawning a process per
decision costs ~1.3 s (about 1.2 s of it loading weights); with the checkpoints resident this
endpoint answers in ~95 ms. That is what makes routing *every* request through Laya practical.

Configuration is entirely via environment variables, using upstream's names so an existing systemd
unit or compose file works unchanged:

======================  ============================================  =========
env var                 meaning                                        default
======================  ============================================  =========
``LAYA_HOST``           bind address (loopback by default)             127.0.0.1
``LAYA_PORT``           bind port                                      8000
``LAYA_DEVICE``         MLX device for every checkpoint (gpu/cpu)      (auto)
``LAYA_PRELOAD``        build the checkpoints at startup, not lazily   1
``LAYA_MODELS``         comma list to preload (english,multilingual,   (all)
                        typed-decisions); empty = every checkpoint
``LAYA_THREADS``        accepted and ignored -- see below              (n/a)
``LAYA_MODEL_DIR``      directory of converted checkpoints to serve   (unset: Hub)
                        from disk instead of the Hub (``<dir>/<name>``)
``LAYA_AUTO_TASK``      auto-route to the typed-decisions checkpoint   0
``LAYA_API_KEY``        if set, require ``Authorization: Bearer <it>``  (none)
``LAYA_LOG_LEVEL``      uvicorn log level                              info
======================  ============================================  =========

Three deliberate differences from the upstream server:

* ``LAYA_HOST`` defaults to ``127.0.0.1``, not upstream's ``0.0.0.0``. Binding every interface
  would expose a decision endpoint to the whole network, and without ``LAYA_API_KEY`` it is
  unauthenticated. The intended caller is another process on the same machine; serving a LAN or a
  container neighbour is an explicit ``LAYA_HOST=0.0.0.0`` away.
* ``LAYA_THREADS`` has no equivalent here. Upstream caps ``torch.set_num_threads``; MLX runs its
  own thread pool and exposes no such knob. The variable is still accepted so an unchanged service
  file does not simply break, but it is reported as ignored at startup rather than silently doing
  nothing.
* The checkpoint ids a client may name include this port's published MLX exports
  (``aac6fef/*-mlx``) alongside upstream's ``convaiinnovations/*`` ids.

Heavy imports (fastapi, uvicorn, and the MLX router) are deferred into the functions that need
them, so ``import laya_mlx.serve`` stays cheap and touches no GPU.
"""

import hmac
import json
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# The three checkpoint names the router understands; used to decide whether a client's `model`
# field names a Laya checkpoint (honour it) or is some other Jev model id (ignore it and let the
# router auto-select).
_KNOWN_MODELS = {"english", "multilingual", "typed-decisions"}

# Guardrails for unauthenticated remote input. The state is tokenized once per question and
# collated into one tensor, so an unbounded body can exhaust memory; the single-worker pool means
# one large request would also starve /health.
MAX_QUESTIONS = 64
MAX_STATE_CHARS = 50000
MAX_BODY_BYTES = 2 * 1024 * 1024

# Public model ids accepted so a client can name a checkpoint. The upstream bundle root is
# deliberately absent on both sides: ``convaiinnovations/laya`` means "let the Router choose"
# rather than pinning the English checkpoint, and ``aac6fef/laya-mlx`` names the English export.
_PUBLISHED_MODEL_IDS = {
    "convaiinnovations/laya-multilingual": "multilingual",
    "convaiinnovations/laya-typed-decisions": "typed-decisions",
    "aac6fef/laya-mlx": "english",
    "aac6fef/laya-multilingual-mlx": "multilingual",
    "aac6fef/laya-typed-decisions-mlx": "typed-decisions",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_model(model: Optional[str]) -> Optional[str]:
    """Map a client's `model` field onto a Laya checkpoint, or None to auto-route."""
    if not model:
        return None
    published = _PUBLISHED_MODEL_IDS.get(str(model).strip().lower())
    if published is not None:
        return published
    from .router import normalise_name

    # normalise_name raises ValueError on anything that is not a known checkpoint or alias. A Jev
    # client's `model` field (e.g. "jev-1") is expected to miss; treat that as "no explicit
    # checkpoint" and let the router auto-select.
    try:
        key = normalise_name(model)
    except Exception:
        return None
    return key if key in _KNOWN_MODELS else None


def _resolve_port() -> int:
    """Port from LAYA_PORT, validated. Exits with a message instead of a traceback."""
    raw = os.environ.get("LAYA_PORT", "8000")
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError):
        raise SystemExit("invalid LAYA_PORT %r: must be an integer 1-65535" % (raw,))
    if not 1 <= port <= 65535:
        raise SystemExit("invalid LAYA_PORT %r: must be an integer 1-65535" % (raw,))
    return port


def _resolve_host() -> str:
    """Bind address from LAYA_HOST. Loopback by default: the intended caller is another process on
    this machine, and without LAYA_API_KEY the endpoint is unauthenticated."""
    return os.environ.get("LAYA_HOST", "127.0.0.1")


def _check_request_limits(state: Any, questions: Any) -> None:
    """Reject oversized inference requests before tokenization (413)."""
    from fastapi import HTTPException

    if not isinstance(questions, dict):
        raise HTTPException(status_code=400, detail="'questions' must be an object")
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(
            status_code=413,
            detail="too many questions (%d > %d)" % (len(questions), MAX_QUESTIONS),
        )
    try:
        state_len = len(state) if isinstance(state, str) else len(str(state))
    except Exception:
        state_len = MAX_STATE_CHARS + 1
    if state_len > MAX_STATE_CHARS:
        raise HTTPException(
            status_code=413,
            detail="state too large (%d > %d chars)" % (state_len, MAX_STATE_CHARS),
        )


async def _read_body_capped(request: Any) -> bytes:
    """Read the request body, refusing to buffer more than ``MAX_BODY_BYTES``.

    ``Content-Length`` cannot be the only gate. It is a value the client chooses, and under
    ``Transfer-Encoding: chunked`` it is absent altogether -- HTTP/2 and HTTP/3 have no such header
    at all -- so a request that simply omits it would be read into memory in full, whatever its
    size. The body is streamed here and abandoned as soon as it exceeds the cap, so the limit holds
    for every framing rather than only for clients that announce their length honestly.
    """
    from fastapi import HTTPException

    total = 0
    chunks = []
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            # Stop reading rather than draining the rest: the peer is already over the limit and
            # nothing further can make the request acceptable.
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _local_models(directory: str) -> Dict[str, str]:
    """Map checkpoint names onto converted checkpoints on disk, for offline serving.

    The port keeps converted checkpoints under ``models/<name>`` (the benchmark harnesses use the
    same layout). When ``LAYA_MODEL_DIR`` names such a directory, any checkpoint present there is
    served from disk instead of being fetched from the Hub -- which also avoids downloading the
    upstream PyTorch-format weights when an MLX export is already cached.
    """
    base = Path(directory).expanduser()
    return {name: str(base / name) for name in sorted(_KNOWN_MODELS) if (base / name).is_dir()}


def _warn_about_thread_limit() -> None:
    """Upstream's LAYA_THREADS caps torch intra-op threads. MLX has no equivalent knob."""
    if os.environ.get("LAYA_THREADS"):
        warnings.warn(
            "LAYA_THREADS is ignored: it caps torch intra-op threads, and this port runs on MLX, "
            "which manages its own thread pool. Remove it from the service environment or expect "
            "no effect.",
            RuntimeWarning,
            stacklevel=2,
        )


def build_router():
    """Build a Router from the environment, preloading unless told otherwise."""
    from .router import Router

    _warn_about_thread_limit()
    device = os.environ.get("LAYA_DEVICE") or None
    models_env = os.environ.get("LAYA_MODELS", "").strip()
    preload_names = [name.strip() for name in models_env.split(",") if name.strip()] or None
    models_dir = os.environ.get("LAYA_MODEL_DIR")
    models = _local_models(models_dir) if models_dir else None
    router = Router(
        models=models, device=device, auto_task_detection=_env_bool("LAYA_AUTO_TASK", False)
    )
    if _env_bool("LAYA_PRELOAD", True):
        router.preload(preload_names)
    return router


def create_app(router: Optional[Any] = None):
    """Build the FastAPI app. Pass a Router to inject one (tests); otherwise one is built from the
    environment (and preloaded) at app-creation time."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, Header, HTTPException, Request

    if router is None:
        router = build_router()
    api_key = os.environ.get("LAYA_API_KEY") or None

    # Inference is synchronous and a call takes tens to hundreds of milliseconds, so it must not
    # run on the event loop: one request would stall every other client, `GET /health` included.
    # One worker, because one forward pass at a time is what a single Agent wants; the Router
    # already guards checkpoint lifecycle.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")
    # Created on first request, not here: an `asyncio.Lock` binds to the loop that is running when
    # it is first awaited, and `create_app` may be called before that loop exists (module scope,
    # TestClient startup, a preload script).
    gate: Optional[asyncio.Lock] = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            # TestClient, embedded ASGI apps and process supervisors all need the executor to
            # drain when the app stops.
            pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="laya-mlx-serve",
        summary="Laya System-1 decisions over the TypeSafe Jev /v1/systemone protocol",
        lifespan=lifespan,
    )

    # Compared as bytes, not str. `hmac.compare_digest` raises TypeError when a str operand holds a
    # non-ASCII character, and Starlette decodes request headers as latin-1 -- so
    # `Authorization: Bearer s\xe9cret`, which is legal on the wire, made the comparison itself
    # raise. That surfaced as HTTP 500 plus a traceback, reachable by any unauthenticated client
    # with one byte. Encoding both sides first keeps the comparison constant-time and total: every
    # header a client can send now answers 401.
    expected_auth = ("Bearer " + api_key).encode("utf-8", "surrogateescape") if api_key else b""

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is None:
            return
        supplied = (authorization or "").encode("utf-8", "surrogateescape")
        if not hmac.compare_digest(supplied, expected_auth):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "loaded": router.loaded,
            "device": os.environ.get("LAYA_DEVICE") or "auto",
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        nonlocal gate
        _check_auth(authorization)
        # A declared length over the cap is rejected before anything is read; the streaming cap
        # below is what actually enforces it, for bodies that declare no length or understate it.
        if request.headers.get("content-length"):
            try:
                if int(request.headers["content-length"]) > MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="request body too large")
            except ValueError:
                pass
        raw = await _read_body_capped(request)
        try:
            # Every parse failure a client can cause is a ValueError: JSONDecodeError for
            # malformed/empty/truncated bodies, UnicodeDecodeError for invalid UTF-8. A broader
            # catch would also swallow ClientDisconnect and Starlette's own stream errors,
            # reporting a transport or server fault as the client's.
            body = json.loads(raw)
        except ValueError:
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(
                status_code=400, detail="request body must be an object with a 'questions' field"
            )
        state = body.get("state")
        questions = body["questions"]
        _check_request_limits(state, questions)
        model = _resolve_model(body.get("model"))
        if gate is None:
            gate = asyncio.Lock()
        try:
            # Laya's result is already Jev-shaped: {model, answers, usage, routing}. A Jev client
            # decodes `answers` and `usage` and ignores the rest.
            async with gate:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    pool, lambda: router.predict(state, questions, model=model)
                )
        except HTTPException:
            raise
        except ValueError as error:
            # Question validation errors name the question and what to fix: safe for clients.
            raise HTTPException(status_code=422, detail=str(error))
        except Exception:  # noqa: BLE001 -- never leak paths/weights/memory text to clients
            raise HTTPException(status_code=500, detail="inference failed")

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=_resolve_host(),
        port=_resolve_port(),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
