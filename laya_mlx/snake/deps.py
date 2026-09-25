"""The Snake demo's optional dependencies, imported on demand.

``rich`` and Pillow ship in the ``demo`` extra, but the ``laya-snake`` console script is installed
by a plain ``pip install laya-mlx`` (the same is true for the ``laya_mlx.snake.replay`` module and
the benchmark subcommand). Everything reachable from argument parsing must therefore import
without those packages, and only the code that actually draws or encodes should require them.
Each helper below turns a bare ``ModuleNotFoundError`` into a message that names the extra.
"""

EXTRA = "demo"


def _raise(dependency, error):
    raise ModuleNotFoundError(
        f"The Snake demo needs {dependency}; install it with: pip install 'laya-mlx[{EXTRA}]'"
    ) from error


def require_rich():
    """Return ``(Console, Live, BG, compose, layout_size)`` from rich and the shared palette."""
    try:
        from rich.console import Console
        from rich.live import Live

        from .ui import BG, compose, layout_size
    except ModuleNotFoundError as error:  # pragma: no cover - depends on installed extras
        _raise("rich", error)
    return Console, Live, BG, compose, layout_size


def require_palette():
    """Return the shared terminal palette ``(BG, DIM, MUTED)`` used while rasterizing."""
    try:
        from .ui import BG, DIM, MUTED
    except ModuleNotFoundError as error:  # pragma: no cover - depends on installed extras
        _raise("rich", error)
    return BG, DIM, MUTED


def require_pillow():
    """Return ``(Image, ImageDraw, ImageFont)``, needed only by the PNG/MP4 rasterizer."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as error:  # pragma: no cover - depends on installed extras
        _raise("Pillow", error)
    return Image, ImageDraw, ImageFont
