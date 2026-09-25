import hashlib
import importlib.util
import json
import random
import socket
import subprocess
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from laya_mlx.snake.game import DIRECTIONS, SnakeGame, hamiltonian_cycle
from laya_mlx.snake.policy import LayaPolicy, local_checkpoint

MONOSPACE_FONTS = (
    Path("/System/Library/Fonts/Menlo.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
)

# Rasterizing needs Pillow and rich (the `demo` extra) plus a monospace font. Record validation
# and sidecar writing deliberately need none of that, so only the render tests carry this.
requires_renderer = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None
    or importlib.util.find_spec("rich") is None
    or not any(font.is_file() for font in MONOSPACE_FONTS),
    reason="rendering needs the demo extra (Pillow and rich) and a monospace font",
)


def _recording(name):
    return Path(__file__).parents[1] / "benchmarks/results" / name


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("width,height", [(4, 4), (4, 5), (5, 4), (24, 16)])
def test_cycle_visits_every_cell_and_closes(width, height):
    cycle = hamiltonian_cycle(width, height)
    assert len(cycle) == len(set(cycle)) == width * height
    assert all(0 <= x < width and 0 <= y < height for x, y in cycle)
    assert all(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(cycle, cycle[1:] + cycle[:1])
    )


def test_collision_growth_tail_vacancy_and_board_clear():
    game = SnakeGame(4, 4, initial_length=4)
    game.body = deque([(1, 1), (1, 2), (0, 2), (0, 1)])
    game.food = (3, 3)
    assert game.legal_reason("LEFT") == "legal"
    assert game.legal_reason("DOWN") == "reverse"
    game.step("LEFT")
    assert game.alive and len(game.body) == 4
    game.step("LEFT")
    assert not game.alive and game.death_reason == "wall"

    full = SnakeGame(4, 4, initial_length=15)
    move = next(m for m in full.moves() if m.safe)
    assert full.step(move.direction)
    assert full.won and full.food is None and len(full.body) == 16 and full.score == 1
    assert full.moves() == []


@pytest.mark.parametrize("seed", range(12))
def test_arbitrary_shielded_choices_complete_board_without_starving(seed):
    game = SnakeGame(6, 6, seed=seed)
    rng = random.Random(seed + 100)
    last_food = 0
    # A safe action advances at least one cycle position and never passes food.
    for _ in range(game.capacity * (game.capacity - game.initial_length)):
        allowed = [m.direction for m in game.moves() if m.safe]
        assert allowed
        ate = game.step(rng.choice(allowed))
        assert game.alive and game.cycle_order_valid()
        assert len(game.body) == len(set(game.body))
        assert len(game.body) == game.initial_length + game.score
        assert game.ticks - last_food <= game.capacity
        if ate:
            last_food = game.ticks
        if game.won:
            break
    assert game.won


def test_seed_reproduces_foods_and_actions():
    first, second = SnakeGame(seed=71), SnakeGame(seed=71)
    for _ in range(100):
        direction = max((m for m in first.moves() if m.safe), key=lambda m: m.advance).direction
        first.step(direction)
        second.step(direction)
        assert first.snapshot() == second.snapshot()


def test_guard_preserves_raw_probabilities_and_reports_intervention():
    game = SnakeGame()
    safe = [m.direction for m in game.moves() if m.safe]
    unsafe = next(d for d in DIRECTIONS if d not in safe)
    probabilities = {d: 0.9 if d == unsafe else 0.1 / 3 for d in DIRECTIONS}

    class StubAgent:
        def predict(self, *_):
            return {
                "answers": {
                    "move": {"probabilities": probabilities},
                    "risk": {"noul": 0.97},
                    "food": {"noul": 0.92},
                },
                "usage": {"input_tokens": 100},
            }

    policy = LayaPolicy.__new__(LayaPolicy)
    policy.agent, policy.guarded = StubAgent(), True
    policy.prompt = "compact"
    result = policy.decide(game)
    assert result.proposed == unsafe and result.executed in safe and result.intervened
    assert result.probabilities is probabilities
    assert result.dead_end_risk == pytest.approx(0.03)
    policy.guarded = False
    assert policy.decide(game).executed == unsafe


def test_missing_local_model_fails_without_a_network_attempt(monkeypatch, tmp_path):
    attempts = []

    def forbidden(*_, **__):
        attempts.append(True)
        raise AssertionError("Network access attempted")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        local_checkpoint(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="Download it"):
        local_checkpoint("nonexistent-snake-demo-test/no-cache")
    assert attempts == []


def test_small_or_odd_boards_are_rejected():
    for shape in ((3, 4), (4, 3), (5, 5)):
        with pytest.raises(ValueError):
            SnakeGame(*shape)


@pytest.mark.parametrize("times", [(1, 1), (2, 1), (float("nan"),), (-1,)])
def test_recording_rejects_invalid_wall_clock_timestamps(tmp_path, times):
    from laya_mlx.snake.replay import load_record

    path = tmp_path / "record.jsonl"
    events = [{"type": "metadata", "format": "laya-snake-v1"}]
    events.extend({"type": "frame", "at": t} for t in times)
    path.write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="strictly increase"):
        load_record(path)


def test_recording_preserves_actual_timestamps_and_probabilities(tmp_path):
    from laya_mlx.snake.replay import load_record

    path = tmp_path / "record.jsonl"
    metadata = {"type": "metadata", "format": "laya-snake-v1"}
    frames = [
        {"type": "frame", "at": 0.019, "decision": {"probabilities": {"UP": 0.8721}}},
        {"type": "frame", "at": 0.119, "decision": {"probabilities": {"UP": 0.0342}}},
    ]
    path.write_text("\n".join(json.dumps(e) for e in [metadata, *frames]))
    assert load_record(path) == (metadata, frames)


@pytest.mark.parametrize("filename", ["snake-showcase.jsonl", "snake-fast.jsonl"])
def test_published_real_showcase_replays_every_board_and_action_exactly(filename):
    from laya_mlx.snake.replay import load_record

    path = _recording(filename)
    metadata, frames = load_record(path)
    settings = metadata["settings"]
    game = SnakeGame(
        settings["width"], settings["height"], settings["seed"], settings["initial_length"]
    )
    interventions = 0
    for frame in frames:
        assert frame["game"] == game.snapshot()
        decision = frame["decision"]
        probabilities = decision["probabilities"]
        assert set(probabilities) == set(DIRECTIONS)
        assert sum(probabilities.values()) == pytest.approx(1, abs=0.00021)
        assert decision["proposed"] == max(DIRECTIONS, key=probabilities.__getitem__)
        safe = [m.direction for m in game.moves() if m.safe]
        assert decision["executed"] in safe
        interventions += decision["intervened"]
        assert frame["stats"]["interventions"] == interventions
        game.step(decision["executed"])
        assert game.alive and game.cycle_order_valid()
    end = json.loads(path.read_text().splitlines()[-1])
    assert end["game"] == game.snapshot()
    assert end["summary"]["steps"] == end["summary"]["inference_calls"] == len(frames)
    assert end["summary"]["interventions"] == interventions


@requires_renderer
def test_replay_raster_paints_a_real_recorded_frame():
    from laya_mlx.snake.replay import TerminalRaster, load_record
    from laya_mlx.snake.ui import compose

    _, frames = load_record(_recording("snake-fast.jsonl"))
    entry = frames[0]
    canvas = compose(entry["game"], entry["decision"], {**entry["stats"], "replay": True})
    raster = TerminalRaster(canvas.width, canvas.height, width=640, height=480)
    image = raster.render(canvas)
    assert image.size == (640, 480)
    # Board, chrome and glyphs must actually be painted, not left as flat background.
    assert len(image.getcolors(maxcolors=1 << 24)) > 4


def test_still_export_sidecar_records_its_source_frame(tmp_path):
    # A still is as much a claim about a real run as a video is, so it carries the same
    # provenance: the recording hash, the exact source offset and the renderer fingerprint.
    from laya_mlx.snake.replay import write_sidecar

    recording = _recording("snake-fast.jsonl")
    output = tmp_path / "frame.png"
    args = SimpleNamespace(recording=recording, output=output, fps=30)
    sidecar = write_sidecar(args, {"model": {"name": "m"}}, start=1.0, end=1.0, kind="still")

    assert sidecar["output_kind"] == "still"
    assert sidecar["source_sha256"] == _sha256(recording)
    assert sidecar["source_start_seconds"] == sidecar["source_end_seconds"] == 1.0
    assert sidecar["video_frames"] == 1
    assert sidecar["video_fps"] is None
    assert sidecar["gif_seconds"] is None
    assert sidecar["renderer_source_sha256"]
    assert "interpolated" in sidecar["note"]
    assert "frame rate" not in sidecar["note"]
    assert json.loads((tmp_path / "frame.json").read_text()) == sidecar


def test_video_export_sidecar_records_rate_frames_and_gif(tmp_path):
    # No ffmpeg needed: this pins the video schema that the published clip sidecar uses.
    from laya_mlx.snake.replay import write_sidecar

    recording = _recording("snake-showcase.jsonl")
    output = tmp_path / "clip.mp4"
    args = SimpleNamespace(recording=recording, output=output, fps=30)
    sidecar = write_sidecar(
        args,
        {"model": {"name": "m"}},
        start=65.0,
        end=95.0,
        kind="video",
        frames=900,
        gif_seconds=15,
    )

    assert sidecar["output_kind"] == "video"
    assert sidecar["source_sha256"] == _sha256(recording)
    assert (sidecar["source_start_seconds"], sidecar["source_end_seconds"]) == (65.0, 95.0)
    assert sidecar["video_fps"] == 30
    assert sidecar["video_frames"] == 900
    assert sidecar["gif_seconds"] == 15
    assert "sampled at its stated frame rate" in sidecar["note"]
    assert (tmp_path / "clip.json").exists()


@pytest.mark.parametrize(
    "output, gif, message",
    [
        ("poster.png", "clip.gif", "encoded from the MP4"),
        ("clip.mp4", "clip.txt", "must end in .gif"),
    ],
)
def test_gif_export_rejects_an_impossible_combination(tmp_path, capsys, output, gif, message):
    # Both of these used to be accepted silently: a still output dropped the GIF entirely and
    # still exited 0, and the GIF path was never checked for being a file ffmpeg can write.
    from laya_mlx.snake.replay import main

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                str(_recording("snake-fast.jsonl")),
                "--output",
                str(tmp_path / output),
                "--gif",
                str(tmp_path / gif),
            ]
        )
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / gif).exists()


BLOCKED_DEMO_EXTRAS = """
import importlib.abc
import sys


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"rich", "PIL"}:
            raise ModuleNotFoundError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, Blocker())
from laya_mlx.snake.cli import main

for argv in (["--help"], ["benchmark", "--help"], ["export", "--help"]):
    try:
        main(argv)
    except SystemExit as exit_info:
        assert exit_info.code == 0, (argv, exit_info.code)
    print("PARSED", " ".join(argv))
"""


def test_snake_entry_points_parse_arguments_without_the_demo_extra():
    # `pip install laya-mlx` installs the `laya-snake` console script but not the demo extra it
    # needs to draw. Importing the CLI used to raise a bare ModuleNotFoundError, so even
    # `laya-snake --help` failed with a traceback.
    result = subprocess.run(
        [sys.executable, "-c", BLOCKED_DEMO_EXTRAS],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("PARSED") == 3, result.stdout


def test_replay_module_is_runnable_as_a_module():
    # `python -m laya_mlx.snake.replay` used to define its functions and exit 0 silently,
    # because the module had no `__main__` guard.
    result = subprocess.run(
        [sys.executable, "-m", "laya_mlx.snake.replay", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


@requires_renderer
def test_png_export_writes_a_sidecar_next_to_the_image(tmp_path):
    from laya_mlx.snake.replay import main

    output = tmp_path / "frame.png"
    recording = _recording("snake-fast.jsonl")
    assert (
        main(
            [
                str(recording),
                "--output",
                str(output),
                "--start",
                "1",
                "--width",
                "640",
                "--height",
                "480",
            ]
        )
        == 0
    )
    assert output.stat().st_size > 0
    sidecar = json.loads(output.with_suffix(".json").read_text())
    assert sidecar["output_kind"] == "still"
    assert sidecar["source_sha256"] == _sha256(recording)
    assert sidecar["source_start_seconds"] == 1.0
