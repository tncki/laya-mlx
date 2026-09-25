"""Checkpoint download regression tests; tiny local weights, no network required.

The Hub transport is the only thing replaced: file filtering uses
``huggingface_hub.utils.filter_repo_objects`` with the ``allow_patterns`` the runtime actually
passed, so tokenizer, weights, model construction and inference stay on the real code path.
"""

import shutil
from unittest.mock import patch

import pytest
from huggingface_hub.utils import filter_repo_objects

from laya_mlx import Agent, load

SUBFOLDERS = ("multilingual", "typed-decisions", "variants/english")


@pytest.fixture
def repo(tiny_checkpoint, tmp_path):
    """A Hub-shaped model repository: root checkpoint, bundled subfolders, non-runtime extras."""
    root = tmp_path / "repo"
    shutil.copytree(tiny_checkpoint, root)
    runtime = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    for subfolder in SUBFOLDERS:
        for name in runtime:
            target = root / subfolder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
    (root / "README.md").write_text("An unrelated model card")
    (root / "eval").mkdir()
    (root / "eval" / "results.json").write_text("{}")
    return root


@pytest.fixture
def runtime_files(tiny_checkpoint):
    return {
        p.relative_to(tiny_checkpoint).as_posix() for p in tiny_checkpoint.rglob("*") if p.is_file()
    }


@pytest.fixture
def expected(repo, questions):
    return Agent(repo, device="cpu", dtype="float32").predict("hello", questions)


def fake_hub(repo, destination, downloaded):
    """Patch ``laya_mlx.agent.snapshot_download`` with the Hub's real filtering semantics."""

    def snapshot(repo_id, **kwargs):
        files = sorted(p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file())
        selected = filter_repo_objects(
            files,
            allow_patterns=kwargs.get("allow_patterns"),
            ignore_patterns=kwargs.get("ignore_patterns"),
        )
        for name in selected:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / name, target)
            downloaded.append(name)
        return str(destination)

    return patch("laya_mlx.agent.snapshot_download", side_effect=snapshot)


@pytest.mark.parametrize("repo_id", ["convaiinnovations/laya", "test/custom-model"])
def test_default_load_downloads_only_runtime_files(
    repo, runtime_files, questions, expected, tmp_path, repo_id
):
    destination = tmp_path / "download"
    destination.mkdir()
    downloaded = []
    with fake_hub(repo, destination, downloaded) as download:
        agent = load(repo_id, device="cpu", dtype="float32", token="test-token")
    assert download.call_count == 1
    assert download.call_args.args[0] == repo_id
    assert download.call_args.kwargs["token"] == "test-token"
    assert set(downloaded) == runtime_files
    assert "README.md" not in downloaded
    assert "eval/results.json" not in downloaded
    assert agent.predict("hello", questions) == expected


@pytest.mark.parametrize("subfolder", SUBFOLDERS)
def test_each_subfolder_loads_independently(
    repo, runtime_files, questions, expected, tmp_path, subfolder
):
    destination = tmp_path / "download"
    destination.mkdir()
    downloaded = []
    with fake_hub(repo, destination, downloaded) as download:
        agent = load(
            "test/bundled-models",
            device="cpu",
            dtype="float32",
            subfolder=subfolder,
            token="test-token",
        )
    assert download.call_count == 1
    assert download.call_args.args[0] == "test/bundled-models"
    assert set(downloaded) == {f"{subfolder}/{name}" for name in runtime_files}
    assert all(name.startswith(f"{subfolder}/") for name in downloaded)
    assert agent.predict("hello", questions) == expected


@pytest.mark.parametrize("subfolder", [None, "multilingual", "variants/english"])
def test_local_paths_do_not_download(repo, questions, expected, subfolder):
    with patch("laya_mlx.agent.snapshot_download") as download:
        result = load(str(repo), device="cpu", dtype="float32", subfolder=subfolder).predict(
            "hello", questions
        )
    download.assert_not_called()
    assert result == expected
