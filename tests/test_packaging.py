"""Packaging metadata must match what the package actually needs.

Ported from upstream (``laya/tests/test_packaging.py``, issue #34). Upstream parses
``pyproject.toml`` with regexes because its floor is Python 3.10 and ``tomllib`` only arrives in
3.11; this package requires >=3.11, so the real parser is used and the checks are stronger.

Upstream's ``transformers`` floor assertion is kept in spirit: the reference extra is what the
parity tests install, and it must be new enough to know ModernBERT.
"""

import json
import re
import tomllib
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from re import escape

import pytest

ROOT = Path(__file__).parents[1]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def version_tuple(text):
    return tuple(int(part) for part in text.split("."))


@pytest.fixture(scope="module")
def pyproject():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="module")
def project(pyproject):
    return pyproject["project"]


def test_requires_python_floor_is_declared(project):
    assert re.fullmatch(r">=\s*\d+\.\d+", project["requires-python"])


def test_no_classifier_advertises_python_below_the_floor(project):
    floor = version_tuple(project["requires-python"].lstrip(">= "))
    advertised = [
        version_tuple(value)
        for value in re.findall(
            r"Programming Language :: Python :: (\d+\.\d+)", "\n".join(project["classifiers"])
        )
    ]
    assert advertised, "advertise the specific Python versions that are supported"
    below = [".".join(str(p) for p in v) for v in advertised if v < floor]
    assert below == [], f"classifiers below requires-python {floor}: {below}"


def test_metadata_lives_only_in_pyproject():
    for duplicate in ("setup.py", "setup.cfg"):
        assert not (ROOT / duplicate).exists(), f"{duplicate} would duplicate pyproject metadata"
    assert (ROOT / "pyproject.toml").is_file()


def test_pyproject_version_matches_package_dunder_version(project):
    # Two declarations of the same fact: they must not drift.
    assert project["version"] == import_module("laya_mlx").__version__


def test_tracked_upstream_release_matches_the_ci_pin(pyproject):
    # laya-mlx keeps its own release line, so the upstream release it tracks is recorded
    # separately. Re-pinning upstream in CI must move this metadata with it.
    tracked = pyproject["tool"]["laya-mlx"]
    workflow = read(Path(".github", "workflows", "ci.yml"))
    assert tracked["upstream-commit"] in workflow, (
        "the upstream ref pinned in .github/workflows/ci.yml has drifted from "
        "[tool.laya-mlx].upstream-commit"
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", tracked["upstream-version"])


def test_tracked_upstream_release_matches_the_readme(pyproject):
    tracked = pyproject["tool"]["laya-mlx"]
    readme = read("README.md")
    assert f"upstream v{tracked['upstream-version']}" in readme, (
        f"README.md does not state the tracked upstream release {tracked['upstream-version']}"
    )
    assert tracked["upstream-commit"] in readme, (
        "README.md does not pin the tracked upstream commit"
    )


def test_package_exposes_the_same_tracked_upstream_release(pyproject):
    module = import_module("laya_mlx")
    tracked = pyproject["tool"]["laya-mlx"]
    assert module.UPSTREAM_VERSION == tracked["upstream-version"]
    assert module.UPSTREAM_COMMIT == tracked["upstream-commit"]


def test_own_version_is_not_a_copy_of_the_upstream_version(project, pyproject):
    # This is an independent port, not upstream's release. Reusing upstream's exact version
    # number would misrepresent the package on PyPI; keep the two lines distinct and record
    # the tracked release in [tool.laya-mlx] instead.
    upstream = pyproject["tool"]["laya-mlx"]["upstream-version"]
    assert project["version"] != upstream, (
        f"laya-mlx {project['version']} collides with the tracked upstream release {upstream}; "
        "record compatibility in [tool.laya-mlx] rather than renumbering the package"
    )


def test_changelog_documents_the_current_version(project):
    changelog = read("CHANGELOG.md")
    assert f"## {project['version']}" in changelog, (
        f"CHANGELOG.md has no entry for {project['version']}"
    )


def test_sdist_ships_the_documentation(project, pyproject):
    includes = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    for name in ("/README.md", "/CHANGELOG.md", "/LICENSE", "/NOTICE"):
        assert name in includes, f"{name} would be missing from the source distribution"


FP16_RESULTS = {
    "laya": "laya-mlx-float16.json",
    "laya-multilingual": "laya-multilingual-mlx-float16.json",
    "laya-typed-decisions": "laya-typed-decisions-mlx-float16.json",
}


def _short_latency_rows():
    """Rows of the short-input table in BENCHMARKS.md, split into cells."""
    rows = []
    for line in read("BENCHMARKS.md").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[1].isdigit() and cells[0] in FP16_RESULTS:
            rows.append(cells)
    return rows


def test_benchmarks_md_states_the_headline_latency_of_every_checkpoint():
    # Guard against the docs quietly losing a checkpoint or a row.
    assert {cells[0] for cells in _short_latency_rows()} == set(FP16_RESULTS)
    assert {int(cells[1]) for cells in _short_latency_rows()} == {1, 5, 10, 50}


@pytest.mark.parametrize("model,filename", sorted(FP16_RESULTS.items()))
def test_benchmarks_md_matches_the_committed_latency_json(model, filename):
    # The published table is the number people quote, and the JSON is the measurement. They must
    # agree: two research docs in docs/ quote older, superseded figures, which is exactly the
    # drift this test prevents from reaching BENCHMARKS.md.
    measured = {
        row["questions"]: row["end_to_end"]
        for row in json.loads(read(Path("benchmarks") / "results" / filename))["results"]
        if row["workload"] == "short"
    }
    rows = [cells for cells in _short_latency_rows() if cells[0] == model]
    assert rows, f"BENCHMARKS.md has no short-input row for {model}"
    for cells in rows:
        questions = int(cells[1])
        assert questions in measured, f"{filename} has no short-{questions} workload"
        p50, p95 = (float(value) for value in cells[5].split("/"))
        # The table rounds to two decimals, so the difference may be up to half a unit.
        assert p50 == pytest.approx(measured[questions]["p50_ms"], abs=0.005)
        assert p95 == pytest.approx(measured[questions]["p95_ms"], abs=0.005)


def test_mlx_is_pinned_to_apple_silicon_only(project):
    pin = next(dep for dep in project["dependencies"] if dep.startswith("mlx"))
    # Without the marker a plain `pip install laya-mlx` on Linux/Intel resolves mlx, which
    # has no wheel there, and fails the build instead of installing importable-free metadata.
    assert "platform_machine == 'arm64'" in pin and "sys_platform == 'darwin'" in pin


def test_reference_extra_pins_a_modernbert_capable_transformers(pyproject):
    reference = pyproject["project"]["optional-dependencies"]["reference"]
    floor = next(
        version_tuple(re.search(r"transformers>=(\d+\.\d+)", dep).group(1))
        for dep in reference
        if dep.startswith("transformers")
    )
    assert floor >= (4, 48), f"ModernBERT support starts in transformers 4.48, pinned {floor}"


def test_console_scripts_resolve_to_importable_callables(project):
    scripts = project["scripts"]
    assert scripts, "the CLI entry points are part of the public interface"
    for name, target in scripts.items():
        module_name, separator, attribute = target.partition(":")
        assert separator and attribute, f"{name} must use the module:callable form, got {target!r}"
        assert find_spec(module_name) is not None, f"{name} points at missing module {module_name}"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as error:  # entry point needs an optional extra at runtime
            pytest.skip(f"{module_name} is not importable with the installed extras: {error}")
        assert callable(getattr(module, attribute)), f"{target} is not callable"


def test_wheel_build_ships_the_package(pyproject):
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "laya_mlx" in packages


def test_ci_does_not_test_a_python_below_the_floor(project):
    workflow = read(Path(".github", "workflows", "ci.yml"))
    floor = version_tuple(project["requires-python"].lstrip(">= "))
    tested = [
        version_tuple(value) for value in re.findall(r'python-version:\s*"(\d+\.\d+)"', workflow)
    ]
    assert tested, "CI must pin the interpreter it tests"
    below = [".".join(str(p) for p in v) for v in tested if v < floor]
    assert below == [], f"CI tests Python below requires-python {floor}: {below}"


def test_ci_installs_the_extras_the_test_suite_needs(project):
    # The suite imports the reference model behind an importorskip and the demo extras behind
    # skips, so CI must install at least dev; those extras must also stay declared.
    workflow = read(Path(".github", "workflows", "ci.yml"))
    extras = project["optional-dependencies"]
    assert re.search(escape("[dev,reference,demo]"), workflow)
    for extra in ("dev", "reference", "demo", "benchmark"):
        assert extras.get(extra), f"the {extra} extra disappeared from pyproject.toml"
