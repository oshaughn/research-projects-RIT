import importlib.metadata
import os
import pathlib
import shutil
import subprocess

import pytest


BLUEPRINT_DIR = pathlib.Path(__file__).with_name("blueprints")
SUPPORTED_SERIES = {"0.5"}
FUTURE_SERIES = {"0.6", "0.7"}
EVENT = "GW190426_190642"
RIFT_ANALYSIS = "rift-v5PHM-calmarg"


def _asimov_version():
    try:
        return importlib.metadata.version("asimov")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("asimov is not installed")


def _series(version):
    parts = version.split(".")
    return ".".join(parts[:2])


def _require_supported_asimov():
    version = _asimov_version()
    series = _series(version)
    if series in FUTURE_SERIES:
        pytest.skip(
            "RIFT Asimov CI is wired for this series, but the integration "
            "is currently validated only against Asimov 0.5"
        )
    if series not in SUPPORTED_SERIES:
        pytest.skip(
            "RIFT Asimov CI is currently validated only against Asimov 0.5 "
            f"(found {version})"
        )
    return version


def _require_htcondor():
    try:
        __import__("htcondor")
    except ModuleNotFoundError:
        message = "htcondor Python bindings are required for this Asimov path"
        if os.environ.get("RIFT_ASIMOV_REQUIRE_HTCONDOR"):
            pytest.fail(message)
        pytest.skip(message)


def _run(cmd, cwd, env):
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, (
        "command failed with status {}\n{}\n{}".format(
            result.returncode, " ".join(map(str, cmd)), result.stdout
        )
    )
    return result.stdout


def _tree_text(root):
    chunks = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            chunks.append(path.read_text(errors="ignore"))
        except OSError:
            pass
    return "\n".join(chunks)


def test_asimov_05_can_create_project_and_add_rift_event(tmp_path):
    version = _require_supported_asimov()
    _require_htcondor()
    asimov_cli = shutil.which("asimov")
    assert asimov_cli, "asimov CLI is not on PATH"

    # Import after the version gate so 0.6/0.7 API drift skips cleanly.
    from asimov.pipelines import known_pipelines
    from RIFT.asimov.rift import Rift

    assert "rift" in known_pipelines
    assert known_pipelines["rift"] is Rift

    project = tmp_path / "project"
    project.mkdir()
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "RIFT CI",
        "GIT_AUTHOR_EMAIL": "rift-ci@example.invalid",
        "GIT_COMMITTER_NAME": "RIFT CI",
        "GIT_COMMITTER_EMAIL": "rift-ci@example.invalid",
    })

    _run([asimov_cli, "init", f"RIFT Asimov CI {version}"], project, env)

    for blueprint in [
        "production-pe-o4b.yaml",
        "production-pe-priors.yaml",
        "GW190426_190642.yaml",
    ]:
        _run([asimov_cli, "apply", "-f", str(BLUEPRINT_DIR / blueprint)], project, env)

    for blueprint in [
        "get-data-o4b-production.yaml",
        "pe-configurator-standard.yaml",
        "bayeswave-psd-standard.yaml",
        "analysis_rift_SEOBNRv5PHM.yaml",
    ]:
        _run(
            [asimov_cli, "apply", "-f", str(BLUEPRINT_DIR / blueprint), "-e", EVENT],
            project,
            env,
        )

    state = _tree_text(project)
    assert EVENT in state
    assert RIFT_ANALYSIS in state
    assert "SEOBNRv5PHM" in state
    assert "pipeline" in state.lower()
    assert "rift" in state.lower()
