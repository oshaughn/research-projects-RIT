import importlib.metadata
import pathlib
import shutil
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[4]
TRAVIS_INPUTS = ROOT / ".travis" / "ref_ini"
SUPPORTED_SERIES = {"0.5"}
FUTURE_SERIES = {"0.6", "0.7"}


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


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class _Repository:
    def __init__(self, directory):
        self.directory = str(directory)


class _Event:
    def __init__(self, name, work_dir, repository):
        self.name = name
        self.work_dir = str(work_dir)
        self.repository = repository
        self.productions = []


class _Production:
    def __init__(self, event, name, rundir, ini_relpath, coinc_file):
        self.event = event
        self.name = name
        self.pipeline = "rift"
        self.category = "C01_offline"
        self.rundir = str(rundir)
        self.dependencies = []
        self.status = "ready"
        self.meta = {
            "scheduler": {
                "accounting group": "ligo.dev.o4.cbc.pe.rift",
                "pipeline": {
                    "internal-puff-transverse": True,
                    "cip-explode-jobs-auto-scale": 6,
                    "use-gwsignal": True,
                },
            },
            "waveform": {"approximant": "SEOBNRv5PHM"},
            "likelihood": {"assume": {}},
        }
        self._ini_relpath = ini_relpath
        self._coinc_file = coinc_file

    def get_meta(self, key):
        return self.meta.get(key)

    def set_meta(self, key, value):
        self.meta[key] = value

    def get_coincfile(self):
        return str(self._coinc_file)

    def get_configuration(self):
        return types.SimpleNamespace(ini_loc=self._ini_relpath)

    def get_psds(self, _format):
        return []


def test_asimov_05_rift_build_dag_uses_frozen_inputs(monkeypatch, tmp_path):
    _require_supported_asimov()

    from RIFT.asimov import rift as rift_module
    from RIFT.asimov.rift import Rift

    repo_dir = tmp_path / "repo"
    repo_category = repo_dir / "C01_offline"
    repo_category.mkdir(parents=True)
    ini_name = "rift-ci.ini"
    shutil.copyfile(TRAVIS_INPUTS / "GW150914.ini", repo_category / ini_name)
    coinc_file = tmp_path / "coinc.xml"
    shutil.copyfile(TRAVIS_INPUTS / "coinc.xml", coinc_file)

    event = _Event("GW190426_190642", tmp_path, _Repository(repo_dir))
    production = _Production(
        event=event,
        name="rift-v5PHM-calmarg",
        rundir=tmp_path / "rift-run",
        ini_relpath=ini_name,
        coinc_file=coinc_file,
    )
    event.productions.append(production)

    pipe = Rift.__new__(Rift)
    pipe.production = production
    pipe.category = production.category
    pipe.bootstrap = False
    pipe.logger = _Logger()

    monkeypatch.setattr(Rift, "before_build", lambda self: None)

    def fake_config_get(section, option):
        values = {
            ("condor", "user"): "rift-ci",
            ("general", "calibration"): "C01",
            ("pipelines", "environment"): str(tmp_path / "env"),
            ("rift", "environment"): str(tmp_path / "env"),
        }
        return values[(section, option)]

    monkeypatch.setattr(rift_module.config, "get", fake_config_get)

    calls = []

    class FakePopen:
        def __init__(self, command, stdout=None, stderr=None):
            self.command = command
            calls.append(command)

        def communicate(self):
            rundir = pathlib.Path(self.command[self.command.index("--use-rundir") + 1])
            rundir.mkdir(parents=True)
            dag = rundir / "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag"
            dag.write_text("# frozen Asimov/RIFT build contract\n")
            return b"fake util_RIFT_pseudo_pipe.py completed", None

    monkeypatch.setattr(rift_module.subprocess, "Popen", FakePopen)

    pipe.build_dag()

    assert len(calls) == 1
    command = calls[0]
    assert command[0].endswith("/bin/util_RIFT_pseudo_pipe.py")
    assert command[command.index("--use-coinc") + 1] == str(coinc_file.resolve())
    assert command[command.index("--use-ini") + 1] == str((repo_category / ini_name).resolve())
    assert command[command.index("--approx") + 1] == "SEOBNRv5PHM"
    assert command[command.index("--use-rundir") + 1] == production.rundir
    assert "--internal-puff-transverse" in command
    assert "--cip-explode-jobs-auto-scale=6" in command
    assert "--use-gwsignal" in command
