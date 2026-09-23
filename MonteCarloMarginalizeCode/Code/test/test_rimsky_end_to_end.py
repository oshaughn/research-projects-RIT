"""Submission-level test for Rimsky's real Asimov follow-up hook.

The test exercises both installed projects, their exchanged YAML, Asimov's
ledger and configuration rendering, and RIFT's real input discovery and
conversion.  It replaces only the heavyweight pseudo-pipeline process and the
HTCondor scheduler boundary.
"""

import configparser
import os
import sys
from contextlib import chdir
from importlib.metadata import version
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import yaml
from packaging.version import Version

rimsky = pytest.importorskip("rimsky", reason="Rimsky requires Python >=3.12")

import asimov
from asimov.ledger import YAMLLedger
from asimov.utils import update
from rimsky.settings import PipelineSettings
from rimsky.sinks.gdb_samples import start_asimov
from rimsky.utils.asimov import add_event, build_and_submit

import RIFT.asimov.rift as rift_asimov
from RIFT.rimsky.integration import main


ASIMOV_CONFIG = """
[ledger]
location = ledger.yaml
engine = yamlfile

[project]
name = rimsky-rift-e2e
root = {project}

[logging]
level = info
directory = logs
location = logs/asimov.log

[pipelines]
environment = {environment}

[condor]
user = rimsky-test

[general]
git_default = .
rundir_default = {project}/working
calibration = test
calibration_directory = C01_offline
webroot = pages/
logger = file
"""


def _initialise_asimov(project):
    project.mkdir()
    ledger_path = project / "ledger.yaml"
    config = configparser.ConfigParser()
    config.read_string(
        ASIMOV_CONFIG.format(project=project, environment=sys.prefix)
    )
    asimov.config = config
    asimov.analysis.config = config
    asimov.event.config = config
    asimov.ledger.config = config
    rift_asimov.config = config
    YAMLLedger.create(location=ledger_path, name="rimsky-rift-e2e")
    ledger = YAMLLedger(location=str(ledger_path))
    update(ledger.data, {"pipelines": {"rift": {}}})
    asimov.current_ledger = ledger
    return ledger


def test_first_rimsky_result_creates_bootstrapped_rift_production(
    tmp_path, monkeypatch
):
    assert Version(version("asimov")) >= Version("0.7")

    sid = "S260305df"
    source = tmp_path / "rimsky.yaml"
    followup = tmp_path / "rift-followup.yaml"
    configured_path = tmp_path / "rimsky-rift.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "detectors": ["H1", "L1"],
                "channels": {"H1": "STRAIN", "L1": "STRAIN"},
                "output_dir": "online-output",
                "event_sink": {
                    "bilby_pipe_defaults": {
                        "minimum_frequency": 20,
                        "maximum_frequency": 1024,
                    },
                    "trigger_dependent_settings": {},
                    "prior_defaults": {},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    main(
        [
            str(source),
            str(followup),
            "--configured-rimsky",
            str(configured_path),
        ]
    )

    # Load the generated file through Rimsky itself. These are the defaults
    # which make Bilby run first and the sample sink enqueue RIFT afterwards.
    settings = PipelineSettings.from_yaml(configured_path)
    assert settings.event_sink.bilby_pipe_format == "full-submit"
    assert Path(settings.sample_sink.asimov_configuration) == followup
    assert Path(settings.asimovdir) == tmp_path / "asimov"

    result = (
        Path(settings.output_dir)
        / sid[1:5]
        / sid[5:7]
        / sid
        / "results_page"
        / "metafile.hdf5"
    )
    result.parent.mkdir(parents=True)
    posterior = np.zeros(
        2,
        dtype=[
            ("mass_1", "f8"),
            ("mass_2", "f8"),
            ("chirp_mass", "f8"),
            ("luminosity_distance", "f8"),
            ("phase", "f8"),
            ("iota", "f8"),
            ("spin_1x", "f8"),
            ("spin_1y", "f8"),
            ("spin_1z", "f8"),
            ("spin_2x", "f8"),
            ("spin_2y", "f8"),
            ("spin_2z", "f8"),
        ],
    )
    posterior["mass_1"] = [35, 36]
    posterior["mass_2"] = [30, 29]
    posterior["chirp_mass"] = [28, 27]
    posterior["luminosity_distance"] = [400, 420]
    posterior["iota"] = [0.5, 0.6]
    posterior["spin_1z"] = [0.1, 0.2]
    posterior["spin_2z"] = [-0.1, -0.2]
    with h5py.File(result, "w") as metafile:
        analysis = metafile.create_group("bilby-online")
        analysis.create_dataset("posterior_samples", data=posterior)

    frames = {}
    psds = {}
    for detector in settings.detectors:
        frame = tmp_path / "{}-RIMSKY-1456739148-4.gwf".format(detector)
        frame.touch()
        frames[detector] = [str(frame)]
        psd = tmp_path / "{}-psd.txt".format(detector)
        np.savetxt(psd, [[0, 1e-40], [1, 1e-40], [2, 1e-40]])
        psds[detector] = str(psd)

    ledger = _initialise_asimov(Path(settings.asimovdir))
    event_metadata = {
        "name": sid,
        "category": "online",
        "interferometers": settings.detectors,
        "data": {
            "segment length": 8,
            "channels": settings.channels,
            "data files": frames,
        },
        "likelihood": {"sample rate": 2048},
        "psds": psds,
        "priors": {
            "chirp_mass": {"minimum": 10, "maximum": 20},
            "mass_ratio": {"minimum": 0.1, "maximum": 1},
            "luminosity_distance": {"minimum": 10, "maximum": 5000},
            "a_1": {"minimum": 0, "maximum": 0.8},
            "a_2": {"minimum": 0, "maximum": 0.8},
        },
    }
    with patch("git.Repo", return_value=MagicMock()):
        add_event(Path(settings.asimovdir), event_metadata, ledger=ledger)
        start_asimov(
            event=sid,
            asimovdir=Path(settings.asimovdir),
            asimov_configuration=settings.sample_sink.asimov_configuration,
        )
        event = ledger.get_event(sid)[0]
        productions = [
            item for item in event.analyses if item.name == "rift-online"
        ]
        assert len(productions) == 1
        production = productions[0]
        pipeline = production.pipeline
    assert pipeline.__class__.__name__ == "Rift"
    assert pipeline._resolve_bootstrap_file() == str(result)
    assert production.meta["priors"]["chirp mass"]["maximum"] == 20
    assert production.meta["psds"] == {2048: psds}
    caches = pipeline._prepare_frame_caches()
    assert set(caches) == {"H1", "L1"}
    assert all(Path(cache).is_file() for cache in caches.values())

    # Exercise the same input-discovery and template-rendering hook used by
    # ``asimov manage build``.  Keep the installed RIFT scripts discoverable
    # when this test is launched via an explicit virtual-environment Python.
    monkeypatch.setenv(
        "PATH", "{}:{}".format(Path(sys.executable).parent, os.environ["PATH"])
    )
    project_dir = Path(settings.asimovdir)
    with chdir(project_dir), patch("asimov.git.time.sleep"):
        pipeline.before_config()

    for detector in settings.detectors:
        xml_psd = Path(production.xml_psds[detector])
        assert Path(xml_psd).is_file()
    assert set(production.meta["data"]["frame cache"]) == {"H1", "L1"}

    # Give build_dag the repository assets that ``asimov manage build`` stores
    # before submission.  The coinc file is replaced from the bootstrap below,
    # but its initial presence avoids any GraceDB access during this test.
    repository_dir = Path(event.repository.directory)
    if not repository_dir.is_absolute():
        repository_dir = project_dir / repository_dir
    category_dir = repository_dir / production.category
    category_dir.mkdir(parents=True, exist_ok=True)
    (category_dir / "coinc.xml").write_text("synthetic coinc\n")

    commands = []

    class SchedulerBoundary:
        def __init__(self, command, **kwargs):
            commands.append(command)
            executable = Path(command[0]).name
            if executable == "util_RIFT_pseudo_pipe.py":
                rundir = Path(production.rundir)
                rundir.mkdir(parents=True, exist_ok=True)
                dag = (
                    rundir
                    / "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag"
                )
                dag.write_text(
                    "# synthetic DAG emitted at the external RIFT boundary\n"
                )
                self.stdout = b"RIFT DAG prepared"
            elif executable == "condor_submit_dag":
                self.stdout = b"submitted to cluster 4242."
            else:
                raise AssertionError("unexpected external command: {}".format(command))

        def communicate(self):
            return self.stdout, None

    # Run Rimsky's real Asimov submission helper.  Only the heavyweight
    # pseudo-pipeline process and final scheduler process are replaced; PSD
    # conversion, config rendering, posterior reading, and bootstrap conversion
    # run for real against the synthetic files above.
    with chdir(Path(settings.asimovdir)), patch("asimov.git.time.sleep"), patch(
        "RIFT.asimov.rift.subprocess.Popen", SchedulerBoundary
    ):
        build_and_submit(event, production, ledger)

    bootstrap = category_dir / "rift-online_bootstrap.xml.gz"
    assert bootstrap.is_file()
    assert (category_dir / "coinc.xml").is_file()
    configuration = category_dir / "rift-online.ini"
    assert configuration.is_file()
    rendered = configuration.read_text()
    assert "fake-cache" in rendered
    assert str(Path(caches["H1"])) in rendered
    assert production.status == "running"
    assert production.job_id == 4242
    assert [Path(command[0]).name for command in commands] == [
        "util_RIFT_pseudo_pipe.py",
        "condor_submit_dag",
    ]
