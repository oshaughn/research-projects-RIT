"""End-to-end contract test for Rimsky's real Asimov follow-up hook.

This test deliberately stops before submitting external HTCondor jobs.  It
does exercise both installed projects, the YAML files exchanged between them,
Asimov's ledger, RIFT pipeline discovery, and bootstrap-file resolution.
"""

import configparser
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

rimsky = pytest.importorskip("rimsky", reason="Rimsky requires Python >=3.12")

import asimov
from asimov.ledger import YAMLLedger
from asimov.utils import update
from rimsky.settings import PipelineSettings
from rimsky.sinks.gdb_samples import start_asimov
from rimsky.utils.asimov import add_event

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
environment = test

[general]
git_default = .
rundir_default = {project}/working
calibration = test
calibration_directory = test
webroot = pages/
logger = file
"""


def _initialise_asimov(project):
    project.mkdir()
    ledger_path = project / "ledger.yaml"
    config = configparser.ConfigParser()
    config.read_string(ASIMOV_CONFIG.format(project=project))
    asimov.config = config
    asimov.analysis.config = config
    asimov.event.config = config
    asimov.ledger.config = config
    YAMLLedger.create(location=ledger_path, name="rimsky-rift-e2e")
    ledger = YAMLLedger(location=str(ledger_path))
    update(ledger.data, {"pipelines": {"rift": {}}})
    asimov.current_ledger = ledger
    return ledger


def test_first_rimsky_result_creates_bootstrapped_rift_production(tmp_path):
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
    result.touch()

    frames = {}
    psds = {}
    for detector in settings.detectors:
        frame = tmp_path / "{}-RIMSKY-1456739148-4.gwf".format(detector)
        frame.touch()
        frames[detector] = [str(frame)]
        psd = tmp_path / "{}-psd.txt".format(detector)
        psd.touch()
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
