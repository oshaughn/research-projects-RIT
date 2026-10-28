"""Scheduler-free real configuration -> builder lifecycle, including provenance.

The Production below supplies only persistence/lookup interfaces. No pipeline
hook, provenance writer, builder subprocess, directory guard or generated file is
mocked. Run with the branch installed (including its bin scripts) in the selected
Asimov 0.8 environment. These tests generate workflows; they never submit jobs.
"""
from contextlib import contextmanager
import hashlib
import configparser
import importlib.metadata
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

CODE = Path(__file__).resolve().parents[2]
ROOT = CODE.parents[1]
DAG = "marginalize_intrinsic_parameters_BasicIterationWorkflow.dag"


def _snapshot(root):
    """Include bytes, link targets, modes and mtimes, without following links."""
    result = {}
    for entry in [root, *sorted(root.rglob("*"))]:
        stat = entry.lstat()
        content = os.readlink(entry) if entry.is_symlink() else (
            hashlib.sha256(entry.read_bytes()).hexdigest() if entry.is_file() else None)
        result[str(entry.relative_to(root))] = (stat.st_mode, stat.st_mtime_ns, content)
    return result


@contextmanager
def _settings(values):
    from asimov import config
    old = {section: dict(config.items(section)) if config.has_section(section) else None
           for section in values}
    for section, items in values.items():
        if not config.has_section(section):
            config.add_section(section)
        for key, value in items.items():
            config.set(section, key, str(value))
    try:
        yield
    finally:
        for section, items in old.items():
            config.remove_section(section)
            if items is not None:
                config.add_section(section)
                for key, value in items.items():
                    config.set(section, key, value)


@pytest.fixture
def native(tmp_path):
    pytest.importorskip("asimov")
    if importlib.metadata.version("asimov") != "0.8.0":
        pytest.skip("native provenance regression is pinned to Asimov 0.8.0")
    import lal
    import lal.series
    from igwn_ligolw import utils
    from RIFT import lalsimutils
    from RIFT.asimov.rift import Rift
    from asimov import analysis

    # Reuse versioned scientific metadata, with cheap generation-only inputs.
    from test_asimov_eccentric_fref_handoff import _base_meta
    meta = _base_meta()
    meta["waveform"].update({"approximant": "IMRPhenomXPHM", "maximum mode": 2})
    meta["priors"]["luminosity distance"].pop("type", None)
    repo = tmp_path / "repo"
    category = repo / "C01_offline"
    category.mkdir(parents=True)
    coinc = category / "coinc.xml"
    shutil.copy2(ROOT / ".travis/ref_ini/coinc.xml", coinc)
    seed = lalsimutils.ChooseWaveformParams()
    seed.m1, seed.m2 = 35 * lal.MSUN_SI, 30 * lal.MSUN_SI
    seed.fref = seed.fmin = 20
    lalsimutils.ChooseWaveformParams_array_to_xml(
        [seed, seed.manual_copy()], fname=str(tmp_path / "seed"), fref=20.)
    meta["sampler"]["manual grid"] = str(tmp_path / "seed.xml.gz")
    cache = tmp_path / "empty.cache"
    cache.write_text("")
    meta["scheduler"]["pipeline"] = {
        "fake-data-cache": str(cache), "skip-reproducibility": True,
        "ile-no-gpu": True,
    }
    psds = {}
    for ifo in meta["interferometers"]:
        series = lal.CreateREAL8FrequencySeries(ifo, lal.LIGOTimeGPS(0), 0, 1,
                                               lal.SecondUnit, 513)
        series.data.data[:] = 1e-46
        document = lal.series.make_psd_xmldoc({ifo: series})
        path = category / (ifo + "-psd.xml.gz")
        utils.write_filename(document, str(path), compress="gz")
        psds[ifo] = str(path)
    event = SimpleNamespace(name="GW190426_190642", work_dir=str(tmp_path),
                            repository=SimpleNamespace(directory=str(repo)), productions=[])
    production = SimpleNamespace(name="native-rift", pipeline="rift", category="C01_offline",
        event=event, meta=meta, rundir=str(tmp_path / "run"), dependencies=[],
        xml_psds=psds, status="ready",
        get_coincfile=lambda: str(coinc),
        get_configuration=lambda: SimpleNamespace(ini_loc="native.ini"),
        get_meta=lambda key: meta.get(key), set_meta=lambda key, value: meta.__setitem__(key, value))
    event.productions.append(production)
    pipe = Rift(production)
    production.pipeline = pipe
    pipe.logger = logging.getLogger("rift.native.lifecycle")
    settings = {"general": {"calibration_directory": "C01_offline", "calibration": "C01",
                             "webroot": str(tmp_path / "web")},
                "pipelines": {"environment": sys.prefix},
                "rift": {"environment": sys.prefix}, "condor": {"user": "rift-ci"}}
    previous = os.getcwd()
    old_env = dict(os.environ)
    os.environ.update(SINGULARITY_RIFT_IMAGE="generation-only.sif",
                      SINGULARITY_BASE_EXE_DIR="/usr/bin/", RIFT_LOWLATENCY="True")
    with _settings(settings):
        os.chdir(tmp_path)
        try:
            yield SimpleNamespace(pipe=pipe, production=production, run=Path(production.rundir),
                configure=lambda: analysis.Analysis.make_config(production, str(category / "native.ini")),
                ini=category / "native.ini", coinc=coinc, cache=cache, root=tmp_path)
        finally:
            os.chdir(previous)
            os.environ.clear()
            os.environ.update(old_env)


def test_real_native_configuration_and_builder_preserve_provenance(native):
    n = native
    n.pipe.before_config(dryrun=False)
    assert not n.run.exists(), "configuration must not pre-create the builder's run directory"
    n.configure()
    assert n.ini.is_file() and not n.run.exists()
    n.pipe.build_dag()
    assert (n.run / DAG).stat().st_size > 0
    assert list(n.run.glob("*.sub")), "real builder must emit submit descriptions"
    environment = n.run / "environment.json"
    assert environment.is_file() and environment.stat().st_size
    assert (n.run / "environment-pip.txt").is_file()
    recorded = n.production.meta["environment"]
    assert recorded["captured_at"] is True
    assert all(Path(path).parent == n.run for path in recorded["files"].values())
    before = _snapshot(n.run)
    from RIFT.asimov.rift import PipelineException
    for action in (n.pipe.before_config, n.pipe.build_dag):
        with pytest.raises(PipelineException):
            action()
        assert _snapshot(n.run) == before


@pytest.mark.parametrize("kind", ["scientific", "partial", "provenance", "empty", "stale-marker"])
def test_existing_run_refusal_is_byte_and_mtime_preserving(native, kind):
    n = native
    n.run.mkdir()
    content = {
        "scientific": {DAG: "# old workflow", "all.net": "existing science", "H1-psd.xml.gz": "old PSD"},
        "partial": {"args_ile.txt": "interrupted generation", "partial.tmp": "state"},
        "provenance": {"environment.json": '{"old": true}', "environment-pip.txt": "frozen old environment"},
        "empty": {}, "stale-marker": {".rift-owner": "untrusted stale ownership"},
    }[kind]
    for name, value in content.items():
        (n.run / name).write_text(value)
    before = _snapshot(n.run)
    from RIFT.asimov.rift import PipelineException
    for action in (n.pipe.before_config, n.pipe.build_dag):
        with pytest.raises(PipelineException):
            action()
        assert _snapshot(n.run) == before


@pytest.mark.parametrize("dangling", [False, True])
def test_symlink_run_is_never_adopted(native, dangling):
    n = native
    target = n.root / "target"
    if not dangling:
        target.mkdir()
        (target / "science").write_text("untouched")
    n.run.symlink_to(target, target_is_directory=True)
    before = _snapshot(target) if target.exists() else None
    from RIFT.asimov.rift import PipelineException
    for action in (n.pipe.before_config, n.pipe.build_dag):
        with pytest.raises(PipelineException):
            action()
        assert n.run.is_symlink()
        assert (_snapshot(target) if target.exists() else None) == before


def test_real_builder_race_has_only_one_owner(native):
    n = native
    n.pipe.before_config()
    n.configure()
    # Both independently execute the real builder. No patch of mkdir or hooks.
    command = [sys.executable, str(CODE / "bin/util_RIFT_pseudo_pipe.py"),
               "--use-ini", str(n.ini), "--use-coinc", str(n.coinc),
               "--use-rundir", str(n.run), "--ile-no-gpu",
               "--fake-data-cache", str(n.cache), "--skip-reproducibility"]
    children = [subprocess.Popen(command, cwd=n.root, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT) for _ in range(2)]
    outputs = [child.communicate(timeout=240)[0] for child in children]
    assert sorted(child.returncode for child in children) == [0, 1], outputs
    loser = next(output for child, output in zip(children, outputs) if child.returncode)
    assert b"FileExistsError" in loser
    assert (n.run / DAG).is_file()
    assert list(n.run.glob("*.sub"))


def test_interrupted_real_generation_is_preserved_and_cannot_be_retried(native):
    n = native
    n.pipe.before_config()
    n.configure()
    # An unavailable versioned input fails inside the real builder after mkdir.
    n.production.get_coincfile = lambda: str(n.root / "missing-coinc.xml")
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException):
        n.pipe.build_dag()
    assert n.run.is_dir()
    assert not (n.run / "environment.json").exists()
    assert not (n.run / DAG).exists()
    before = _snapshot(n.run)
    for action in (n.pipe.before_config, n.pipe.build_dag):
        with pytest.raises(PipelineException):
            action()
        assert _snapshot(n.run) == before


def test_parent_path_alias_and_workdir_alias_refuse_before_frame_cache_writes(native):
    n = native
    n.run.mkdir()
    (n.run / "environment.json").write_text("old provenance")
    parent_alias = n.root / "parent-alias"
    parent_alias.symlink_to(n.root, target_is_directory=True)
    n.production.rundir = str(parent_alias / "run")
    n.production.event.work_dir = str(n.run)
    n.production.meta["orchestrator"] = "rimsky"
    frame = n.root / "H-fixture-1000000000-4.gwf"
    frame.write_bytes(b"generation-only frame cache input")
    n.production.meta["data"]["data files"] = {"H1": [str(frame)]}
    before = _snapshot(n.run)
    from RIFT.asimov.rift import PipelineException
    for action in (n.pipe.before_config, n.pipe.build_dag):
        with pytest.raises(PipelineException):
            action()
        assert _snapshot(n.run) == before
        assert not (n.run / "H1-rimsky.cache").exists()


def test_directory_created_between_config_and_build_is_untouched(native):
    n = native
    n.pipe.before_config()
    n.configure()
    n.run.mkdir()
    (n.run / DAG).write_text("# plausible old DAG must not imply builder success")
    (n.run / "environment.json").write_text("old provenance")
    before = _snapshot(n.run)
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException):
        n.pipe.build_dag()
    assert _snapshot(n.run) == before


@pytest.mark.parametrize("key", ["help", "h", "help=ignored", "use-rundir", "use-rund", "use-r", "use-rundir=/tmp/forbidden",
                                    "use-ini", "use-in", "use-ini=/tmp/forbidden"])
@pytest.mark.parametrize("value", [True, False, "false", "True", "/tmp/forbidden", None])
def test_scheduler_cannot_override_owned_input_paths(native, key, value):
    n = native
    n.pipe.before_config()
    n.configure()
    n.production.meta["scheduler"]["pipeline"][key] = value
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException):
        n.pipe.build_dag()
    assert not n.run.exists()


@pytest.mark.parametrize("key", ["use-rundir", "use_rundir", "use-ini", "use_ini"])
def test_ini_cannot_override_owned_input_paths(native, key):
    n = native
    n.pipe.before_config()
    n.configure()
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(n.ini)
    parser.set("rift-pseudo-pipe", key, repr(str(n.root / "forbidden")))
    # This is the versioned input under test, before generation, never a DAG edit.
    with n.ini.open("w") as stream:
        parser.write(stream)
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException):
        n.pipe.build_dag()
    assert not n.run.exists()
    assert not (n.root / "forbidden").exists()


def test_late_competitor_with_stale_dag_cannot_trigger_postbuild_writes(native):
    n = native
    n.pipe.before_config()
    n.configure()
    saved = {}

    def lookup_coinc_after_competitor():
        # Simulate an independent actor winning after adapter preflight, at the
        # persistence lookup boundary. The real builder must still refuse.
        n.run.mkdir()
        (n.run / DAG).write_text("# old DAG is not evidence of this invocation succeeding")
        (n.run / "environment.json").write_text("old provenance")
        (n.run / "environment-pip.txt").write_text("old package lock")
        (n.run / "H1-psd.xml.gz").write_bytes(b"old PSD")
        saved.update(_snapshot(n.run))
        return str(n.coinc)

    n.production.get_coincfile = lookup_coinc_after_competitor
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException) as error:
        n.pipe.build_dag()
    assert "FileExistsError" in str(error.value)
    assert _snapshot(n.run) == saved
    assert "environment" not in n.production.meta


def test_concurrent_native_adapters_have_one_success_and_one_provenance_owner(native):
    import multiprocessing
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("CIT Linux fork isolates each adapter's cwd and environment")
    n = native
    n.pipe.before_config()
    n.configure()
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    results = context.Queue()

    def simultaneous_lookup():
        # Persistence interface synchronization ensures both actual adapter
        # prechecks complete before either starts its real builder subprocess.
        barrier.wait(timeout=30)
        return str(n.coinc)

    n.production.get_coincfile = simultaneous_lookup

    def invoke_adapter():
        try:
            n.pipe.build_dag()
            results.put(("success", "environment" in n.production.meta, ""))
        except Exception as error:
            results.put((type(error).__name__, "environment" in n.production.meta, str(error)))

    children = [context.Process(target=invoke_adapter) for _ in range(2)]
    try:
        for child in children:
            child.start()
        receipts = [results.get(timeout=300) for _ in children]
        for child in children:
            child.join(timeout=30)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
        results.close()
    assert sorted(status for status, _, _ in receipts) == ["PipelineException", "success"], receipts
    assert next(captured for status, captured, _ in receipts if status == "success")
    assert not next(captured for status, captured, _ in receipts if status != "success")
    assert "FileExistsError" in next(message for status, _, message in receipts if status != "success")
    assert (n.run / DAG).is_file()
    assert list(n.run.glob("*.sub"))
    assert (n.run / "environment.json").is_file()
    assert (n.run / "environment-pip.txt").is_file()


@pytest.mark.parametrize("name", ["environment.json", "environment-pip.txt", "environment-conda.txt"])
@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_capture_helper_never_replaces_occupied_provenance(native, name, kind):
    n = native
    n.run.mkdir()
    target = n.run / name
    protected = n.root / "protected-provenance"
    protected.write_bytes(b"pre-existing provenance target")
    if kind == "file":
        target.write_bytes(b"old environment record")
    elif kind == "directory":
        target.mkdir()
        (target / "keep").write_bytes(b"old directory content")
    else:
        target.symlink_to(protected)
    before = _snapshot(n.run)
    protected_before = (protected.read_bytes(), protected.stat().st_mtime_ns)
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException):
        n.pipe._capture_build_environment()
    assert _snapshot(n.run) == before
    assert (protected.read_bytes(), protected.stat().st_mtime_ns) == protected_before
    assert "environment" not in n.production.meta


def test_capture_helper_rejects_swallowed_write_failure_with_stale_metadata(native):
    n = native
    blocker = n.root / "not-a-directory"
    blocker.write_bytes(b"preserve me")
    n.production.rundir = str(blocker / "run")
    n.production.meta["environment"] = {
        "captured_at": True,
        "files": {"metadata": str(blocker / "run/environment.json")},
        "unrelated": "preserve metadata outside capture receipt",
    }
    before = (blocker.read_bytes(), blocker.stat().st_mtime_ns)
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException, match="environment capture failed"):
        n.pipe._capture_build_environment()
    assert n.production.status == "stuck"
    assert (blocker.read_bytes(), blocker.stat().st_mtime_ns) == before
    environment = n.production.meta["environment"]
    assert "captured_at" not in environment
    assert "files" not in environment
    assert environment["unrelated"] == "preserve metadata outside capture receipt"


@pytest.mark.parametrize("kind", ["relative", "fallback"])
def test_normalized_existing_paths_preserve_runs(native, kind):
    n = native
    if kind == "relative":
        target = n.run
        n.production.rundir = "run"
    else:
        os.environ["HOME"] = str(n.root)
        target = n.root / n.production.event.name / n.production.name
        n.production.rundir = None
    target.mkdir(parents=True)
    (target / "environment.json").write_text("existing provenance")
    (target / DAG).write_text("existing workflow")
    before = _snapshot(target)
    from RIFT.asimov.rift import PipelineException
    for action in (n.pipe.before_config, n.pipe.build_dag):
        with pytest.raises(PipelineException):
            action()
        assert Path(n.production.rundir) == target
        assert _snapshot(target) == before
