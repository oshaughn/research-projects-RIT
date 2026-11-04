"""Scheduler-free real configuration -> builder lifecycle, including provenance.

The Production below supplies only persistence/lookup interfaces. No pipeline
hook, provenance writer, builder subprocess, directory guard or generated file is
mocked in the native success and concurrency tests. Separate negative contract
tests inject explicit builder/capture/converter failures. Run with the branch installed (including its bin scripts) in the selected
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
        # Asimov keeps a refused production ready unless it is marked stuck.
        assert n.production.status == "stuck"
        n.production.status = "ready"


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
                                    "use-ini", "use-in", "use-ini=/tmp/forbidden", "use_rundir", "use_r", "use_ini"])
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


def _stub_builder(native, body):
    """Replace only the builder executable; the adapter under test is real."""
    from asimov import config
    prefix = native.root / "stub-env"
    (prefix / "bin").mkdir(parents=True)
    builder = prefix / "bin/util_RIFT_pseudo_pipe.py"
    builder.write_text(f"#!{sys.executable}\nimport sys, os\n"
                       "rundir = sys.argv[sys.argv.index('--use-rundir') + 1]\n"
                       f"open({str(native.root / 'builder-called')!r}, 'w').close()\n" + body)
    builder.chmod(0o755)
    config.set("rift", "environment", str(prefix))


@pytest.mark.parametrize("kind", ["missing", "empty", "symlink", "directory"])
def test_builder_exit_zero_without_regular_dag_fails_without_capture(native, kind):
    n = native
    n.pipe.before_config()
    n.configure()
    _stub_builder(n, "os.mkdir(rundir)\n" + {
        "missing": "",
        "empty": f"open(os.path.join(rundir, {DAG!r}), 'w').close()\n",
        "symlink": f"open(os.path.join(rundir, 'real.dag'), 'w').write('JOB A a.sub')\n"
                   f"os.symlink('real.dag', os.path.join(rundir, {DAG!r}))\n",
        "directory": f"os.mkdir(os.path.join(rundir, {DAG!r}))\n",
    }[kind])
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException, match="DAG file could not be created"):
        n.pipe.build_dag()
    assert (n.root / "builder-called").exists()
    assert n.production.status == "stuck"
    assert not (n.run / "environment.json").exists()
    assert "files" not in n.production.meta.get("environment", {})


def test_build_refuses_existing_rundir_before_builder_or_build_hooks(native):
    n = native
    n.pipe.before_config()
    n.configure()
    _stub_builder(n, "os.mkdir(rundir)\n")
    n.run.mkdir()
    called = []
    n.pipe.before_build = lambda *args, **kwargs: called.append(True)
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException, match="already exists"):
        n.pipe.build_dag()
    assert not called
    assert not (n.root / "builder-called").exists()


def test_dryrun_refusal_does_not_change_production(native):
    n = native
    n.run.mkdir()
    n.production.rundir = "run"
    from RIFT.asimov.rift import PipelineException
    for action in (n.pipe.before_config, n.pipe.build_dag):
        with pytest.raises(PipelineException):
            action(dryrun=True)
        assert n.production.rundir == "run" and n.production.status == "ready"


def test_dryrun_configuration_writes_no_frame_cache(native):
    n = native
    frame = n.root / "H-H1_RIMSKY-1000000000-16.gwf"
    frame.write_bytes(b"frame")
    n.production.meta["orchestrator"] = "rimsky"
    n.production.meta.setdefault("data", {})["data files"] = {"H1": [str(frame)]}
    n.pipe.before_config(dryrun=True)
    assert not list(n.root.glob("*-rimsky.cache"))
    n.pipe.before_config()
    assert (n.root / "H1-rimsky.cache").is_file()


@pytest.mark.parametrize("violation", ["no-manifest", "no-metadata", "metadata-elsewhere", "file-elsewhere",
                                       "empty-manifest", "symlink-manifest"])
def test_capture_receipt_must_describe_nonempty_files_in_rundir(native, monkeypatch, violation):
    import asimov.environment
    n = native
    n.run.mkdir()
    elsewhere = n.root / "elsewhere"
    elsewhere.mkdir()

    def write_receipt(directory):
        directory = Path(directory)
        files = {"metadata": directory / "environment.json", "pip": directory / "environment-pip.txt"}
        files["metadata"].write_text("{}")
        if violation == "no-manifest":
            del files["pip"]
        elif violation == "no-metadata":
            del files["metadata"]
        elif violation == "metadata-elsewhere":
            files["metadata"] = elsewhere / "environment.json"
            files["metadata"].write_text("{}")
        elif violation == "file-elsewhere":
            files["conda"] = elsewhere / "environment-conda.txt"
            files["conda"].write_text("conda")
        elif violation == "empty-manifest":
            files["pip"].write_text("")
        elif violation == "symlink-manifest":
            (elsewhere / "pip.txt").write_text("pip")
            files["pip"].symlink_to(elsewhere / "pip.txt")
        if violation not in ("no-manifest", "empty-manifest", "symlink-manifest"):
            files["pip"].write_text("pip")
        return {key: str(path) for key, path in files.items()}

    # Only Asimov's file writer is replaced; Asimov still records its receipt.
    monkeypatch.setattr(asimov.environment, "capture_and_save_environment", write_receipt)
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException, match="environment capture failed"):
        n.pipe._capture_build_environment()
    assert n.production.status == "stuck"


def test_cached_pip_freeze_failure_is_retried(native):
    import asimov.environment
    n = native
    n.run.mkdir()
    environment = asimov.environment.EnvironmentCapture()
    key = (environment.env_type, sys.executable)
    asimov.environment._PACKAGE_CACHE[key] = {"pip_packages": None, "conda_environment": None}
    try:
        n.pipe._capture_build_environment()
    finally:
        asimov.environment._PACKAGE_CACHE.pop(key, None)
    assert (n.run / "environment-pip.txt").stat().st_size > 0


def test_builder_ini_parsing_matches_builder(native):
    n = native
    n.pipe.before_config()
    n.configure()
    text = n.ini.read_text()
    # The builder preserves key case, so these are two distinct valid options.
    n.ini.write_text(text + "\n[rift-case-check]\nRequirements = a\nrequirements = b\n")
    n.pipe._validate_builder_paths(str(n.ini))
    n.ini.write_text(text + "\n[rift-case-check]\nkey = a\nkey = b\n")
    from RIFT.asimov.rift import PipelineException
    with pytest.raises(PipelineException, match="Unable to read INI"):
        n.pipe._validate_builder_paths(str(n.ini))


def test_real_native_q2_configuration_reaches_workers_at_4096_hz(native):
    import shlex
    n = native
    n.production.meta["likelihood"]["sample rate"] = 4096
    n.production.meta["sampler"]["ile"]["q time pregrid factor"] = 2
    # Materialize the versioned D4 source profile before configuration, including
    # removal of a legacy fixed export setting. Generated files stay untouched.
    import yaml
    from asimov.utils import update
    profile = yaml.safe_load((ROOT / "validation/asimov-rundir/o4d-native-q2-profile.yaml").read_text())
    n.production.meta["scheduler"]["pipeline"]["internal-ile-srate-time-resampling"] = 32768
    update(n.production.meta, profile["pipelines"]["rift"])
    n.pipe.before_config()
    n.configure()
    assert not n.run.exists()
    n.pipe.build_dag()
    args = shlex.split((n.run / "args_ile.txt").read_text())
    assert float(args[args.index("--srate") + 1]) == 4096
    assert args[args.index("--q-time-pregrid-factor") + 1] == "2"
    assert "--srate-internal" not in args
    assert "--srate-resample-time-marginalization" not in args
    assert "--time-marginalization" in args and "--vectorized" in args and "--gpu" in args
    submits = list(n.run.glob("ILE*.sub"))
    assert submits and (n.run / DAG).is_file()
    for path in submits:
        assert "--q-time-pregrid-factor 2" in path.read_text(), path
    assert n.production.meta["environment"]["captured_at"] is True


def _native_bootstrap_source(n):
    import h5py
    import numpy as np
    columns = ("mass_1", "mass_2", "chirp_mass", "luminosity_distance",
               "geocent_time", "iota", "psi", "ra", "dec", "phase",
               "spin_1x", "spin_1y", "spin_1z", "spin_2x", "spin_2y", "spin_2z")
    samples = np.zeros(2, dtype=[(name, float) for name in columns])
    samples["mass_1"] = 25
    samples["mass_2"] = 20
    samples["chirp_mass"] = (25*20)**.6 / 45**.2
    samples["luminosity_distance"] = 400
    samples["geocent_time"] = n.production.meta["event time"]
    samples["iota"] = .5
    source = n.root / "bootstrap-input.h5"
    with h5py.File(source, "w") as handle:
        handle.create_group("native").create_dataset("posterior_samples", data=samples)
    n.production.meta["scheduler"].update({"bootstrap file": str(source), "bootstrap size": 2})
    n.production.meta["sampler"].pop("manual grid", None)
    n.production.meta["priors"]["chirp mass"].update(minimum=10, maximum=50)
    return source


@pytest.mark.parametrize("requested", [None, False, True])
def test_native_bootstrap_binds_final_coincidence_and_preserves_initial_input(native, requested):
    from RIFT import lalsimutils
    import lal
    from igwn_ligolw import lsctables, utils
    n = native
    _native_bootstrap_source(n)
    if requested is not None:
        n.production.meta["scheduler"]["bootstrap coinc"] = requested
    initial = n.root / "repo/analyses/coinc.xml"
    initial.parent.mkdir()
    shutil.copy2(n.coinc, initial)
    old = initial.read_bytes(), initial.stat().st_mtime_ns
    initial_row = lsctables.SnglInspiralTable.get_table(utils.load_filename(
        str(initial), contenthandler=lalsimutils.cthdler))[0]
    original_mass = initial_row.mass1
    category_before = n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns
    # The repository is project-relative; the real builder changes cwd.
    n.production.event.repository.directory = "repo"
    n.production.get_coincfile = lambda: "repo/analyses/coinc.xml"
    work = n.root / "work"
    work.mkdir()
    n.production.event.work_dir = str(work)
    n.pipe.before_config()
    n.configure()
    n.pipe.build_dag()
    assert (initial.read_bytes(), initial.stat().st_mtime_ns) == old
    target = lalsimutils.xml_to_ChooseWaveformParams_array(str(n.run / "target_params.xml.gz"))[0]
    if requested is True:
        assert target.m1 / lal.MSUN_SI == pytest.approx(25, abs=1e-4)
        assert target.m2 / lal.MSUN_SI == pytest.approx(20, abs=1e-4)
        import shlex
        args = shlex.split((n.run / "args_ile.txt").read_text())
        assert float(args[args.index("--event-time") + 1]) == pytest.approx(
            n.production.meta["event time"], abs=2e-6)
    else:
        assert target.m1 / lal.MSUN_SI == pytest.approx(original_mass, abs=1e-4)
        assert (n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns) == category_before
    assert (n.run / DAG).is_file()


@pytest.mark.parametrize("value", ["False", "true", 0, 1, None])
def test_bootstrap_coincidence_requires_explicit_boolean(native, value):
    from RIFT.asimov.rift import PipelineException
    n = native
    n.production.meta["scheduler"]["bootstrap coinc"] = value
    before = _snapshot(n.root)
    with pytest.raises(PipelineException, match="YAML boolean"):
        n.pipe.build_dag()
    assert not n.run.exists() and _snapshot(n.root) == before


@pytest.mark.parametrize("source", ["absent", "unresolved"])
def test_requested_bootstrap_coincidence_cannot_use_stale_input_without_source(native, source):
    from RIFT.asimov.rift import PipelineException
    n = native
    n.production.meta["scheduler"]["bootstrap coinc"] = True
    if source == "unresolved":
        n.production.meta["scheduler"]["bootstrap upstream"] = True
        n.production.meta["sampler"].pop("manual grid", None)
        n.configure()
    before = n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns
    with pytest.raises(PipelineException, match="bootstrap source"):
        n.pipe.build_dag()
    assert not n.run.exists()
    assert (n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns) == before


@pytest.mark.parametrize("failure", ["nonzero", "missing", "invalid-xml"])
def test_native_bootstrap_conversion_failure_never_selects_stale_coincidence(native, monkeypatch, failure):
    from RIFT.asimov import rift
    from RIFT.asimov.rift import PipelineException
    n = native
    _native_bootstrap_source(n)
    n.production.meta["scheduler"]["bootstrap coinc"] = True
    n.pipe.before_config()
    n.configure()
    before = n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns
    real_run = rift.subprocess.run
    def fail_only_coincidence(command, *args, **kwargs):
        if Path(command[0]).name != "util_SimInspiralToCoinc.py":
            return real_run(command, *args, **kwargs)
        if failure != "missing":
            Path(command[command.index("--output") + 1]).write_text("<invalid/>")
        return subprocess.CompletedProcess(command, 1 if failure == "nonzero" else 0, stdout="injected conversion result")
    monkeypatch.setattr(rift.subprocess, "run", fail_only_coincidence)
    with pytest.raises(PipelineException, match="coincidence conversion failed"):
        n.pipe.build_dag()
    assert not n.run.exists()
    assert (n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns) == before
    assert not list(n.coinc.parent.glob(".rift-bootstrap-coinc-*"))


@pytest.mark.parametrize("coincidence", [False, True])
def test_native_bootstrap_dryrun_creates_no_grid_coincidence_or_run(native, capsys, coincidence):
    n = native
    _native_bootstrap_source(n)
    n.production.meta["scheduler"]["bootstrap coinc"] = coincidence
    n.configure()
    before = _snapshot(n.root)
    n.pipe.build_dag(dryrun=True)
    preview = capsys.readouterr().out
    assert "<new-private-bootstrap-grid>" in preview
    if coincidence:
        assert "<new-private-bootstrap-input>" in preview
    assert not n.run.exists() and _snapshot(n.root) == before


@pytest.mark.parametrize("key", ["use-coinc", "use-coi", "use-coinc=/tmp/other.xml"])
def test_bootstrap_coincidence_cannot_be_redirected_after_native_selection(native, key):
    from RIFT.asimov.rift import PipelineException
    n = native
    _native_bootstrap_source(n)
    n.production.meta["scheduler"].update({"bootstrap coinc": True})
    n.production.meta["scheduler"]["pipeline"][key] = True
    n.configure()
    with pytest.raises(PipelineException, match="owned by Asimov"):
        n.pipe.build_dag()
    assert not n.run.exists()


def test_bootstrap_coincidence_inputs_remain_private_across_invocations(native):
    from RIFT import lalsimutils
    import lal
    n = native
    inputs = []
    for mass in (25, 35):
        point = lalsimutils.ChooseWaveformParams()
        point.m1, point.m2 = mass * lal.MSUN_SI, 20 * lal.MSUN_SI
        point.fref = point.fmin = 20
        point.tref = n.production.meta["event time"]
        stem = n.root / ("grid-" + str(mass))
        lalsimutils.ChooseWaveformParams_array_to_xml([point], fname=str(stem), fref=20.)
        path = Path(n.pipe._convert_bootstrap_coinc(str(stem)+".xml.gz", sys.prefix))
        inputs.append((path, path.read_bytes()))
    assert inputs[0][0] != inputs[1][0]
    for path, content in inputs:
        assert path.read_bytes() == content
    from igwn_ligolw import lsctables, utils
    for (path, _), mass in zip(inputs, (25, 35)):
        rows = lsctables.SnglInspiralTable.get_table(utils.load_filename(
            str(path), contenthandler=lalsimutils.cthdler))
        assert rows[0].mass1 == pytest.approx(mass, abs=1e-4)


@pytest.mark.parametrize("key", ["manual-initial-grid", "manual-init", "manual_initial_grid"])
def test_native_bootstrap_rejects_competing_grid_selection(native, key):
    from RIFT.asimov.rift import PipelineException
    n = native
    _native_bootstrap_source(n)
    n.production.meta["scheduler"]["pipeline"][key] = "/tmp/foreign.xml.gz"
    n.configure()
    with pytest.raises(PipelineException, match="owned by Asimov"):
        n.pipe.build_dag()
    assert not n.run.exists()


def test_native_bootstrap_rejects_competing_ini_grid_selection(native):
    from RIFT.asimov.rift import PipelineException
    n = native
    _native_bootstrap_source(n)
    n.production.meta["sampler"]["manual grid"] = str(n.root / "seed.xml.gz")
    n.configure()
    with pytest.raises(PipelineException, match="owned by Asimov"):
        n.pipe.build_dag()
    assert not n.run.exists()


def test_legacy_manual_bootstrap_uses_existing_readonly_grid(native):
    n = native
    target = n.coinc.parent / (n.production.name + "_bootstrap.xml.gz")
    shutil.copy2(n.root / "seed.xml.gz", target)
    before = target.read_bytes(), target.stat().st_mtime_ns
    n.production.meta["sampler"].pop("manual grid", None)
    n.pipe.bootstrap = "manual"
    n.configure()
    n.pipe.build_dag()
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert (n.run / DAG).is_file()


def test_legacy_manual_bootstrap_missing_grid_fails_explicitly(native):
    from RIFT.asimov.rift import PipelineException
    n = native
    n.pipe.bootstrap = "manual"
    n.production.meta["sampler"].pop("manual grid", None)
    n.configure()
    with pytest.raises(PipelineException, match="Manual bootstrap grid is missing"):
        n.pipe.build_dag()
    assert not n.run.exists()


@pytest.mark.parametrize("selection", [
    {}, {"bootstrap upstream": False},
    {"bootstrap upstream": False, "bootstrap file": None},
    {"bootstrap upstream": False, "bootstrap file": ""},
], ids=["missing", "disabled", "disabled-null-file", "disabled-empty-file"])
def test_disabled_upstream_bootstrap_preserves_native_manual_grid(native, selection):
    n = native
    n.production.meta["scheduler"].pop("bootstrap upstream", None)
    n.production.meta["scheduler"].update(selection)
    before = n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns
    n.pipe.before_config()
    n.configure()
    n.pipe.build_dag()
    assert (n.run / DAG).is_file()
    assert n.pipe.bootstrap is False
    assert not list(n.coinc.parent.glob(".rift-bootstrap-*"))
    assert (n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns) == before


def test_disabled_upstream_does_not_disable_explicit_native_bootstrap_file(native):
    from RIFT import lalsimutils
    import lal
    n = native
    _native_bootstrap_source(n)
    n.production.meta["scheduler"].update({"bootstrap upstream": False, "bootstrap coinc": True})
    before = n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns
    n.pipe.before_config()
    n.configure()
    n.pipe.build_dag()
    target = lalsimutils.xml_to_ChooseWaveformParams_array(str(n.run / "target_params.xml.gz"))[0]
    assert target.m1 / lal.MSUN_SI == pytest.approx(25, abs=1e-4)
    assert target.m2 / lal.MSUN_SI == pytest.approx(20, abs=1e-4)
    assert (n.run / DAG).is_file()
    assert (n.coinc.read_bytes(), n.coinc.stat().st_mtime_ns) == before
    assert list(n.coinc.parent.glob(".rift-bootstrap-grid-*"))
    assert list(n.coinc.parent.glob(".rift-bootstrap-coinc-*"))


@pytest.mark.parametrize("file_value", ["missing", None, ""])
def test_disabled_upstream_cannot_satisfy_requested_bootstrap_coincidence(native, file_value):
    from RIFT.asimov.rift import PipelineException
    n = native
    n.production.meta["scheduler"].update({"bootstrap upstream": False, "bootstrap coinc": True})
    if file_value != "missing":
        n.production.meta["scheduler"]["bootstrap file"] = file_value
    before = _snapshot(n.root)
    with pytest.raises(PipelineException, match="configured bootstrap source"):
        n.pipe.build_dag()
    assert not n.run.exists() and _snapshot(n.root) == before
