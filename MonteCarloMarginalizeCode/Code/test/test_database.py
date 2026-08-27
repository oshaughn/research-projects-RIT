"""Unit tests for RIFT.simulation_manager.database (v2 archive).

No condor required. The cluster-aware bits (DualCondorRunQueue) are
exercised up to the actual condor_submit_dag call (which is missing
in CI and the file-generation path is what matters here).

Run:
    pytest MonteCarloMarginalizeCode/Code/test/test_database.py -v
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

from RIFT.simulation_manager.database import (
    Archive, Manifest, Index, StatusRecord,
    LocalRequestQueue, LocalRunQueue,
    DualCondorRunQueue, DualCondorRequestQueue,
    SlurmRunQueue, SlurmRequestQueue,
    make_queues_from_manifest,
    freeze_code, load_entrypoint,
    DEFAULT_GETENV_ALLOWLIST,
)


# ---------------------------------------------------------------------------
# Frozen-code helpers used by several tests
# ---------------------------------------------------------------------------

def _gen(params, sim_dir, level, prev_levels):
    import json, math, os
    val = float(params) * math.sqrt(2.0) * int(level)
    out = os.path.join(sim_dir, "level_{}.json".format(level))
    with open(out, "w") as f:
        json.dump({"level": int(level), "value": val,
                   "prev_count": len(prev_levels)}, f)


def _summ(sim_dir, params, levels=None):
    import json, os
    if levels:
        latest = levels[-1]
    else:
        cands = sorted(p for p in os.listdir(sim_dir)
                       if p.startswith("level_") and p.endswith(".json"))
        latest = os.path.join(sim_dir, cands[-1])
    with open(latest) as f:
        data = json.load(f)
    return {"value": data["value"], "level": data["level"]}


def _same_q(a, b):
    return abs(float(a) - float(b)) < 1e-9


def _lookup_key(p):
    return round(float(p), 9)


def make_archive(tmp_path, name="t", **manifest_kw):
    base = tmp_path / name
    manifest = Manifest.new(name=name,
                            request_queue_kind="local",
                            run_queue_kind="local",
                            summarizer_entrypoint="summarizer:_summ",
                            same_q_entrypoint="same_q:_same_q",
                            lookup_key_entrypoint="lookup_key:_lookup_key",
                            **manifest_kw)
    archive = Archive(base_location=base, manifest=manifest,
                      generator_spec=_gen,
                      summarizer_spec=_summ,
                      same_q_spec=_same_q,
                      lookup_key_spec=_lookup_key)
    archive.run_queue = LocalRunQueue()
    archive.request_queue = LocalRequestQueue(run_queue=archive.run_queue)
    return archive


# ---------------------------------------------------------------------------
# Manifest / Index / StatusRecord round trips
# ---------------------------------------------------------------------------

def test_manifest_roundtrip(tmp_path):
    m = Manifest.new(name="x", request_queue_kind="local", run_queue_kind="local")
    m.write(tmp_path)
    m2 = Manifest.read(tmp_path)
    assert m2.data["name"] == "x"
    assert m2.data["schema_version"] == 1


def test_index_upsert_and_query(tmp_path):
    idx = Index(tmp_path)
    idx.upsert({"name": "a", "status": "ready", "params": 1})
    idx.upsert({"name": "b", "status": "complete", "params": 2})
    idx.upsert({"name": "a", "status": "complete", "params": 1})  # update
    assert idx.by_name("a")["status"] == "complete"
    assert {r["name"] for r in idx.with_status("complete")} == {"a", "b"}


def test_status_record_transitions(tmp_path):
    rec = StatusRecord.new("1", params=0.5, target_level=2)
    rec.transition("running")
    assert rec.data["status"] == "running"
    assert rec.data["started_at"] is not None
    rec.append_level(1, "sims/1/level_1.json")
    rec.append_level(2, "sims/1/level_2.json")
    rec.transition("complete")
    assert rec.data["completed_at"] is not None
    assert rec.data["current_level"] == 2
    assert len(rec.data["levels"]) == 2


def test_status_record_rejects_unknown_status():
    rec = StatusRecord.new("1", params=0.5)
    with pytest.raises(ValueError):
        rec.transition("bogus")


# ---------------------------------------------------------------------------
# freeze_code / load_entrypoint
# ---------------------------------------------------------------------------

def test_freeze_code_callable(tmp_path):
    code = tmp_path / "code"
    ep = freeze_code(_gen, code)
    assert ep.startswith("generator:")
    assert (code / "generator.py").exists()
    fn = load_entrypoint(code, ep)
    assert callable(fn)


def test_freeze_code_path(tmp_path):
    src = tmp_path / "mygen.py"
    src.write_text("def run(params, sim_dir, level, prev_levels):\n    pass\n")
    code = tmp_path / "code"
    ep = freeze_code(str(src), code)
    assert (code / "generator.py").exists()
    assert ":run" in ep


# ---------------------------------------------------------------------------
# Archive registration / dedup / refinement
# ---------------------------------------------------------------------------

def test_register_idempotent_same_q(tmp_path):
    archive = make_archive(tmp_path)
    n1 = archive.register(0.5)
    n2 = archive.register(0.5 + 1e-12)   # within same_q tolerance
    assert n1 == n2
    n3 = archive.register(0.5 + 1e-3)    # outside
    assert n3 != n1


def test_register_bumps_target_level(tmp_path):
    archive = make_archive(tmp_path)
    n = archive.register(0.5, target_level=1)
    archive.request_queue.submit_pending(archive)
    assert archive.get_status(n) == "complete"
    n2 = archive.register(0.5, target_level=3)   # bump
    assert n2 == n
    assert archive.get_status(n) == "refine_ready"


def test_refine_and_complete(tmp_path):
    archive = make_archive(tmp_path)
    n = archive.register(0.5, target_level=1)
    archive.request_queue.submit_pending(archive)
    archive.refine(n, target_level=4)
    archive.request_queue.submit_pending(archive)
    rec = StatusRecord.read(archive.sim_dir(n))
    assert rec.data["current_level"] == 4
    assert rec.data["status"] == "complete"
    # Each level file exists, levels[] tracks history.
    assert len(rec.data["levels"]) == 4


# ---------------------------------------------------------------------------
# refresh_status_from_disk
# ---------------------------------------------------------------------------

def test_refresh_status_from_disk_promotes(tmp_path):
    archive = make_archive(tmp_path)
    n = archive.register(0.7, target_level=2)
    # Manually drop level files (as a worker would have).
    sd = archive.sim_dir(n)
    (sd / "level_1.json").write_text('{"level":1}')
    (sd / "level_2.json").write_text('{"level":2}')
    changed = archive.refresh_status_from_disk()
    assert changed[n] == "complete"


def test_refresh_status_promotes_to_refine_ready(tmp_path):
    archive = make_archive(tmp_path)
    n = archive.register(0.7, target_level=3)
    sd = archive.sim_dir(n)
    (sd / "level_1.json").write_text('{"level":1}')
    (sd / "level_2.json").write_text('{"level":2}')
    changed = archive.refresh_status_from_disk()
    assert changed[n] == "refine_ready"


# ---------------------------------------------------------------------------
# Per-sim resource overrides
# ---------------------------------------------------------------------------

def test_set_resources_persists(tmp_path):
    archive = make_archive(tmp_path)
    n = archive.register(0.5)
    archive.set_resources(n, request_memory=8192, request_disk="16G",
                          extra_condor_cmds={"+SpecialFlag": "true"})
    res = archive.get_resources(n)
    assert res["request_memory"] == 8192
    assert res["request_disk"] == "16G"
    assert res["extra_condor_cmds"]["+SpecialFlag"] == "true"


def test_per_sim_overrides_appear_in_submit_file(tmp_path):
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="condor",
                            run_queue_kind="condor")
    archive = Archive(base_location=base, manifest=manifest,
                      generator_spec=_gen)
    rq = DualCondorRunQueue(request_memory=2048, request_disk="2G",
                            auto_release_on_oom=False)
    n = archive.register(0.5, target_level=1)
    archive.set_resources(n, request_memory=12345, request_disk="9G",
                          extra_condor_cmds={"+CustomKey": "42"})
    sub_path = rq.build_worker(archive, n, level=1)
    text = Path(sub_path).read_text()
    assert "request_memory          = 12345M" in text, text
    assert "request_disk            = 9G" in text, text
    assert "+CustomKey              = 42" in text, text


# ---------------------------------------------------------------------------
# DualCondorRunQueue submit-file generation (no real condor)
# ---------------------------------------------------------------------------

def test_dual_condor_run_queue_submit_file_basics(tmp_path):
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="condor",
                            run_queue_kind="condor")
    archive = Archive(base_location=base, manifest=manifest,
                      generator_spec=_gen)
    rq = DualCondorRunQueue()   # all defaults
    n = archive.register(0.7, target_level=3)
    sub = Path(rq.build_worker(archive, n, level=2))
    text = sub.read_text()
    # transfer_input_files declares level_1.json regardless of disk presence.
    assert "level_1.json" in text
    # output remap to canonical archive path.
    assert "level_2.json={}".format(archive.sim_dir(n) / "level_2.json") in text
    # OOM auto-release is on by default.
    assert "periodic_release" in text
    assert "InitialRequestMemory" in text
    # Default getenv is the allowlist, not True.
    assert "getenv                  = " + DEFAULT_GETENV_ALLOWLIST in text
    assert "getenv                  = True" not in text


def test_dag_chains_levels_per_sim(tmp_path):
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="condor",
                            run_queue_kind="condor")
    archive = Archive(base_location=base, manifest=manifest,
                      generator_spec=_gen)
    rq = DualCondorRunQueue()
    n1 = archive.register(0.1, target_level=2)
    n2 = archive.register(0.2, target_level=3)
    rq.submit(archive, [n1, n2])
    # *.dag, not iterdir(): where condor_submit_dag IS installed it runs and
    # drops four companion files beside the DAG (.condor.sub, .dagman.log,
    # .lib.err, .lib.out), so a directory count is 5 there and 1 on a machine
    # without HTCondor. This test was green only on the latter.
    dags = sorted((base / "run_queue" / "dags").glob("*.dag"))
    assert len(dags) == 1, [p.name for p in
                            (base / "run_queue" / "dags").iterdir()]
    text = dags[0].read_text()
    assert "PARENT {}_lvl1 CHILD {}_lvl2".format(n1, n1) in text
    assert "PARENT {}_lvl2 CHILD {}_lvl3".format(n2, n2) in text


def test_make_queues_from_manifest(tmp_path):
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="condor",
                            run_queue_kind="condor",
                            run_queue_extra={"request_memory": 1234})
    archive = Archive(base_location=base, manifest=manifest, generator_spec=_gen)
    req_q, run_q = make_queues_from_manifest(archive)
    assert isinstance(req_q, DualCondorRequestQueue)
    assert isinstance(run_q, DualCondorRunQueue)
    assert run_q.request_memory == 1234


# ---------------------------------------------------------------------------
# DualCondorRunQueue subdag_factory + submit_mode
# ---------------------------------------------------------------------------

def test_dual_condor_subdag_factory_emits_subdag_nodes(tmp_path):
    """When subdag_factory is set, the wrapper DAG should use SUBDAG
    EXTERNAL nodes instead of JOB nodes; the factory is invoked once
    per (sim, level)."""
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="condor",
                            run_queue_kind="condor")
    archive = Archive(base_location=base, manifest=manifest, generator_spec=_gen)

    factory_calls = []
    def fake_factory(arch, sim_name, level):
        factory_calls.append((sim_name, level))
        out = arch.base / "fake_{}_{}.dag".format(sim_name, level)
        out.write_text("# noop\n")
        return str(out)

    rq = DualCondorRunQueue(subdag_factory=fake_factory, submit_mode="embed")
    n1 = archive.register(0.1, target_level=2)
    n2 = archive.register(0.2, target_level=1)
    rq.submit(archive, [n1, n2])

    # Factory invoked for every (sim, level) pair.
    assert sorted(factory_calls) == sorted([(n1, 1), (n1, 2), (n2, 1)])

    # Wrapper DAG uses SUBDAG EXTERNAL, not JOB.
    wrapper = Path(rq.last_wrapper_dag_path)
    text = wrapper.read_text()
    assert "SUBDAG EXTERNAL {}_lvl1".format(n1) in text
    assert "SUBDAG EXTERNAL {}_lvl2".format(n1) in text
    assert "SUBDAG EXTERNAL {}_lvl1".format(n2) in text
    assert "JOB " not in text, "should not emit JOB nodes when factory is set"
    # PARENT/CHILD chains for n1 are still emitted.
    assert "PARENT {}_lvl1 CHILD {}_lvl2".format(n1, n1) in text


def test_submit_mode_embed_does_not_dispatch(tmp_path, monkeypatch):
    """submit_mode='embed' must NOT call condor_submit_dag; the wrapper
    path is on self.last_wrapper_dag_path."""
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="condor",
                            run_queue_kind="condor")
    archive = Archive(base_location=base, manifest=manifest, generator_spec=_gen)
    rq = DualCondorRunQueue(submit_mode="embed")

    import RIFT.simulation_manager.database as db
    calls = []
    def panic(*a, **kw):
        calls.append(a)
        raise AssertionError("condor_submit_dag must not be invoked in embed mode")
    monkeypatch.setattr(db.subprocess, "run", panic)

    n = archive.register(0.5, target_level=2)
    results = rq.submit(archive, [n])
    assert results == [(n, "")]                # cluster id empty in embed mode
    assert rq.last_wrapper_dag_path is not None
    assert Path(rq.last_wrapper_dag_path).exists()
    assert calls == []                         # subprocess.run never called


def test_submit_mode_default_still_dispatches(tmp_path, monkeypatch):
    """submit_mode default is 'submit': condor_submit_dag should be called."""
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="condor",
                            run_queue_kind="condor")
    archive = Archive(base_location=base, manifest=manifest, generator_spec=_gen)
    rq = DualCondorRunQueue()  # defaults

    import RIFT.simulation_manager.database as db
    calls = []
    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        class R:
            stdout = "submitted to cluster 4242.\n"
            stderr = ""
            returncode = 0
        return R()
    monkeypatch.setattr(db.subprocess, "run", fake_run)

    n = archive.register(0.5, target_level=1)
    results = rq.submit(archive, [n])
    assert results == [(n, "4242")]
    assert any("condor_submit_dag" in c[0] for c in calls), calls


def test_invalid_submit_mode_raises():
    with pytest.raises(ValueError):
        DualCondorRunQueue(submit_mode="bogus")


# ---------------------------------------------------------------------------
# SlurmRunQueue submit-script generation
# ---------------------------------------------------------------------------

def test_slurm_run_queue_submit_script_basics(tmp_path):
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="slurm",
                            run_queue_kind="slurm")
    archive = Archive(base_location=base, manifest=manifest, generator_spec=_gen)
    rq = SlurmRunQueue(partition="compute", time_limit="00:05:00",
                       request_memory=1024,
                       prelude="module load python/3.11")
    n = archive.register(0.7, target_level=2)
    sub = Path(rq.build_worker(archive, n, level=2))
    text = sub.read_text()
    # Sbatch directives
    assert "#SBATCH --job-name={}_lvl2".format(n) in text
    assert "#SBATCH --partition=compute" in text
    assert "#SBATCH --time=00:05:00" in text
    assert "#SBATCH --mem=1024M" in text
    # Cd to sim_dir, set up code-dir, exec bootstrap
    assert "cd {}".format(archive.sim_dir(n)) in text
    assert "module load python/3.11" in text
    assert "--code-dir {}".format(archive.base / "code") in text
    assert "--sim-name {}".format(n) in text
    assert "--level 2" in text
    # Level 2 has level_1.json as a prev-level (regardless of disk presence).
    assert "--prev-levels level_1.json" in text


def test_slurm_per_sim_overrides_appear_in_sbatch(tmp_path):
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="slurm",
                            run_queue_kind="slurm")
    archive = Archive(base_location=base, manifest=manifest, generator_spec=_gen)
    rq = SlurmRunQueue(partition="default", request_memory=2048)
    n = archive.register(0.5)
    archive.set_resources(n, request_memory=12345)
    sub = Path(rq.build_worker(archive, n, level=1))
    text = sub.read_text()
    assert "#SBATCH --mem=12345M" in text


def test_slurm_submit_chains_with_dependency_afterok(tmp_path, monkeypatch):
    """SlurmRunQueue.submit() should pass --dependency=afterok:<prev> for
    every level after the first within a sim."""
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="slurm",
                            run_queue_kind="slurm")
    archive = Archive(base_location=base, manifest=manifest, generator_spec=_gen)
    rq = SlurmRunQueue(partition="compute")

    captured = []
    counter = [1000]

    def fake_run(cmd, *a, **kw):
        captured.append(list(cmd))
        counter[0] += 1
        class R:
            stdout = "{}\n".format(counter[0])
            stderr = ""
            returncode = 0
        return R()

    import RIFT.simulation_manager.database as db
    monkeypatch.setattr(db.subprocess, "run", fake_run)

    n = archive.register(0.5, target_level=3)
    rq.submit(archive, [n])

    # Three sbatch invocations; only the 2nd and 3rd carry --dependency.
    assert len(captured) == 3, captured
    assert not any(a.startswith("--dependency=") for a in captured[0])
    assert any(a == "--dependency=afterok:1001" for a in captured[1]), captured[1]
    assert any(a == "--dependency=afterok:1002" for a in captured[2]), captured[2]


def test_slurm_submit_handles_missing_sbatch(tmp_path, monkeypatch):
    """When sbatch isn't on PATH, _sbatch logs a warning and returns
    None; submit() records no jobids but doesn't crash."""
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="slurm",
                            run_queue_kind="slurm")
    archive = Archive(base_location=base, manifest=manifest, generator_spec=_gen)
    rq = SlurmRunQueue(partition="compute")

    import RIFT.simulation_manager.database as db
    def fnf(*a, **kw):
        raise FileNotFoundError("sbatch")
    monkeypatch.setattr(db.subprocess, "run", fnf)

    n = archive.register(0.5, target_level=2)
    results = rq.submit(archive, [n])
    # No crash; per-sim entry recorded with empty jobid.
    assert results == [(n, "")]
    # Sub script still written to disk.
    assert (archive.base / "run_queue" / "submit_files"
            / "{}_lvl1.sh".format(n)).exists()


def test_make_queues_from_manifest_picks_slurm(tmp_path):
    base = tmp_path / "x"
    manifest = Manifest.new(name="x", request_queue_kind="slurm",
                            run_queue_kind="slurm",
                            run_queue_extra={"partition": "main",
                                             "time_limit": "00:30:00",
                                             "request_memory": 8192})
    archive = Archive(base_location=base, manifest=manifest, generator_spec=_gen)
    req_q, run_q = make_queues_from_manifest(archive)
    assert isinstance(req_q, SlurmRequestQueue)
    assert isinstance(run_q, SlurmRunQueue)
    assert run_q.partition == "main"
    assert run_q.time_limit == "00:30:00"
    assert run_q.request_memory == 8192
    assert req_q.run_queue is run_q


# ---------------------------------------------------------------------------
# Unstick + bump-memory
# ---------------------------------------------------------------------------

def test_unstick_clears_stuck_status(tmp_path):
    archive = make_archive(tmp_path)
    n = archive.register(0.5, target_level=2)
    # Force a stuck state with prior level present.
    sd = archive.sim_dir(n)
    (sd / "level_1.json").write_text('{"level":1}')
    archive.refresh_status_from_disk()        # current_level -> 1, refine_ready
    archive.transition(n, "stuck")
    assert archive.get_status(n) == "stuck"
    archive.unstick(n)
    # level >= 1 -> goes back to refine_ready
    assert archive.get_status(n) == "refine_ready"


def test_unstick_with_bump_memory_raises_request_memory(tmp_path):
    archive = make_archive(tmp_path)
    n = archive.register(0.5)
    archive.set_resources(n, request_memory=4000)
    archive.transition(n, "stuck")
    archive.unstick(n, bump_memory=True, bump_factor=1.5)
    res = archive.get_resources(n)
    assert res["request_memory"] == 6000


# ---------------------------------------------------------------------------
# Admin: resummarize / verify / rebuild_index
# ---------------------------------------------------------------------------

def test_resummarize_all_updates_summary(tmp_path):
    archive = make_archive(tmp_path)
    n = archive.register(0.5, target_level=1)
    archive.request_queue.submit_pending(archive)
    # Stomp the summary on disk; resummarize should fix it.
    (archive.sim_dir(n) / "summary.json").write_text('{"value": 999}')
    archive.update_summary(n, {"value": 999})
    report = archive.resummarize_all()
    assert report[n] == "ok"
    summary = archive.index.by_name(n)["summary"]
    # _summ produces value = 0.5 * sqrt(2) * 1
    import math
    assert abs(summary["value"] - 0.5 * math.sqrt(2.0)) < 1e-12


def test_verify_clean_archive(tmp_path):
    archive = make_archive(tmp_path)
    archive.register(0.5, target_level=1)
    archive.request_queue.submit_pending(archive)
    report = archive.verify()
    assert report["healthy"] is True


def test_verify_detects_missing_index_row(tmp_path):
    archive = make_archive(tmp_path)
    n = archive.register(0.5, target_level=1)
    archive.request_queue.submit_pending(archive)
    # Manually remove the index row.
    archive.index.remove(n)
    report = archive.verify()
    assert n in report["missing_in_index"]
    assert report["healthy"] is False


def test_rebuild_index_recovers_from_corruption(tmp_path):
    archive = make_archive(tmp_path)
    n1 = archive.register(0.1, target_level=1)
    n2 = archive.register(0.2, target_level=1)
    archive.request_queue.submit_pending(archive)
    # Wipe index.jsonl.
    archive.index.path.unlink()
    n_rebuilt = archive.rebuild_index()
    assert n_rebuilt == 2
    assert archive.index.by_name(n1) is not None
    assert archive.index.by_name(n2) is not None


# ---------------------------------------------------------------------------
# Concurrent register
# ---------------------------------------------------------------------------

def test_concurrent_register_is_safe(tmp_path):
    """N threads register N distinct params each. With locking, all 4N
    registrations should appear with no collisions or lost rows."""
    archive = make_archive(tmp_path)

    N = 25
    threads = 4

    def worker(shift):
        for i in range(N):
            archive.register(float(shift * 1000 + i))

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert len(archive.index.all()) == threads * N
    names = {row["name"] for row in archive.index.all()}
    assert len(names) == threads * N


# ---------------------------------------------------------------------------
# __main__: pytest fallback
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "-v" not in " ".join(sys.argv[1:]):
        sys.argv.append("-v")
    sys.argv.append("-rs")
    sys.exit(pytest.main(args=[__file__] + sys.argv[1:]))
