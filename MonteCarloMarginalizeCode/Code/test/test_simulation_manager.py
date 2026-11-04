"""Lightweight tests for RIFT.simulation_manager.

These tests do NOT submit anything to a real schedd. They exercise:
  * the in-memory and on-disk archive round trips
  * the queue lifecycle FSM (ready -> submit_ready -> running -> complete -> stuck)
  * register_simulation: deferred semantics (no inline generator call)
  * SimulationArchiveOnLocalDiskIntegratedCondorQueue.build_master_job /
    generate_dag_for_all_ready_simulations: produces a .sub and .dag and
    transitions statuses correctly.

The condor end-to-end test that DOES submit lives in
.travis/test-simulation-manager-condor.sh and is gated on
RIFT_TEST_CONDOR=1.

Run directly:
    python test_simulation_manager.py
or under pytest:
    pytest test_simulation_manager.py -v
"""

from __future__ import print_function

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

from RIFT.simulation_manager import BaseManager as bm

# Optional: only meaningful if glue.pipeline is importable.
try:
    from RIFT.simulation_manager import CondorManager as cm
    _has_glue = getattr(cm, 'has_glue_pipeline', False)
except Exception as _exc:                                            # pragma: no cover
    cm = None
    _has_glue = False


# ---------------------------------------------------------------------------
# In-memory archive
# ---------------------------------------------------------------------------

def test_simulation_archive_roundtrip_in_memory():
    archive = bm.SimulationArchive(name="t")
    archive.generator = lambda k, **kw: np.array([k, k * 2])
    archive.generate_simulation(0.7)
    assert archive.simulation_exists_q(0.7)
    out = archive.retrieve_simulation(0.7)
    assert out.tolist() == [0.7, 1.4]


# ---------------------------------------------------------------------------
# On-disk archive
# ---------------------------------------------------------------------------

def test_local_disk_archive_writes_and_rehydrates(tmp_path):
    base = str(tmp_path / "disk_archive")

    def gen(k, **kwargs):
        return np.linspace(0, k, 5)

    a1 = bm.SimulationArchiveOnLocalDisk(name="t", base_location=base)
    a1.generator = gen
    a1.generate_simulation(0.5)
    a1.generate_simulation(1.5)
    assert a1.simulation_exists_q(0.5)
    assert os.path.isfile(os.path.join(base, "1"))
    assert os.path.isfile(os.path.join(base, "metadata_1"))

    # Re-instantiate; should rehydrate from disk.
    a2 = bm.SimulationArchiveOnLocalDisk(name="t", base_location=base)
    a2.load_simulation = np.loadtxt
    # The on-disk archive uses the file basename as the dict key.
    assert "1" in a2.simulations and "2" in a2.simulations


# ---------------------------------------------------------------------------
# Queue FSM
# ---------------------------------------------------------------------------

def test_queue_fsm_register_does_not_run_generator(tmp_path):
    base = str(tmp_path / "queue_no_run")
    calls = []

    def gen(k, **kwargs):
        calls.append(k)
        return np.array([1, 2, 3])

    archive = bm.SimulationArchiveOnLocalDiskExternalQueue(
        name="t", base_location=base)
    archive.generator = gen
    sim_name = archive.register_simulation(0.42)

    # No call into the generator.
    assert calls == []
    # Metadata file written, output file NOT written.
    assert os.path.isfile(os.path.join(base, "metadata_" + sim_name))
    sim_path = archive.simulations[sim_name][1]
    assert not os.path.exists(sim_path)
    # Status is 'ready'.
    assert archive.get_status(sim_name=sim_name) == 'ready'


def test_queue_fsm_status_transitions(tmp_path):
    base = str(tmp_path / "queue_fsm")
    archive = bm.SimulationArchiveOnLocalDiskExternalQueue(
        name="t", base_location=base)
    archive.generator = lambda k, **kw: np.array([k])
    archive._internal_check_complete = bm.output_file_exists_check

    n1 = archive.register_simulation(0.1)
    n2 = archive.register_simulation(0.2)
    assert sorted(archive.simulations_with_status('ready')) == sorted([n1, n2])

    archive.set_status('submit_ready', sim_name=n1)
    archive.set_status('running', sim_name=n1)
    assert archive.simulations_with_status('running') == [n1]
    assert archive.simulations_with_status('ready') == [n2]

    # Simulate worker writing output for n1.
    sim_path = archive.simulations[n1][1]
    with open(sim_path, 'w') as f:
        f.write("0.1\n")
    promoted = archive.refresh_status_from_disk()
    assert promoted == 1
    assert archive.get_status(sim_name=n1) == 'complete'

    # Invalid status rejected.
    with pytest.raises(ValueError):
        archive.set_status('bogus', sim_name=n2)


def test_queue_fsm_rehydrates_status_from_disk(tmp_path):
    base = str(tmp_path / "queue_resume")
    a1 = bm.SimulationArchiveOnLocalDiskExternalQueue(
        name="t", base_location=base)
    a1.register_simulation(0.1)
    n2 = a1.register_simulation(0.2)
    # "Worker" writes only n2's output.
    with open(a1.simulations[n2][1], 'w') as f:
        f.write("0.2\n")

    a2 = bm.SimulationArchiveOnLocalDiskExternalQueue(
        name="t", base_location=base)
    statuses = {k: a2.get_status(sim_name=k) for k in a2.simulations}
    # We can't predict ordering of fname enumeration; just count.
    counts = {'ready': 0, 'complete': 0}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1
    assert counts.get('complete', 0) == 1
    assert counts.get('ready', 0) == 1


def test_register_simulation_is_idempotent(tmp_path):
    base = str(tmp_path / "idem")
    archive = bm.SimulationArchiveOnLocalDiskExternalQueue(
        name="t", base_location=base)
    n1 = archive.register_simulation(0.7)
    n1_again = archive.register_simulation(0.7)
    assert n1 == n1_again
    assert len(archive.simulations) == 1


# ---------------------------------------------------------------------------
# Condor integrated queue (no schedd contact)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_glue,
                    reason="glue.pipeline not available")
def test_condor_build_master_and_dag(tmp_path):
    base = str(tmp_path / "condor_no_schedd")
    archive = cm.SimulationArchiveOnLocalDiskIntegratedCondorQueue(
        name="t", base_location=base)
    # exe is intentionally a path that need not exist for build_master_job;
    # we are not submitting.
    archive.build_master_job(tag="me", exe="/bin/echo", request_memory=4)

    sub_path = os.path.join(base, "me.sub")
    assert os.path.isfile(sub_path)
    with open(sub_path) as f:
        sub_text = f.read()
    # Spot-checks on the submit file content.
    assert "executable" in sub_text.lower()
    assert "request_memory" in sub_text.lower()

    n1 = archive.generate_simulation(0.1)
    n2 = archive.generate_simulation(0.2)
    assert archive.get_status(sim_name=n1) == 'ready'
    dag_path = archive.generate_dag_for_all_ready_simulations(tag="t_dag")
    assert dag_path is not None
    assert os.path.isfile(dag_path)
    # Both entries should now be in submit_ready.
    assert sorted(archive.simulations_with_status('submit_ready')) == sorted([n1, n2])
    assert archive.simulations_with_status('ready') == []


@pytest.mark.skipif(not _has_glue,
                    reason="glue.pipeline not available")
def test_condor_generate_dag_returns_none_when_no_ready_sims(tmp_path):
    base = str(tmp_path / "condor_empty")
    archive = cm.SimulationArchiveOnLocalDiskIntegratedCondorQueue(
        name="t", base_location=base)
    archive.build_master_job(tag="me", exe="/bin/echo", request_memory=4)
    assert archive.generate_dag_for_all_ready_simulations(tag="empty_dag") is None


# ---------------------------------------------------------------------------
# Standalone __main__ runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "-v" not in " ".join(sys.argv[1:]):
        sys.argv.append("-v")
    sys.argv.append("-rs")
    sys.exit(pytest.main(args=[__file__] + sys.argv[1:]))
