#!/usr/bin/env python3
"""Driver for the live-HTCondor end-to-end test of RIFT.simulation_manager.

Workflow:
  1. Build a tiny worker script that reads a metadata file (1 float),
     computes k * sqrt(2), and writes the result to a given output path.
  2. Create SimulationArchiveOnLocalDiskIntegratedCondorQueue(base_location).
  3. Build the master submit file pointing at the worker.
  4. Register N sims, generate a DAG, submit it.
  5. Poll the schedd until all jobs leave the queue (or timeout).
  6. Re-instantiate the archive from disk; assert every sim is 'complete'
     and every retrieved value matches expectation.

Run via .travis/test-simulation-manager-condor.sh (which gates on
RIFT_TEST_CONDOR=1) or directly:
    python3 _simmgr_condor_e2e.py --base /tmp/sim_e2e --n-sims 3 --timeout 300
"""

import argparse
import math
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import time

import numpy as np

import RIFT.simulation_manager.BaseManager as bm
import RIFT.simulation_manager.CondorManager as cm

WORKER_TEMPLATE = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Trivial worker: reads a single float from the metadata file, computes
    k * sqrt(2), writes the result to the given output path.\"\"\"
    import math, sys
    if len(sys.argv) != 3:
        print('usage: worker.py <sim_path> <sim_meta>', file=sys.stderr)
        sys.exit(2)
    sim_path, sim_meta = sys.argv[1], sys.argv[2]
    with open(sim_meta) as f:
        k = float(f.read().strip())
    val = k * math.sqrt(2.0)
    with open(sim_path, 'w') as f:
        f.write('{:.17g}\\n'.format(val))
""")


def write_worker(base):
    worker_path = os.path.join(base, 'worker.py')
    with open(worker_path, 'w') as f:
        f.write(WORKER_TEMPLATE)
    st = os.stat(worker_path)
    os.chmod(worker_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return worker_path


def wait_for_completion(archive, timeout):
    """Poll until every sim is marked 'complete' or 'stuck', or until
    `timeout` seconds elapse. Returns the final status counts."""
    deadline = time.time() + timeout
    last_log = 0.0
    while True:
        archive.refresh_status_from_condor()
        archive.refresh_status_from_disk()
        outstanding = [
            n for n in archive.simulations
            if archive.get_status(sim_name=n) not in ('complete', 'stuck')]
        if not outstanding:
            break
        if time.time() - last_log > 5:
            counts = {}
            for n in archive.simulations:
                s = archive.get_status(sim_name=n)
                counts[s] = counts.get(s, 0) + 1
            print('  ... waiting; statuses:', counts, flush=True)
            last_log = time.time()
        if time.time() > deadline:
            print('TIMEOUT waiting for sims to complete', file=sys.stderr)
            for n in archive.simulations:
                print('  ', n, archive.get_status(sim_name=n))
            return False
        time.sleep(2)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True,
                    help='Archive base_location (will be created)')
    ap.add_argument('--n-sims', type=int, default=3)
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--keep', action='store_true',
                    help='Do not delete --base on success')
    args = ap.parse_args()

    if os.path.exists(args.base):
        shutil.rmtree(args.base)
    os.makedirs(args.base)

    worker_path = write_worker(args.base)
    print('Wrote worker:', worker_path)

    archive = cm.SimulationArchiveOnLocalDiskIntegratedCondorQueue(
        name='e2e', base_location=args.base)
    archive._internal_check_complete = cm.condor_check_complete

    # Build master submit. Override arg_str so the worker receives
    # <sim_path> <sim_meta>; both macros are populated by get_node_for_dag.
    archive.build_master_job(
        tag='e2e',
        exe=worker_path,
        request_memory=64,
        arg_str='$(macro_sim_path) $(macro_sim_meta)',
        universe='vanilla',
    )
    print('Built master submit file under', args.base)

    expected = {}
    for i in range(args.n_sims):
        k = 0.5 * (i + 1)
        sim_name = archive.generate_simulation(k)
        expected[sim_name] = k * math.sqrt(2.0)
    print('Registered', args.n_sims, 'sims:', list(archive.simulations.keys()))

    dag_path = archive.generate_dag_for_all_ready_simulations(tag='e2e_dag')
    if dag_path is None:
        print('FAIL: no DAG produced', file=sys.stderr)
        sys.exit(1)
    print('DAG written:', dag_path)

    if not archive.submit_dag(dag_path):
        print('FAIL: submit_dag returned False', file=sys.stderr)
        sys.exit(1)
    print('DAG submitted; cluster id =', archive._internal_dag_cluster_id)

    if not wait_for_completion(archive, args.timeout):
        sys.exit(2)

    # Re-instantiate from disk: every sim should rehydrate as 'complete'.
    archive2 = cm.SimulationArchiveOnLocalDiskIntegratedCondorQueue(
        name='e2e', base_location=args.base)
    bad = []
    for sim_name in archive.simulations:
        status = archive2.get_status(sim_name=sim_name)
        if status != 'complete':
            bad.append((sim_name, 'status=' + str(status)))
            continue
        sim_path = archive2.simulations[sim_name][1]
        try:
            val = float(open(sim_path).read().strip())
        except Exception as exc:
            bad.append((sim_name, 'read failed: ' + str(exc)))
            continue
        ref = expected[sim_name]
        if abs(val - ref) > 1e-12:
            bad.append((sim_name, 'value {} != expected {}'.format(val, ref)))

    if bad:
        print('FAIL:')
        for name, why in bad:
            print('  ', name, why)
        sys.exit(3)

    print('PASS:', args.n_sims, 'sims completed and verified.')
    if not args.keep:
        shutil.rmtree(args.base, ignore_errors=True)


if __name__ == '__main__':
    main()
