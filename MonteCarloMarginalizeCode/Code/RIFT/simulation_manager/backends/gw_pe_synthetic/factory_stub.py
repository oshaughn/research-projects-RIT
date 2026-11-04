"""Stub `subdag_factory` for the GW PE synthetic-targeted backend.

Produces a tiny no-op sub-DAG that just creates the expected
`level_<N>.json` file in `<sim_dir>` (so `Archive.refresh_status_from_disk`
treats the sim as complete). Use this for development and smoke tests
that need to exercise the full archive flow without lalsuite or RIFT
PE.

Swap this for `factory_pseudo_pipe.subdag_factory` to do real work.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path


def subdag_factory(archive, sim_name: str, level: int) -> str:
    """Write a no-op SUBDAG and return its path. The DAG contains one
    JOB whose action is `cat > level_<N>.json <<EOF ... EOF`, mimicking
    what the real pseudo_pipe-built DAG's terminal node would write."""
    sd = archive.sim_dir(sim_name)
    rundir = sd / "level_{}_stub".format(level)
    rundir.mkdir(parents=True, exist_ok=True)

    # The "fake worker" is a shell that drops level_<N>.json into sim_dir.
    fake_sh = rundir / "fake_worker.sh"
    fake_sh.write_text(textwrap.dedent('''\
        #!/bin/bash
        set -euo pipefail
        cat > {sd}/level_{level}.json <<'EOF'
        {{"level": {level}, "stub": true, "sim_name": "{sim_name}"}}
        EOF
    ''').format(sd=str(sd), level=int(level), sim_name=sim_name))
    fake_sh.chmod(0o755)

    # Submit description for the one fake JOB.
    log_dir = archive.base / "run_queue" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = "stub_{}_lvl{}".format(sim_name, level)
    sub = rundir / (tag + ".sub")
    sub.write_text(textwrap.dedent('''\
        universe                = vanilla
        executable              = {fake_sh}
        log                     = {log_dir}/{tag}.log
        output                  = {log_dir}/{tag}.out
        error                   = {log_dir}/{tag}.err
        getenv                  = True
        request_memory          = 256M
        queue 1
    ''').format(fake_sh=str(fake_sh), log_dir=str(log_dir), tag=tag))

    # The sub-DAG: one JOB.
    dag = rundir / (tag + ".dag")
    dag.write_text("JOB {tag} {sub}\n".format(tag=tag, sub=str(sub)))
    return str(dag)


def execute_stub_inline(archive, sim_name: str, level: int) -> None:
    """Helper: simulate the no-op sub-DAG completing without involving
    condor at all. Drops the expected level_<N>.json into sim_dir.
    Useful for tests that want to step past the dispatch boundary
    deterministically."""
    sd = archive.sim_dir(sim_name)
    out = sd / "level_{}.json".format(level)
    out.write_text(json.dumps({
        "level": int(level), "stub": True, "sim_name": sim_name,
        "_note": "written by execute_stub_inline (no condor)",
    }))
