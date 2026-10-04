"""Build (never submit) a real pseudo_pipe DAG that opts in without --cip-fit-method.

The helper switches the unforced gp fit to rf. The DAG must then be built as an explicit
rf run: flat CIP workers, so no worker is handed both physics3 and --fit-load-gp.
"""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REF_INI = REPO / ".travis" / "ref_ini" / "GW150914.ini"
COINC = REPO / ".travis" / "ref_ini" / "coinc.xml"


def test_unforced_opt_in_builds_flat_cip_workers(tmp_path):
    pytest.importorskip("lal")
    text = REF_INI.read_text()
    text = re.sub(r'(?m)^cip-fit-method=.*\n', '', text)
    text = re.sub(r'(?m)^cip-explode-jobs=.*$',
                  'cip-explode-jobs=3\nrf-transverse-spin-coordinates="physics3"', text)
    assert 'cip-fit-method' not in text
    ini = tmp_path / "opt_in.ini"
    ini.write_text(text)
    (tmp_path / "foo.cache").write_text("")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
               PATH=str(ROOT / "bin") + os.pathsep + env.get("PATH", ""),
               OMP_NUM_THREADS="1", RIFT_LOWLATENCY="True",
               SINGULARITY_RIFT_IMAGE="foo", SINGULARITY_BASE_EXE_DIR="/usr/bin/")
    rundir = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bin" / "util_RIFT_pseudo_pipe.py"), "--use-ini", str(ini),
         "--use-coinc", str(COINC), "--use-rundir", str(rundir),
         "--fake-data-cache", str(tmp_path / "foo.cache")],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "--cip-explode-jobs-flat" in proc.stdout
    subs = {p.name: p.read_text() for p in rundir.glob("CIP*.sub")}
    workers = [n for n in subs if n.startswith("CIP_worker")]
    assert workers
    activated = [n for n in workers if "--rf-transverse-spin-coordinates physics3" in subs[n]]
    assert activated, sorted(subs)
    for name, sub in subs.items():
        assert not ("physics3" in sub and "--fit-load-gp" in sub), name
    for name in workers:
        assert "--fit-method rf" in subs[name], name
