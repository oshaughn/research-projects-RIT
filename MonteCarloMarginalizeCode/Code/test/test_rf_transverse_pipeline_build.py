"""Build (never submit) a real pseudo_pipe DAG that opts in without --cip-fit-method.

The helper switches the unforced gp fit to rf. The DAG must then be built as an explicit
rf run: flat CIP workers, so no worker is handed both RF transverse coordinates and --fit-load-gp.
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


def _opt_in_ini(tmp_path, mode="geometric4", extra=(), chirpmass=None):
    text = REF_INI.read_text()
    # This contract builds a DAG with an empty placeholder cache; it performs
    # no strain IO. Automatic truncation must reject that cache in real runs.
    text = re.sub(r'(?m)^internal-truncate-files-for-osg-file-transfer=.*$',
                  'internal-truncate-files-for-osg-file-transfer=False', text)
    text = re.sub(r'(?m)^cip-fit-method=.*\n', '', text)
    text = re.sub(r'(?m)^cip-explode-jobs=.*$', '\n'.join(
        ['cip-explode-jobs=3', 'rf-transverse-spin-coordinates="{}"'.format(mode)] + list(extra)), text)
    if chirpmass:
        text = re.sub(r'(?m)^chirpmass-min = .*$', 'chirpmass-min = {}'.format(chirpmass[0]), text)
        text = re.sub(r'(?m)^chirpmass-max = .*$', 'chirpmass-max = {}'.format(chirpmass[1]), text)
    assert 'cip-fit-method' not in text
    ini = tmp_path / "opt_in.ini"
    ini.write_text(text)
    return ini


def _build(tmp_path, ini, coinc=COINC):
    (tmp_path / "foo.cache").write_text("")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
               # Source scripts first; then this interpreter for their `env python` shebangs.
               PATH=os.pathsep.join([str(ROOT / "bin"), os.path.dirname(sys.executable),
                                     env.get("PATH", "")]),
               OMP_NUM_THREADS="1", RIFT_LOWLATENCY="True",
               SINGULARITY_RIFT_IMAGE="foo", SINGULARITY_BASE_EXE_DIR="/usr/bin/")
    rundir = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bin" / "util_RIFT_pseudo_pipe.py"), "--use-ini", str(ini),
         "--use-coinc", str(coinc), "--use-rundir", str(rundir),
         "--fake-data-cache", str(tmp_path / "foo.cache")],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=900)
    return proc, rundir


@pytest.mark.parametrize('mode',['geometric4','geometric4-phase-excess'])
def test_unforced_opt_in_builds_flat_cip_workers(tmp_path,mode):
    pytest.importorskip("lal")
    proc, rundir = _build(tmp_path, _opt_in_ini(tmp_path,mode))
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "--cip-explode-jobs-flat" in proc.stdout
    subs = {p.name: p.read_text() for p in rundir.glob("CIP*.sub")}
    workers = [n for n in subs if n.startswith("CIP_worker")]
    assert workers
    chart = "--parameter-implied rf_phase_excess" if mode == "geometric4-phase-excess" else "--parameter-implied rf_total_perp"
    activated = [n for n in workers if chart in subs[n]]
    assert activated, sorted(subs)
    for name, sub in subs.items():
        assert "--rf-transverse-spin-coordinates" not in sub, name
        assert not ("rf_total_azimuth" in sub and "--fit-load-gp" in sub), name
    for name in workers:
        assert "--fit-method rf" in subs[name], name
