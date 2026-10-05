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


def _opt_in_ini(tmp_path, mode="physics3", extra=(), chirpmass=None):
    text = REF_INI.read_text()
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


def _low_mass_coinc(tmp_path, mass=13.8):
    # Same trigger, equal masses: detector chirp mass about 12 Msun, so auto activates.
    utils = pytest.importorskip("igwn_ligolw.utils")
    from igwn_ligolw import lsctables
    doc = utils.load_filename(str(COINC))
    for row in lsctables.SnglInspiralTable.get_table(doc):
        row.mass1 = row.mass2 = mass
        row.mtotal = 2 * mass
        row.mchirp = mass * 2 ** -0.2
    out = tmp_path / "coinc_low.xml"
    utils.write_filename(doc, str(out))
    return out


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


def test_unforced_opt_in_builds_flat_cip_workers(tmp_path):
    pytest.importorskip("lal")
    proc, rundir = _build(tmp_path, _opt_in_ini(tmp_path))
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


@pytest.mark.parametrize("option", ["cip-internal-use-eta-in-sampler=True",
                                    "hierarchical-merger-prior-1g=True"])
def test_physics3_refuses_coordinate_rewrites_before_the_helper(tmp_path, option):
    # These options replace delta_mc in every CIP stage after the helper runs.
    pytest.importorskip("lal")
    proc, rundir = _build(tmp_path, _opt_in_ini(tmp_path, extra=[option]))
    assert proc.returncode != 0
    assert "incompatible with --" + option.split("=")[0] in proc.stdout, proc.stdout[-3000:]
    assert not (rundir / "helper_cip_arg_list.txt").exists()


def test_auto_refuses_eta_sampler_once_it_activates(tmp_path):
    # auto activation depends on the event mass, so pseudo_pipe checks the final CIP lines.
    pytest.importorskip("lal")
    ini = _opt_in_ini(tmp_path, "auto", ["cip-internal-use-eta-in-sampler=True"], (8, 16))
    proc, rundir = _build(tmp_path, ini, _low_mass_coinc(tmp_path))
    assert proc.returncode != 0
    assert "activated but CIP would refuse it" in proc.stdout, proc.stdout[-3000:]
    assert "--cip-internal-use-eta-in-sampler" in proc.stdout
    assert not (rundir / "args_cip_list.txt").exists()
