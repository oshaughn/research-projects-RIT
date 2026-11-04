"""ILE refuses an early-time multirate configuration it cannot honour before its event loop, with a
nonzero exit.  Inside analyze_event the same refusal is caught by the per-point handler, which
prints FAILED ANALYSIS and exits 0 with no output (review of research-projects-RIT #436)."""
import json
import os
import subprocess
import sys

import lal
import lalsimulation as lalsim
import pytest

import RIFT.lalsimutils as lsu
from RIFT.likelihood import factored_likelihood_multirate as flm

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ILE = os.path.join(CODE, "bin", "integrate_likelihood_extrinsic_batchmode")
SCHED = dict(kind="two_rate", fs_e=128, tau_tr=20.0, guard=1.0, atten_db=160.0, late_start=22.0,
             late_buffer=32, post=2, uv_margin=16.0, mc_range=[28.0, 28.6])
# Blocks one module import, then runs ILE as __main__: reproduces a tree without #465.
_RUNNER = r"""
import sys, runpy, importlib.abc
block = sys.argv.pop(1)
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if block and name == block:
            raise ImportError("No module named %r (blocked by test)" % name)
sys.meta_path.insert(0, _Block())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _sched(tmp_path, **kw):
    p = tmp_path / "sched.json"
    p.write_text(json.dumps(dict(SCHED, **kw)))
    return str(p)


def _ile(tmp_path, extra, sched=None, block=""):
    # Data and PSD files do not exist: the refusal must come before anything reads them.
    argv = [ILE, "--time-marginalization", "--vectorized", "--gpu", "--force-xpy",
            "--early-time-multirate", sched or _sched(tmp_path),
            "--event-time", "1000000000", "--data-start-time", "999999976", "--data-end-time", "1000000008",
            "--cache-file", str(tmp_path / "absent.cache"), "--channel-name", "H1=FAKE-STRAIN",
            "--psd-file", "H1=" + str(tmp_path / "absent_psd.xml.gz"),
            "--fmin-template", "20", "--reference-freq", "20", "--fmax", "1700", "--srate", "4096",
            "--l-max", "4", "--d-max", "2000", "--n-max", "100", "--output-file", str(tmp_path / "out")] + extra
    env = dict(os.environ, PYTHONPATH=CODE + os.pathsep + os.environ.get("PYTHONPATH", ""),
               OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-c", _RUNNER, block] + argv, capture_output=True, text=True,
                          env=env, timeout=600, cwd=str(tmp_path))


def _assert_refused(r, reason):
    assert r.returncode != 0, r.stdout[-2000:]
    assert "refusing this job before analysis" in r.stderr, r.stderr[-2000:]
    assert reason in r.stderr, r.stderr[-2000:]
    assert "NOT READY FOR HIGH-SNR USE" in r.stderr, r.stderr[-2000:]
    assert "FAILED ANALYSIS" not in r.stdout


def test_ile_refuses_unsupported_approximant(tmp_path):
    r = _ile(tmp_path, ["--mass1", "36", "--mass2", "29", "--approximant", "IMRPhenomD"])
    _assert_refused(r, "IMRPhenomXHM/XPHM only")


def test_ile_refuses_in_plane_spin_grid_point(tmp_path):
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 36 * lal.MSUN_SI, 29 * lal.MSUN_SI
    P.s1x, P.approx = 0.3, lalsim.IMRPhenomXPHM
    xml = str(tmp_path / "grid.xml.gz")
    lsu.ChooseWaveformParams_array_to_xml([P], xml.replace(".xml.gz", ""))
    r = _ile(tmp_path, ["--sim-xml", xml, "--n-events-to-analyze", "1", "--approximant", "IMRPhenomXPHM"])
    _assert_refused(r, "in-plane spins")


def test_ile_refuses_xp_waveform_without_gpu_waveform_xhm(tmp_path):
    r = _ile(tmp_path, ["--mass1", "36", "--mass2", "29", "--approximant", "IMRPhenomXHM",
                        "--early-time-multirate-waveform", "xp"], block="RIFT.likelihood.gpu_waveform_xhm")
    _assert_refused(r, "#465")


def _P(**kw):
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2, P.approx = 36 * lal.MSUN_SI, 29 * lal.MSUN_SI, lalsim.IMRPhenomXHM
    for k, v in kw.items():
        setattr(P, k, v)
    return P


@pytest.mark.parametrize("sched_kw, P_kw, kw, msg", [
    (dict(fs_e=0), {}, {}, "finite and positive"),
    ({}, dict(m1=50 * lal.MSUN_SI), {}, "outside the schedule's frozen range"),
    ({}, dict(s2y=0.1), {}, "in-plane"),
    ({}, dict(approx=lalsim.IMRPhenomXPHM), dict(waveform="xp"), "IMRPhenomXHM only"),
    ({}, {}, dict(waveform="xp", backend="reference"), "non-reference"),
    (dict(kind="one_segment"), {}, dict(waveform="xp"), "two_rate"),
    ({}, {}, dict(Lmax=5), "Lmax"),
])
def test_validate_job_refusals(tmp_path, sched_kw, P_kw, kw, msg):
    Lmax = kw.pop("Lmax", 4)
    with pytest.raises(ValueError, match=msg):
        flm.validate_job(_sched(tmp_path, **sched_kw), [_P(), _P(**P_kw)], Lmax, **kw)


def test_validate_job_accepts_supported_points(tmp_path, capsys):
    sch = flm.validate_job(_sched(tmp_path), [_P(), _P(approx=lalsim.IMRPhenomXPHM, s1z=0.4)], 4)
    assert sch["kind"] == "two_rate"
    assert "PROTOTYPE, NOT READY FOR HIGH-SNR USE" in capsys.readouterr().err


def test_ile_refuses_td_inner_product_before_loading_data(tmp_path):
    r = _ile(tmp_path, ["--internal-ile-inner-product", "td-toeplitz"])
    assert r.returncode != 0
    assert "TD likelihood operators are not supported by multirate precomputation" in r.stderr
    assert "FAILED ANALYSIS" not in r.stdout
    assert not (tmp_path / "out").exists()
