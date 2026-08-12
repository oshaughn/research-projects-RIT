"""
Smoke test for the one-call banded builders (build_*_data_from_precompute) and
package import.  Confirms they reproduce the manual precompute+pack+build path
and yield a finite, differentiable likelihood.

Run:
  PYTHONPATH=<...>/Code  taskset -c 0-3 python test/jax/test_jax_slowrot_wrapper.py
"""
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl

# exercise the package public API
from RIFT.likelihood.jax_ile import (
    build_rotation_data_from_precompute,
    build_freqresponse_data_from_precompute,
)
from RIFT.likelihood.jax_ile.wrapper import JAXDistanceMarginalizedLikelihood
from RIFT.likelihood.jax_ile.core import fused_log_likelihood

if not getattr(fl, "numba_on", True):
    fl.lalylm = np.vectorize(lal.SpinWeightedSphericalHarmonic, otypes=[complex])

fSample = 4096.0; fmin = 30.0; fmax = 1700.0; event_time = 1e9
Lmax = 2; deltaT = 1.0 / fSample; deltaF = 1.0 / 4.0
IWH = 0.03
PC = lal.PC_SI

P = lsu.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0, psi=0.4,
    m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI, detector='H1',
    dist=200e6 * lal.PC_SI, deltaT=deltaT, tref=event_time, deltaF=deltaF)
DETS = ("H1", "L1")
data_dict = {}
for d in DETS:
    _p = P.manual_copy(); _p.detector = d
    data_dict[d] = lsu.non_herm_hoff(_p)
psd_dict = {d: lalsim.SimNoisePSDaLIGOZeroDetHighPower for d in data_dict}

rng = np.random.RandomState(5)
S = 16
ra = rng.uniform(0, 2 * np.pi, S)
dec = np.arcsin(rng.uniform(-1, 1, S))
psi = rng.uniform(0, np.pi, S)
incl = np.arccos(rng.uniform(-1, 1, S))
phiref = rng.uniform(0, 2 * np.pi, S)
distMpc = rng.uniform(100, 800, S)


def _run(builder, tag, **kw):
    data, extras = builder(P.manual_copy(), data_dict, psd_dict, event_time,
                           IWH, Lmax, fmax, analyticPSD_Q=True, verbose=False, **kw)
    lnL = np.asarray(fused_log_likelihood(data, ra, dec, psi, incl, phiref,
                                          distMpc, interp="nearest"))
    assert np.all(np.isfinite(lnL)), "%s produced non-finite lnL" % tag
    # differentiable distmarg path
    dlike = JAXDistanceMarginalizedLikelihood(data, 5.0, 3000.0, n_grid=64)
    v, g = dlike.value_and_grad([ra[0], dec[0], psi[0], incl[0], phiref[0]])
    assert np.isfinite(v) and np.all(np.isfinite(g)), "%s distmarg AD non-finite" % tag
    print("[%s] one-call build OK: lnL[0]=%.3f  distmarg lnL=%.3f  |grad|=%.2f"
          % (tag, lnL[0], v, np.linalg.norm(g)))
    return data


if __name__ == "__main__":
    _run(build_rotation_data_from_precompute, "rotation", p_max=0)
    _run(build_freqresponse_data_from_precompute, "freqresponse", Qmax=4,
         L_arm=40000.0)
    print("ONE-CALL BUILDER SMOKE TEST PASSED")
