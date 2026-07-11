"""
Gate 1a: JAX response-coefficient ports vs the numpy references, to ~1e-12.

Validates (no heavy precompute needed -- pure analytic algebra):
  * response_slowrot.rotation_coefficients_dict   vs
      factored_likelihood_with_rotation.rotation_coefficients_vector   (Path A & B)
  * response_freqresponse.response_coefficients_dict vs
      factored_likelihood_freqresponse.response_coefficients           (Path D)

over H1/L1/V1 and random (RA,DEC,psi).

Run:
  PYTHONPATH=<...>/Code  taskset -c 0-3 python test/jax/test_jax_slowrot_coeffs.py
"""
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import lal
import lalsimulation as lalsim

import RIFT.likelihood.factored_likelihood_with_rotation as flwr
import RIFT.likelihood.factored_likelihood_freqresponse as ffr
import RIFT.likelihood.slowrot_freqresponse as sfr
from RIFT.likelihood.jax_ile import response_slowrot as rs
from RIFT.likelihood.jax_ile import response_freqresponse as rf

DETS = ["H1", "L1", "V1"]
TREF = 1126259462.0
HARM = (-2, -1, 0, 1, 2)


def _gmst(tref):
    return float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(tref))))


def test_rotation_coefficients():
    rng = np.random.default_rng(3)
    S = 64
    RA = rng.uniform(0, 2 * np.pi, S)
    DEC = np.arcsin(rng.uniform(-1, 1, S))
    psi = rng.uniform(0, np.pi, S)
    gmst = _gmst(TREF)
    worst = 0.0
    for p_max in (0, 1, 2):
        for det in DETS:
            lald = lalsim.DetectorPrefixToLALDetector(det)
            resp = np.asarray(lald.response, dtype=float)
            loc = np.asarray(lald.location, dtype=float)
            C_np = flwr.rotation_coefficients_vector(det, RA, DEC, psi, TREF, p_max)
            C_jx = rs.rotation_coefficients_dict(resp, loc, RA, DEC, psi, gmst, p_max)
            # compare the union of keys
            keys = set(C_np) | set(C_jx)
            for k in keys:
                a = np.asarray(C_np.get(k, np.zeros(S, complex)))
                b = np.asarray(C_jx.get(k, np.zeros(S, complex)))
                d = np.max(np.abs(a - b))
                worst = max(worst, d)
    print("[rotation coeff] max|jax-np| over dets/p_max = %.3e" % worst)
    assert worst < 1e-11, "rotation coefficient mismatch %g" % worst


def test_freqresponse_coefficients():
    rng = np.random.default_rng(7)
    S = 64
    RA = rng.uniform(0, 2 * np.pi, S)
    DEC = np.arcsin(rng.uniform(-1, 1, S))
    psi = rng.uniform(0, np.pi, S)
    gmst = _gmst(TREF)
    Qmax = 4
    worst = 0.0
    for L_arm in (None, 40000.0):     # native LIGO arm and a 40-km CE arm
        for det in DETS:
            resp, x_arm, y_arm, L = sfr.detector_geometry(det, L_arm=L_arm)
            b_jx = rf.response_coefficients_dict(resp, x_arm, y_arm, RA, DEC, psi,
                                                 gmst, Qmax)
            # numpy reference is scalar -> loop the S samples
            b_np = {p: np.empty(S, complex) for p in range(Qmax + 2)}
            for i in range(S):
                bi = ffr.response_coefficients(det, float(RA[i]), float(DEC[i]),
                                               float(psi[i]), TREF, Qmax, L_arm=L_arm)
                for p in range(Qmax + 2):
                    b_np[p][i] = bi[p]
            for p in range(Qmax + 2):
                d = np.max(np.abs(np.asarray(b_jx[p]) - b_np[p]))
                worst = max(worst, d)
    print("[freqresponse coeff] max|jax-np| over dets/L = %.3e" % worst)
    assert worst < 1e-11, "freqresponse coefficient mismatch %g" % worst


if __name__ == "__main__":
    test_rotation_coefficients()
    test_freqresponse_coefficients()
    print("COEFFICIENT PORTS VALIDATED (Gate 1a PASSED)")
