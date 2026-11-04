"""
test_selfterm_basis.py -- verify the low-rank SVD basis expansion of the per-
realization |C_c|^2-weighted cross terms (BuildCalibrationSelfTermBasis +
CalibrationSelfTermCrossTermsFromBasis) reproduces the DIRECT per-draw band
integral U_c = <C_c h | C_c h>, on real (spline-drawn) calibration realizations.

This guards the cost-optimized path (rank << n_cal band integrals + cheap per-draw
combo) against the brute-force reference it replaces.  CPU-only, fast.

Run:  PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/Code python3 -m RIFT.calmarg.test_selfterm_basis
"""
from __future__ import print_function
import tempfile, os
import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.calmarg.generate_realizations as genr

fmin, fmax, seglen, srate = 20.0, 448.0, 16.0, 1024.0
deltaF = 1.0 / seglen
fNyq = srate / 2.0
len2side = int(2 * (int(fNyq / deltaF)))   # ComplexIP.len2side = 2*(len1side-1)

# two random smooth "hlm" series (len2side COMPLEX16FrequencySeries), 2 modes
rng = np.random.default_rng(7)
modes = [(2, 2), (2, -2)]
hlms = {}
for m in modes:
    h = lal.CreateCOMPLEX16FrequencySeries("h", lal.LIGOTimeGPS(0), -fNyq, deltaF,
                                           lsu.lsu_HertzUnit, len2side)
    z = (rng.standard_normal(len2side) + 1j * rng.standard_normal(len2side))
    h.data.data = z / (1.0 + np.arange(len2side))   # decaying, smooth-ish
    hlms[m] = h

psd = lalsim.SimNoisePSDaLIGOZeroDetHighPower
IP = lsu.ComplexIP(fmin, fmax, fNyq, deltaF, psd, analyticPSD_Q=True)

# real spline-drawn calibration realizations (width 8%), n_cal=120
log_f = np.linspace(np.log10(fmin), np.log10(fmax), 60)
env = np.zeros((60, 7)); env[:, 0] = 10 ** log_f
env[:, 1] = 1.0; env[:, 3] = 0.92; env[:, 4] = -0.08; env[:, 5] = 1.08; env[:, 6] = 0.08
ef = tempfile.mktemp(suffix=".txt"); np.savetxt(ef, env)
np.random.seed(3)
n_cal = 120
cal = genr.create_realizations(ef, seglen, 1.0 / srate, fmin, fmax, 10, n_cal)
os.remove(ef)
assert cal.shape[0] == IP.len2side, (cal.shape, IP.len2side)

# --- brute force: direct per-draw |C_c|^2-weighted band integral ---
base_w2 = IP.weights2side.copy()
pairs = [(modes[0], modes[0]), (modes[1], modes[1]), (modes[0], modes[1]), (modes[1], modes[0])]
U_brute = []
for c in range(n_cal):
    IP.weights2side = base_w2 * (np.abs(cal[:, c]) ** 2)
    U_brute.append({p: IP.ip(hlms[p[0]], hlms[p[1]]) for p in pairs})
IP.weights2side = base_w2

# --- basis path ---
basis = fl.BuildCalibrationSelfTermBasis(cal, base_w2, use_cache=False, verbose=True)
U_basis = fl.CalibrationSelfTermCrossTermsFromBasis(IP, hlms, hlms, basis,
                                                    prefix="U", same_waveform_Q=True)

err = 0.0
for c in range(n_cal):
    for p in pairs:
        err = max(err, abs(complex(U_basis[c][p]) - complex(U_brute[c][p])))
scale = max(abs(complex(U_brute[0][pairs[0]])), 1e-30)
rel = err / scale
print("basis rank=%d / n_cal=%d ; max|U_basis - U_brute| = %.3e (rel %.3e)"
      % (basis["rank"], n_cal, err, rel))
ok = rel < 1e-9
print("# RESULT:", "PASS" if ok else "MISMATCH")
raise SystemExit(0 if ok else 1)
