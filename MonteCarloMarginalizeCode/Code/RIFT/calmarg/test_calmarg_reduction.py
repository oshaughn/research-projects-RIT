"""
Validate the n_cal>1 calibration-marginalization reduction in
DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop against a brute-force reference:
running the (unchanged) n_cal==1 path on each realization block separately and
combining with  lnL = logsumexp_c(lnL_c) - log(n_cal).

Runs entirely on CPU (xpy=np), so no GPU required.
"""
import numpy as np
import lal
from scipy.special import logsumexp
import RIFT.likelihood.factored_likelihood as fl

rng = np.random.default_rng(1234)

det = "H1"
n_lms = 2
N_window = 256         # per-realization buffer length (must exceed sky time-delay spread + npts)
npts = 16              # integration sub-window (len(tvals))
n_cal = 5
npts_extrinsic = 6
deltaT = 1.0 / 4096

# lookup table of (l,m) pairs
lookupNKDict = {det: np.array([[2, 2], [2, -2]], dtype=int)}

# concatenated rholm timeseries: (n_lms, N_window*n_cal); block c is realization c
npts_full = N_window * n_cal
rho = (rng.standard_normal((n_lms, npts_full)) + 1j*rng.standard_normal((n_lms, npts_full)))
rholmsArrayDict = {det: rho}

# template-template cross terms (Hermitian-ish; values don't matter for the identity)
U = (rng.standard_normal((n_lms, n_lms)) + 1j*rng.standard_normal((n_lms, n_lms)))
U = U + U.conj().T
V = (rng.standard_normal((n_lms, n_lms)) + 1j*rng.standard_normal((n_lms, n_lms)))
ctUArrayDict = {det: U}
ctVArrayDict = {det: V}

epochDict = {det: 0.0}

# extrinsic parameter vector (mock P_vec)
class PV: pass
P = PV()
P.phi = rng.uniform(0, 2*np.pi, npts_extrinsic)
P.theta = rng.uniform(0.2, np.pi-0.2, npts_extrinsic)
P.psi = rng.uniform(0, np.pi, npts_extrinsic)
P.incl = rng.uniform(0.2, np.pi-0.2, npts_extrinsic)
P.phiref = rng.uniform(0, 2*np.pi, npts_extrinsic)
P.dist = np.full(npts_extrinsic, 500.0) * (lal.PC_SI*1e6)   # 500 Mpc
P.tref = 1000000000.0
P.deltaT = deltaT
# Place the integration window near the middle of the buffer so ifirst stays in
# [0, N_window-npts] for all sky positions (TimeDelayFromEarthCenter is +-0.021s).
epochDict[det] = P.tref - 0.03

tvals = np.linspace(-npts//2*deltaT, npts//2*deltaT, npts)

# --- reference: per-block n_cal==1 evaluations, combined by hand ---
lnL_blocks = np.zeros((n_cal, npts_extrinsic))
for c in range(n_cal):
    block = rho[:, c*N_window:(c+1)*N_window].copy()
    lnL_blocks[c] = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P, lookupNKDict, {det: block}, ctUArrayDict, ctVArrayDict, epochDict,
        Lmax=2, xpy=np, n_cal=1)
lnL_ref = logsumexp(lnL_blocks, axis=0) - np.log(n_cal)

# --- new path: single call with n_cal>1 ---
lnL_new = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
    tvals, P, lookupNKDict, rholmsArrayDict, ctUArrayDict, ctVArrayDict, epochDict,
    Lmax=2, xpy=np, n_cal=n_cal)

print("reference lnL :", np.array(lnL_ref))
print("in-loop   lnL :", np.array(lnL_new))
maxerr = np.max(np.abs(np.array(lnL_new) - np.array(lnL_ref)))
print("max abs error :", maxerr)
assert maxerr < 1e-9, "MISMATCH: cal-marg reduction != brute-force reference"

# --- return_cal_components: raw per-realization integrated log L, (npts_extrinsic, n_cal) ---
# Collapsing it by hand must reproduce the cal-marg lnL:
#   lnL_marg(i) = logsumexp_c comp[i,c] - log(n_cal)   (uniform weights)
comp = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
    tvals, P, lookupNKDict, rholmsArrayDict, ctUArrayDict, ctVArrayDict, epochDict,
    Lmax=2, xpy=np, n_cal=n_cal, return_cal_components=True)
comp = np.array(comp)
assert comp.shape == (npts_extrinsic, n_cal), "cal_components shape %s" % (comp.shape,)
# each column must equal the per-block n_cal==1 evaluation (raw, no weight)
assert np.max(np.abs(comp.T - lnL_blocks)) < 1e-9, "cal_components != per-block reference"
lnL_from_comp = logsumexp(comp, axis=1) - np.log(n_cal)
assert np.max(np.abs(lnL_from_comp - np.array(lnL_new))) < 1e-9, \
    "collapse of cal_components != cal-marg lnL"
print("cal_components check: max err vs blocks = %.2e ; collapse matches lnL" %
      np.max(np.abs(comp.T - lnL_blocks)))

# --- also confirm n_cal==1 on the full concat == block-0 evaluation (regression) ---
lnL_n1_full = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
    tvals, P, lookupNKDict, {det: rho[:, :N_window].copy()}, ctUArrayDict, ctVArrayDict,
    epochDict, Lmax=2, xpy=np, n_cal=1)
assert np.allclose(np.array(lnL_n1_full), np.array(lnL_blocks[0])), "block-0 regression failed"

print("\nPASS: in-loop calibration marginalization matches brute-force reference.")
