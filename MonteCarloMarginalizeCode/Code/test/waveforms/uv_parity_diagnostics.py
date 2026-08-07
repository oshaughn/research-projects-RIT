#!/usr/bin/env python
"""
uv_parity_diagnostics.py : prototype waveform-QA checks expressed purely at
the level of RIFT's U/V cross-term matrices (the objects ILE already builds
in PrecomputeLikelihoodTerms).  Goal: catch waveform parity violations at
precompute time, before they bias PE.

Definitions (RIFT conventions, ComplexIP over two-sided band, even PSD):
    U_{(lm),(l'm')} = < h_lm | h_l'm' >
    V_{(lm),(l'm')} = < conj(h_lm) | h_l'm' >

Diagnostics:
 D1 (exact identities; failure = code bug, ANY waveform):
     U = U^dagger ,   V = V^T
 D2 (single config; applies whenever the CONFIG is reflection-symmetric,
     i.e. nonprecessing -- even if the model is a precessing model):
     V_{(l,m),(l',m')} = (-1)^l U_{(l,-m),(l',m')}
     Failure = spurious (+m,-m) asymmetry (parity violation) in the model.
 D3 (reflected pair; applies to ANY config; two waveform generations):
     with primes = quantities of the reflected config (s_xy -> -s_xy):
     U'_{(lm),(l'm')} = (-1)^{l+l'} U_{(l',-m'),(l,-m)}
     V'_{(lm),(l'm')} = (-1)^{l+l'} conj( V_{(l,-m),(l',-m')} )
     Failure = parity violation, full precessing case.

Each check reports a relative Frobenius-norm residual.
"""
import numpy as np
import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl

LMAX = 4

def make_P(m1, m2, s1, s2, approx_str):
    P = lsu.ChooseWaveformParams()
    P.m1 = m1*lal.MSUN_SI; P.m2 = m2*lal.MSUN_SI
    P.s1x, P.s1y, P.s1z = s1
    P.s2x, P.s2y, P.s2z = s2
    P.fmin = 20.; P.fref = 20.
    P.deltaT = 1./4096
    P.deltaF = 1./16
    P.dist = 400.*1e6*lal.PC_SI
    P.phiref = 0.; P.incl = 0.; P.psi = 0.; P.tref = 0.
    P.approx = lalsim.GetApproximantFromString(approx_str)
    return P

def uv_matrices(P):
    hlmF = lsu.hlmoff(P.copy(), LMAX)
    hlmF_conj = lsu.conj_hlmoff(P.copy(), LMAX)
    if not isinstance(hlmF, dict):
        hlmF = lsu.SphHarmFrequencySeries_to_dict(hlmF, LMAX)
        hlmF_conj = lsu.SphHarmFrequencySeries_to_dict(hlmF_conj, LMAX)
    fNyq = 0.5/P.deltaT
    U = fl.ComputeModeCrossTermIP(hlmF, hlmF, lal.LIGOIPsd, P.fmin, fNyq, fNyq,
                                  P.deltaF, analyticPSD_Q=True, verbose=False)
    V = fl.ComputeModeCrossTermIP(hlmF_conj, hlmF, lal.LIGOIPsd, P.fmin, fNyq, fNyq,
                                  P.deltaF, analyticPSD_Q=True, verbose=False, prefix="V")
    return U, V

def frob(d, keys):
    return np.sqrt(sum(abs(d[k])**2 for k in keys))

def rel(dA, dB, keys):
    return np.sqrt(sum(abs(dA[k]-dB[k])**2 for k in keys))/max(frob(dA, keys), 1e-300)

def check_D1(U, V):
    keys = list(U.keys())
    Udag = {(p1, p2): np.conj(U[(p2, p1)]) for (p1, p2) in keys}
    Vt = {(p1, p2): V[(p2, p1)] for (p1, p2) in keys}
    return rel(U, Udag, keys), rel(V, Vt, keys)

def check_D2(U, V):
    keys = list(U.keys())
    Vpred = {}
    for (p1, p2) in keys:
        Vpred[(p1, p2)] = (-1)**p1[0]*U[((p1[0], -p1[1]), p2)]
    return rel(V, Vpred, keys)

def check_D3(U, V, Up, Vp):
    keys = list(U.keys())
    Upred, Vpred = {}, {}
    for (p1, p2) in keys:
        f1 = (p1[0], -p1[1]); f2 = (p2[0], -p2[1])
        s = (-1)**(p1[0]+p2[0])
        Upred[(p1, p2)] = s*U[(f2, f1)]
        Vpred[(p1, p2)] = s*np.conj(V[(f1, f2)])
    return rel(Up, Upred, keys), rel(Vp, Vpred, keys)

CONFIGS = {
    "nonprec":   dict(m1=44., m2=36., s1=(0., 0., 0.5), s2=(0., 0., -0.3)),
    "superkick_perturbed": dict(m1=40.8, m2=39.2,
                      s1=(0.8*np.cos(0.4), 0.8*np.sin(0.4), 0.),
                      s2=(-0.75*np.cos(0.45), -0.75*np.sin(0.45), 0.)),
    "generic_prec": dict(m1=48., m2=32., s1=(0.5, 0.2, 0.3), s2=(-0.1, 0.4, -0.2)),
}

import sys
MODELS = sys.argv[1:] if len(sys.argv) > 1 else ["IMRPhenomTPHM", "NRSur7dq4"]

for model in MODELS:
    for cname, c in CONFIGS.items():
        try:
            P = make_P(c["m1"], c["m2"], c["s1"], c["s2"], model)
            U, V = uv_matrices(P)
            e1u, e1v = check_D1(U, V)
            e2 = check_D2(U, V)
            Pr = P.copy()
            Pr.s1x, Pr.s1y = -P.s1x, -P.s1y
            Pr.s2x, Pr.s2y = -P.s2x, -P.s2y
            Up, Vp = uv_matrices(Pr)
            e3u, e3v = check_D3(U, V, Up, Vp)
            print(f"[{model:14s}] {cname:20s} D1(U)={e1u:.1e} D1(V)={e1v:.1e} | "
                  f"D2={e2:.3e} | D3(U)={e3u:.3e} D3(V)={e3v:.3e}", flush=True)
        except Exception as e:
            print(f"[{model:14s}] {cname:20s} ERROR: {e}", flush=True)
