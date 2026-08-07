#!/usr/bin/env python
"""
parity_check_hlm.py : waveform-level parity (orbital-plane reflection) check.

Exact GR requirement: reflecting the binary through the orbital plane at fref
maps (s_ix, s_iy, s_iz) -> (-s_ix, -s_iy, s_iz) for both spins (spins are
pseudovectors; L is preserved; positions/orbital phase unchanged), and the
spin-weighted spherical harmonic modes must satisfy

    h_lm[reflected params](t) = (-1)^l  conj( h_{l,-m}[original params](t) )

with NO time or phase shift freedom (same fmin/fref/phiref).  Nonprecessing
configurations are fixed points of the reflection, recovering the usual
equatorial identity h_{l,-m} = (-1)^l conj(h_lm).

We test this mode-by-mode for superkick-like and generic precessing configs.
Metrics per (l,m):
  amp_resid : relative L2 difference of |h^B_lm| vs |h^A_{l,-m}|   (phase-convention robust)
  cplx_resid: relative L2 difference of h^B_lm vs (-1)^l conj(h^A_{l,-m})
  asym      : physical (+m,-m) asymmetry content of config A itself,
              ||h^A_lm - (-1)^l conj(h^A_{l,-m})|| / ||h^A_lm||  (scale reference)
"""
import os, sys, json
import numpy as np
import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils

DELTA_T = 1./4096
LMAX = 4

def make_P(m1_msun, m2_msun, s1, s2, fmin=20., fref=20., approx_str=None):
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = m1_msun*lal.MSUN_SI
    P.m2 = m2_msun*lal.MSUN_SI
    P.s1x, P.s1y, P.s1z = s1
    P.s2x, P.s2y, P.s2z = s2
    P.fmin = fmin; P.fref = fref
    P.deltaT = DELTA_T
    P.dist = 400.*1e6*lal.PC_SI
    P.phiref = 0.0; P.incl = 0.0; P.psi = 0.0
    P.tref = 0.0
    if approx_str is not None:
        P.approx = lalsim.GetApproximantFromString(approx_str)
    return P

def reflected(P):
    Pr = P.copy()
    Pr.s1x, Pr.s1y = -P.s1x, -P.s1y
    Pr.s2x, Pr.s2y = -P.s2x, -P.s2y
    return Pr

def get_modes(approx_str, P, Lmax=LMAX):
    """Return dict {(l,m): (epoch_float, complex ndarray)}"""
    P = P.copy()
    if approx_str.startswith("SEOBNRv5"):
        import RIFT.physics.GWSignal as rgws
        hlmT = rgws.hlmoft(P, Lmax=Lmax, approx_string=approx_str)
    else:
        P.approx = lalsim.GetApproximantFromString(approx_str)
        hlmT = lalsimutils.hlmoft(P, Lmax=Lmax)
    if not isinstance(hlmT, dict):
        hlmT = lalsimutils.SphHarmTimeSeries_to_dict(hlmT, Lmax)
    out = {}
    for k, v in hlmT.items():
        out[k] = (float(v.epoch), np.array(v.data.data, dtype=complex))
    return out

def l2(x):
    return np.sqrt(np.sum(np.abs(x)**2))

def aligned_pair(eA, hA, eB, hB, deltaT=DELTA_T):
    """Trim two series (epochs eA,eB) to their common time support, nearest-sample."""
    off = (eB - eA)/deltaT
    n = int(round(off))
    if abs(off - n) > 1e-3:
        # non-integer offset: interpolate B onto A's grid
        tA = eA + deltaT*np.arange(len(hA))
        tB = eB + deltaT*np.arange(len(hB))
        re = np.interp(tA, tB, hB.real, left=0, right=0)
        im = np.interp(tA, tB, hB.imag, left=0, right=0)
        return hA, re + 1j*im, ("interp", off)
    # integer offset: shift
    if n >= 0:
        a = hA[n:]; b = hB
    else:
        a = hA; b = hB[-n:]
    m = min(len(a), len(b))
    return a[:m], b[:m], ("shift", n)

def compare(modesA, modesB, label=""):
    rows = []
    for (l, m) in sorted(modesA.keys()):
        if (l, -m) not in modesA or (l, m) not in modesB:
            continue
        eB, hB = modesB[(l, m)]
        eA, hA = modesA[(l, -m)]
        target = (-1)**l * np.conj(hA)   # prediction for h^B_lm from config A
        hB_al, tgt_al, how = aligned_pair(eB, hB, eA, target)
        nrm = max(l2(hB_al), l2(tgt_al))
        if nrm == 0:
            continue
        cplx_resid = l2(hB_al - tgt_al)/nrm
        amp_resid = l2(np.abs(hB_al) - np.abs(tgt_al))/nrm
        # physical asymmetry content of config A
        eA2, hA2 = modesA[(l, m)]
        a1, a2, _ = aligned_pair(eA2, hA2, eA, target)
        nrma = max(l2(a1), l2(a2))
        asym = l2(a1 - a2)/nrma if nrma > 0 else 0.
        rows.append(dict(l=l, m=m, cplx_resid=float(cplx_resid),
                         amp_resid=float(amp_resid), asym=float(asym)))
    return rows

CONFIGS = {
    # superkick: q=1, antiparallel in-plane spins, generic azimuth
    "superkick": dict(m1=40., m2=40.,
                      s1=(0.8*np.cos(0.4), 0.8*np.sin(0.4), 0.),
                      s2=(-0.8*np.cos(0.4), -0.8*np.sin(0.4), 0.)),
    # hangup-kick-like: in-plane antiparallel + aligned component
    "superkick_tilted": dict(m1=40., m2=40.,
                      s1=(0.6*np.cos(0.4), 0.6*np.sin(0.4), 0.5),
                      s2=(-0.6*np.cos(0.4), -0.6*np.sin(0.4), 0.5)),
    # superkick broken slightly: unequal masses + azimuth offset -> not a fixed point
    # of exchange symmetry, and total in-plane spin no longer exactly zero
    "superkick_perturbed": dict(m1=40.8, m2=39.2,
                      s1=(0.8*np.cos(0.4), 0.8*np.sin(0.4), 0.),
                      s2=(-0.75*np.cos(0.45), -0.75*np.sin(0.45), 0.)),
    # generic precessing, unequal mass
    "generic_prec": dict(m1=48., m2=32., s1=(0.5, 0.2, 0.3), s2=(-0.1, 0.4, -0.2)),
    # nonprecessing control: reflection is the identity
    "nonprec_control": dict(m1=44., m2=36., s1=(0., 0., 0.5), s2=(0., 0., -0.3)),
}

MODELS = sys.argv[1:] if len(sys.argv) > 1 else \
    ["IMRPhenomTPHM", "SEOBNRv4PHM", "NRSur7dq4", "SEOBNRv5PHM"]

only = os.environ.get("PARITY_CONFIGS")
if only:
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in only.split(",")}

results = {}
for model in MODELS:
    results[model] = {}
    for cname, c in CONFIGS.items():
        try:
            PA = make_P(c["m1"], c["m2"], c["s1"], c["s2"])
            PB = reflected(PA)
            mA = get_modes(model, PA)
            mB = get_modes(model, PB)
            rows = compare(mA, mB)
            results[model][cname] = rows
            worst = max(rows, key=lambda r: r["cplx_resid"])
            print(f"[{model:14s}] {cname:18s} worst mode ({worst['l']},{worst['m']:+d}): "
                  f"cplx={worst['cplx_resid']:.3e} amp={worst['amp_resid']:.3e} "
                  f"(physical asym scale {worst['asym']:.3e})", flush=True)
            for r in rows:
                print(f"     ({r['l']},{r['m']:+d})  cplx={r['cplx_resid']:.3e}  "
                      f"amp={r['amp_resid']:.3e}  asym={r['asym']:.3e}", flush=True)
        except Exception as e:
            results[model][cname] = f"ERROR: {e}"
            print(f"[{model:14s}] {cname:18s} ERROR: {e}", flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_check_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print("wrote", out)
