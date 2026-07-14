#!/usr/bin/env python
"""
reconstruct_strain.py -- whitened GW strain reconstruction with a confidence band.

Given RIFT ILE extrinsic *fair-draw* samples (one or more compact .npz files
from extract_ile_samples.py), reconstruct the whitened detector strain with a
90% credible band and overlay it on the whitened data.

Two waveform back-ends:
  --approx IMRPhenomD           : a waveform MODEL, built from the hlmoft modes
                                  (RIFT.lalsimutils.hlmoft -> hoft_from_hlm) using each
                                  sample's own m1,m2,spins(incl. in-plane),extrinsic.
                                  This is the same mode construction ILE's likelihood uses,
                                  so the time/phase reference matches the reported
                                  geocent_end_time (avoids an fref-dependent ~1 ms bias
                                  that the direct lalsimutils.hoft path has).
  --group G --nr-param F.h5      : a fixed NR simulation, via real_hoft() from a
                                  WaveformModeCatalog instantiated with the SAME options
                                  ILE's likelihood uses (factored_likelihood.py:286:
                                  align_at_peak_l2_m2_emission, shift_by_extraction_radius,
                                  clean_*, perturbative_extraction=False; NOT
                                  reference_phase_at_peak) so the time reference matches.
                                  (real_hoft, not hlmoft->hoft_from_hlm: the generic mode
                                  sum scrambles the NR phase; residual real_hoft-vs-hlmoff
                                  time offset ~0.5 ms, below plot resolution.)

The essential point: the ILE run MUST have been produced with
    --fairdraw-extrinsic-output --resample-time-marginalization
so every sample carries its OWN coalescence time (the 'time' column varies),
coherent with its coa_phase. Each realization is generated at that (time, phase)
-> the band is phase-coherent and tight, with NO post-hoc alignment.

Whitening is done in the frequency domain against the SAME PSD ILE used
(gwpy .whiten() is avoided: the analysis PSD often only reaches srate/2).
"""
import argparse
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
try:
    from scipy.signal.windows import tukey
except ImportError:
    from scipy.signal import tukey


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", action="append", required=True,
                    help="compact .npz from extract_ile_samples.py (repeat to POOL runs)")
    ap.add_argument("--fair-draw", action="store_true",
                    help="treat input rows as equal-weight fair posterior draws "
                         "(use with ILE --fairdraw-extrinsic-output output)")
    ap.add_argument("--approx", default=None, help="waveform model, e.g. IMRPhenomD (model mode)")
    ap.add_argument("--group", default=None, help="NR group, e.g. RIT-Five (NR mode)")
    ap.add_argument("--nr-param", default=None, help="NR strain file (NR mode)")
    ap.add_argument("--fref", type=float, default=20.0, help="reference frequency (model mode)")
    ap.add_argument("--lmax", type=int, default=4)
    ap.add_argument("--psd-file", action="append", required=True,
                    help="IFO=path.xml.gz (repeat per detector; must match ILE PSDs)")
    ap.add_argument("--event-time", type=float, required=True)
    ap.add_argument("--event-name", default="event")
    ap.add_argument("--sim-id", default="model")
    ap.add_argument("--data-txt", action="append", default=[],
                    help="IFO=path : 2-col (t, strain) data; else fetch from GWOSC")
    ap.add_argument("--intrinsic", default=None,
                    help="NR mode only: intrinsic_params .dat to weight along the mass curve")
    ap.add_argument("--ngen", type=int, default=1500, help="importance draws (ignored with --fair-draw)")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--srate", type=float, default=4096.0, help="sample rate (match ILE --srate)")
    ap.add_argument("--flow", type=float, default=20.0)
    ap.add_argument("--fhigh", type=float, default=None, help="whitening high cut (default srate/2)")
    ap.add_argument("--tlo", type=float, default=-0.10)
    ap.add_argument("--thi", type=float, default=0.06)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-out", default=None)
    ap.add_argument("--align", action="store_true",
                    help="DIAGNOSTIC ONLY. You should NOT need this. A visible offset "
                         "means the samples lack per-sample times (the ILE run was missing "
                         "--resample-time-marginalization).")
    o = ap.parse_args(argv)
    if not o.approx and not (o.group and o.nr_param):
        ap.error("choose a back-end: --approx MODEL, or --group G --nr-param F.h5")
    return o


def whiten_fd(x, psd, fs, flo, fhi):
    x = np.asarray(x, float)
    n = len(x)
    x = x * tukey(n, alpha=min(0.2, 8.0 / (n / fs)))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    pf = psd.f0 + psd.deltaF * np.arange(psd.data.length)
    S = np.interp(f, pf, psd.data.data, left=psd.data.data[0], right=psd.data.data[-1])
    Xf = np.fft.rfft(x) / np.sqrt(S)
    Xf[(f < flo) | (f > fhi)] = 0.0
    return np.fft.irfft(Xf, n=n)


_W = {}


def _init(mode, group, nr_param, approx, fref, psd_files, fs, flo, fhi, lmax, tev):
    _W.update(mode=mode, fs=fs, flo=flo, fhi=fhi, tev=tev, fref=fref, lmax=lmax,
              psd={ifo: lalsimutils.get_psd_series_from_xmldoc(f, ifo) for ifo, f in psd_files.items()})
    if mode == "nr":
        import NRWaveformCatalogManager3 as nrwf
        wfP = nrwf.WaveformModeCatalog(group, nr_param,
                                       clean_initial_transient=True, clean_final_decay=True,
                                       shift_by_extraction_radius=True, align_at_peak_l2_m2_emission=True,
                                       perturbative_extraction=False, lmax=lmax, use_provided_strain=True)
        wfP.P.taper = lalsimutils.lsu_TAPER_START
        wfP.P.deltaF = 1.0 / 16.0
        wfP.P.deltaT = 1.0 / fs
        wfP.P.radec = True
        _W.update(nrwf=nrwf, wfP=wfP)
    else:
        _W["approx"] = lalsim.GetApproximantFromString(approx)


def _project_whiten(h, tev, fs, flo, fhi, psd):
    t = lalsimutils.evaluate_tvals(h) - tev
    return (t.astype(np.float32), whiten_fd(h.data.data, psd, fs, flo, fhi).astype(np.float32))


def _gen(a):
    (m1, m2, mtot, a1z, a2z, a1x, a1y, a2x, a2y, ecc, incl, dist, ra, dec, psi, phiorb, tgeo) = a
    fs, tev, flo, fhi = _W["fs"], _W["tev"], _W["flo"], _W["fhi"]
    out = {}
    if _W["mode"] == "nr":
        wfP, nrwf = _W["wfP"], _W["nrwf"]
        wfP.P.m1, wfP.P.m2 = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
        wfP.P.s1z, wfP.P.s2z = a1z, a2z
        wfP.P.s1x = wfP.P.s1y = wfP.P.s2x = wfP.P.s2y = 0.0
        wfP.P.eccentricity = ecc
        wfP.P.fmin = 2 * wfP.fOrbitLower / (mtot * nrwf.MsunInSec)
        wfP.P.incl, wfP.P.dist = incl, dist * lal.PC_SI * 1e6
        wfP.P.phi, wfP.P.theta, wfP.P.psi = ra, dec, psi
        wfP.P.phiref, wfP.P.tref = phiorb, tgeo
        for ifo, psd in _W["psd"].items():
            wfP.P.detector = ifo
            out[ifo] = _project_whiten(wfP.real_hoft(), tev, fs, flo, fhi, psd)
    else:
        P = lalsimutils.ChooseWaveformParams()
        P.approx = _W["approx"]
        P.m1, P.m2 = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
        P.s1z, P.s2z = a1z, a2z
        P.s1x, P.s1y, P.s2x, P.s2y = a1x, a1y, a2x, a2y   # in-plane spins (precession)
        P.fmin, P.fref = flo, _W["fref"]
        P.deltaT = 1.0 / fs
        P.deltaF = 1.0 / 16.0
        P.radec = True
        P.incl, P.dist = incl, dist * lal.PC_SI * 1e6
        P.phi, P.theta, P.psi = ra, dec, psi
        P.phiref, P.tref = phiorb, tgeo
        # Build h(t) from the hlmoft modes (fd_standoff 0.9, as ILE's internal_hlm_generator uses)
        # and combine with hoft_from_hlm.  This is the SAME mode construction ILE's likelihood
        # uses, so the coalescence-time (and phase) reference matches the geocent_end_time ILE
        # reports.  Reconstructing with lalsimutils.hoft() instead leaves an fref-dependent ~1 ms
        # time bias (its coalescence reference differs from the hlmoff likelihood path).
        hlms = lalsimutils.hlmoft(P, Lmax=_W["lmax"], fd_standoff_factor=0.9)
        for ifo, psd in _W["psd"].items():
            P.detector = ifo
            out[ifo] = _project_whiten(lalsimutils.hoft_from_hlm(hlms, P), tev, fs, flo, fhi, psd)
    return out


def main(argv=None):
    o = get_args(argv)
    mode = "nr" if o.group else "model"
    fs = o.srate
    fhi = o.fhigh if o.fhigh else fs / 2.0
    PSD = {c.split("=")[0]: c.split("=")[1] for c in o.psd_file}
    IFOS = list(PSD.keys())
    psd_series = {ifo: lalsimutils.get_psd_series_from_xmldoc(f, ifo) for ifo, f in PSD.items()}

    keys = ["m1", "m2", "a1z", "a2z", "a1x", "a1y", "a2x", "a2y",
            "ra", "dec", "time", "phiorb", "incl", "psi",
            "distance", "lnL", "p", "ps", "eccentricity"]
    parts = {k: [] for k in keys}
    for f in o.samples:
        Z = np.load(f)
        n = len(Z["m1"])
        for k in keys:                      # tolerate older npz lacking in-plane spins etc.
            parts[k].append(np.asarray(Z[k], float) if k in Z.files else np.zeros(n))
    D = {k: np.concatenate(parts[k]) for k in keys}
    lnL, p, ps = D["lnL"], D["p"], D["ps"]
    fin = np.isfinite(lnL) & np.isfinite(p) & np.isfinite(ps) & (ps > 0)

    rng = np.random.default_rng(0)
    if o.fair_draw:
        sel = np.where(fin)[0]
        wsel = np.full(len(sel), 1.0 / len(sel))
        print("[select] fair-draw: %d equal-weight samples from %d file(s)"
              % (len(sel), len(o.samples)), flush=True)
    else:
        w = np.zeros_like(lnL)
        w[fin] = np.exp(lnL[fin] - np.nanmax(lnL[fin])) * p[fin] / ps[fin]
        w[~np.isfinite(w)] = 0.0
        w /= w.sum()
        pos = (rng.random() + np.arange(o.ngen)) / o.ngen
        sel, cnt = np.unique(np.searchsorted(np.cumsum(w), pos), return_counts=True)
        wsel = cnt.astype(float) / cnt.sum()
        print("[select] importance-resample: %d draws -> %d unique (weight-ESS=%.1f)"
              % (o.ngen, len(sel), 1.0 / np.sum(w[fin] ** 2)), flush=True)

    if mode == "nr" and o.intrinsic and os.path.exists(o.intrinsic):
        I = np.loadtxt(o.intrinsic)
        if I.ndim == 2 and I.shape[0] > 1:
            mt, lm = I[:, 1] + I[:, 2], I[:, 10]
            g = np.isfinite(lm)
            wM = np.exp(lm[g] - lm[g].max()); wM /= wM.sum()
            mtot = rng.choice(mt[g], size=len(sel), p=wM)
        else:
            mtot = (D["m1"] + D["m2"])[sel]
    else:
        mtot = (D["m1"] + D["m2"])[sel]

    args = [(float(D["m1"][j]), float(D["m2"][j]), float(mtot[i]),
             float(D["a1z"][j]), float(D["a2z"][j]),
             float(D["a1x"][j]), float(D["a1y"][j]), float(D["a2x"][j]), float(D["a2y"][j]),
             float(D["eccentricity"][j]),
             float(D["incl"][j]), float(D["distance"][j]), float(D["ra"][j]), float(D["dec"][j]),
             float(D["psi"][j]), float(D["phiorb"][j]), float(D["time"][j]))
            for i, j in enumerate(sel)]
    store = {("%s_%s" % (ifo, k)): [None] * len(sel) for ifo in IFOS for k in ("t", "h")}
    with ProcessPoolExecutor(max_workers=o.nproc, initializer=_init,
                             initargs=(mode, o.group, o.nr_param, o.approx, o.fref, PSD, fs,
                                       o.flow, fhi, o.lmax, o.event_time)) as ex:
        for i, res in enumerate(ex.map(_gen, args, chunksize=4)):
            for ifo in IFOS:
                store["%s_t" % ifo][i], store["%s_h" % ifo][i] = res[ifo]
            if (i + 1) % 100 == 0:
                print("  ... %d/%d" % (i + 1, len(sel)), flush=True)
    if o.cache_out:
        np.savez(o.cache_out, wsel=wsel, ifos=np.array(IFOS),
                 **{k: np.array(v, dtype=object) for k, v in store.items()})

    from gwpy.timeseries import TimeSeries
    data_txt = {c.split("=")[0]: c.split("=")[1] for c in o.data_txt}
    dw, dstd = {}, {}
    for ifo in IFOS:
        if ifo in data_txt:
            arr = np.loadtxt(data_txt[ifo]); td, ww = arr[:, 0] - o.event_time, arr[:, 1]
        else:
            from pesummary.gw.file.strain import StrainData
            Dt = StrainData.fetch_open_data(ifo, o.event_time - 16, o.event_time + 8)
            Dt = TimeSeries(np.array(Dt.value), t0=float(Dt.t0.value), dt=float(Dt.dt.value)).resample(fs)
            td = np.array(Dt.times.value) - o.event_time
            ww = whiten_fd(Dt.value, psd_series[ifo], fs, o.flow, fhi)
        q = (td > -4) & (td < -1)
        dstd[ifo] = np.std(ww[q]) if q.any() else 1.0
        dw[ifo] = (td, ww / dstd[ifo])

    from pesummary.core.plots.interpolate import Bounded_interp1d
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def wquant(v, ww, qs):
        ok = np.isfinite(v); v, ww = v[ok], ww[ok]
        order = np.argsort(v); cw = np.cumsum(ww[order]); cw /= cw[-1]
        return np.interp(qs, cw, v[order])

    def band(ifo):
        tt = [np.asarray(t, float) for t in store["%s_t" % ifo]]
        hh = [np.asarray(h, float) / dstd[ifo] for h in store["%s_h" % ifo]]
        lo = max(t[0] for t in tt); hi = min(t[-1] for t in tt)
        g = np.arange(max(lo, o.tlo - 0.05), min(hi, o.thi + 0.05), 1.0 / fs)
        A = np.array([Bounded_interp1d(t, h, xlow=t[0], xhigh=t[-1])(g) for t, h in zip(tt, hh)])
        up = np.array([wquant(A[:, k], wsel, 0.95) for k in range(A.shape[1])])
        dn = np.array([wquant(A[:, k], wsel, 0.05) for k in range(A.shape[1])])
        md = np.array([wquant(A[:, k], wsel, 0.50) for k in range(A.shape[1])])
        return g, up, dn, md

    fig, axes = plt.subplots(len(IFOS), 1, figsize=(15, 4 * len(IFOS)), sharex=True)
    if len(IFOS) == 1:
        axes = [axes]
    labels = {"H1": "LIGO Hanford", "L1": "LIGO Livingston", "V1": "Virgo"}
    for ax, ifo in zip(axes, IFOS):
        td, hd = dw[ifo]
        ax.plot(td, hd, color="0.5", alpha=0.7, lw=1.3, label="Whitened %s data" % ifo)
        g, up, dn, md = band(ifo)
        if o.align:
            m = (td >= -0.06) & (td <= 0.04); tdp = td[m][np.argmax(np.abs(hd[m]))]
            gm = (g >= -0.06) & (g <= 0.04); tmp = g[gm][np.argmax(np.abs(md[gm]))]
            g = g + (tdp - tmp)
        ax.fill_between(g, up, dn, color="tab:green", alpha=0.35, label="%s 90%% band" % o.sim_id)
        ax.plot(g, md, color="tab:green", lw=1.6, alpha=0.9)
        ax.set_xlim(o.tlo, o.thi); ax.set_ylabel("whitened strain"); ax.legend(loc="lower left")
        ax.text(0.5, 0.92, labels.get(ifo, ifo), ha="center", transform=ax.transAxes)
    neff = 1.0 / np.sum(wsel ** 2)
    fig.suptitle("%s : whitened strain and %s 90 pct band  (N=%d, eff=%d)"
                 % (o.event_name, o.sim_id, len(wsel), neff), fontsize=14, y=0.998)
    axes[-1].set_xlabel("time (s) from %.6f" % o.event_time)
    plt.tight_layout()
    plt.savefig(o.out, bbox_inches="tight", dpi=110)
    print("wrote", o.out, flush=True)


if __name__ == "__main__":
    main()
