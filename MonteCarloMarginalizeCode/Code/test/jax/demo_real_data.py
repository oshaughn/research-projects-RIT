"""
Demo / figure generator for the JAX ILE likelihood on REAL detector data.

Reuses validation data already on hand (e.g. the calmarg_GW240925 GWOSC frames +
PSDs, described by an ``event_params.json``) rather than re-inventing inputs.

Two tasks (``--task``):

  equality : run the JAX likelihood (interp='nearest', which reproduces the
             production discrete-shift path) AND the production numpy reference
             ``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`` on the SAME
             precompute and the SAME random extrinsic samples, and write the
             (lnL_jax - lnL_ref) vs lnL_ref scatter -- the lnL-equality target.

  snr      : the high-SNR study done the CHEAP way -- keep the frame data fixed
             and SCALE THE PSD (S -> S/k^2 multiplies the network SNR by k), so a
             single dataset yields an SNR sequence with no re-injection.  At each
             SNR run one flowMC extrinsic evaluation and record sky recovery,
             90% sky credible area, evidence/neff and wall time.

Inputs are taken from ``--data-dir`` (expects ``event_params.json``, per-IFO
``<IFO>-psd.xml.gz``) and ``--frame-dir`` (the ``.gwf`` files; default = data-dir's
parent).  All physics params default from ``event_params.json``.

Examples:
  python demo_real_data.py --task equality \
      --data-dir ~/RIFT_roboto_paper/analyses/calmarg_GW240925/data \
      --n-samples 4000 --l-max 2 --out-prefix /tmp/gw240925_eq
  python demo_real_data.py --task snr  --snr-targets 12,24,48,96 \
      --data-dir ... --out-prefix /tmp/gw240925_snr
"""

import os
import sys
import json
import glob
import time
import argparse
import types
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as FL
from RIFT.likelihood.jax_ile import build_data_from_precompute
from RIFT.likelihood.jax_ile.core import fused_log_likelihood
from RIFT.likelihood.jax_ile.wrapper import JAXDistanceMarginalizedLikelihood

MSUN, PC = lal.MSUN_SI, lal.PC_SI
_IFO_PREFIX = {"H1": "H", "L1": "L", "V1": "V"}


# ---------------------------------------------------------------------------
# Data loading (real frames + PSD), with optional PSD scaling for the SNR study
# ---------------------------------------------------------------------------
def _local_cache(frame_dir, ifos, channels):
    """Write a cache with LOCAL absolute paths to the .gwf files (the shipped
    event.cache often has stale/foreign paths)."""
    lines = []
    for ifo in ifos:
        pfx = _IFO_PREFIX[ifo]
        hits = sorted(glob.glob(os.path.join(frame_dir, "%s-%s*-*.gwf" % (pfx, ifo))))
        if not hits:
            raise FileNotFoundError("no .gwf for %s in %s" % (ifo, frame_dir))
        fn = os.path.abspath(hits[0])
        base = os.path.basename(fn)
        toks = base[:-4].split("-")
        gps, dur = toks[-2], toks[-1]
        lines.append("%s %s_%s %s %s file://localhost%s"
                     % (pfx, ifo, "FRAMES", gps, dur, fn))
    path = os.path.join("/tmp", "demo_real_%d.cache" % os.getpid())
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def load_real(params, data_dir, frame_dir, srate=4096.0, psd_scale=1.0,
              ifos=None):
    """Return (data_dict, psd_dict, detectors).  ``psd_scale`` multiplies every
    PSD (psd_scale<1 raises the SNR; SNR scales as 1/sqrt(psd_scale))."""
    ifos = ifos or params["ifos"]
    event = float(params["trigger_time"])
    seglen = float(params.get("seglen", 16.0))
    deltaT = 1.0 / srate
    cache = _local_cache(frame_dir, ifos, params["channels"])
    start, stop = event - (seglen - 2.0), event + 2.0
    data_dict, psd_dict = {}, {}
    for ifo in ifos:
        chan = params["channels"][ifo]
        d = lalsimutils.frame_data_to_non_herm_hoff(
            cache, ifo + ":" + chan, start=start, stop=stop,
            window_shape=0.0, deltaT=deltaT)
        data_dict[ifo] = d
        psdf = os.path.join(data_dir, "%s-psd.xml.gz" % ifo)
        psd = lalsimutils.get_psd_series_from_xmldoc(psdf, ifo)
        psd = lalsimutils.resample_psd_series(psd, d.deltaF)
        if psd_scale != 1.0:
            psd.data.data = psd.data.data * psd_scale
        psd_dict[ifo] = psd
    return data_dict, psd_dict, list(ifos)


def make_template(params, deltaT, deltaF, fiducial_epoch):
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = params["mass1"] * MSUN, params["mass2"] * MSUN
    P.s1z, P.s2z = params.get("spin1z", 0.0), params.get("spin2z", 0.0)
    P.fmin = params.get("fmin", 20.0)
    P.fref = params.get("fref", params.get("fmin", 20.0))
    P.deltaT, P.deltaF = deltaT, deltaF
    P.fmax = 0.0
    P.approx = lalsim.GetApproximantFromString(params.get("approximant", "IMRPhenomD"))
    P.radec = True
    P.tref = fiducial_epoch
    P.dist = 1000.0 * 1e6 * PC
    return P


def network_snr(data_dict, psd_dict, fmin, fmax):
    snr2 = 0.0
    for det, d in data_dict.items():
        fNyq = d.deltaF * d.data.length / 2.0
        IP = lalsimutils.ComplexIP(fLow=fmin, fNyq=fNyq, deltaF=d.deltaF,
                                   psd=psd_dict[det], fMax=fmax, analyticPSD_Q=False)
        snr2 += float(np.real(IP.norm(d))) ** 2
    return np.sqrt(snr2)


# ---------------------------------------------------------------------------
def _pack_reference(extras, detectors):
    ln, rh, cu, cv, ep = {}, {}, {}, {}, {}
    for det in detectors:
        a = FL.PackLikelihoodDataStructuresAsArrays(
            list(extras["rholms"][det].keys()), extras["rholms_intp"][det],
            extras["rholms"][det], extras["cross_terms"][det],
            extras["cross_terms_V"][det])
        ln[det], cu[det], cv[det], rh[det], ep[det] = a[0], a[3], a[4], a[5], a[7]
    return ln, rh, cu, cv, ep


def task_equality(params, data_dir, frame_dir, opts):
    fid = round(float(params["trigger_time"]))  # integer-ish fiducial epoch
    data_dict, psd_dict, dets = load_real(params, data_dir, frame_dir, opts.srate)
    deltaF = data_dict[dets[0]].deltaF
    P = make_template(params, 1.0 / opts.srate, deltaF, fid)
    fmax = params.get("fmax", 896.0)
    iwh = opts.integration_window_half
    swh = opts.storage_window_half
    data, extras = build_data_from_precompute(
        P.copy(), data_dict, psd_dict, fid, swh, iwh, opts.l_max, fmax,
        analyticPSD_Q=False, verbose=False)
    ln, rh, cu, cv, ep = _pack_reference(extras, dets)
    print("modes:", data.lms, " guessed SNR:", extras["guess_snr"])

    rng = np.random.default_rng(opts.seed)
    S = opts.n_samples
    Pv = types.SimpleNamespace()
    Pv.phi = rng.uniform(0, 2 * np.pi, S)
    Pv.theta = np.arcsin(rng.uniform(-1, 1, S))
    Pv.psi = rng.uniform(0, np.pi, S)
    Pv.incl = np.arccos(rng.uniform(-1, 1, S))
    Pv.phiref = rng.uniform(0, 2 * np.pi, S)
    distMpc = rng.uniform(100.0, 2000.0, S)
    Pv.dist = distMpc * PC * 1e6
    Pv.tref = float(fid)
    Pv.deltaT = 1.0 / opts.srate
    tvals = np.linspace(-iwh, iwh, int(2 * iwh / Pv.deltaT))

    lnL_ref = FL.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, ln, rh, cu, cv, ep, Lmax=opts.l_max, xpy=np)
    lnL_jax = np.asarray(fused_log_likelihood(
        data, Pv.phi, Pv.theta, Pv.psi, Pv.incl, Pv.phiref, distMpc,
        interp="nearest"))

    finite = np.isfinite(lnL_ref) & np.isfinite(lnL_jax)
    diff = lnL_jax[finite] - lnL_ref[finite]
    print("[lnL equality on real data] N=%d  max|jax-ref|=%.3e  (excluded %d ref-underflow)"
          % (finite.sum(), np.max(np.abs(diff)), (~finite).sum()))
    np.savetxt(opts.out_prefix + "_lnL.dat",
               np.column_stack([lnL_ref[finite], lnL_jax[finite]]),
               header="lnL_ref lnL_jax")
    _scatter(lnL_ref[finite], lnL_jax[finite], opts.out_prefix + "_lnLeq.png",
             params.get("event", "event"))


def _scatter(lnL_ref, lnL_jax, path, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (matplotlib unavailable: %r; skipping PNG)" % e); return
    d = lnL_jax - lnL_ref
    med = float(np.median(np.abs(d)))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(lnL_ref, d, s=6, alpha=0.5)
    ax.set_yscale("symlog", linthresh=1e-13)   # show the ~1e-14 bulk and the tail
    ax.set_xlabel(r"$\ln{\cal L}$ (reference, numpy ILE)")
    ax.set_ylabel(r"$\ln{\cal L}_{\rm JAX} - \ln{\cal L}_{\rm ref}$")
    ax.set_title("JAX vs reference ILE likelihood (%s)\n"
                 "median |diff|=%.1e, max=%.1e (far-tail low-lnL only)"
                 % (label, med, np.max(np.abs(d))))
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print("  wrote %s" % path)


def task_snr(params, data_dir, frame_dir, opts):
    from RIFT.likelihood.jax_ile import samplers
    targets = [float(x) for x in opts.snr_targets.split(",")]
    fid = round(float(params["trigger_time"]))
    fmax = params.get("fmax", 896.0)
    fmin = params.get("fmin", 20.0)
    iwh, swh = opts.integration_window_half, opts.storage_window_half

    # baseline signal-SNR estimate at psd_scale = 1 (from the precompute, which
    # reflects the PSD); scaling the PSD by s scales this SNR by 1/sqrt(s).
    dd0, pp0, dets = load_real(params, data_dir, frame_dir, opts.srate)
    deltaF0 = dd0[dets[0]].deltaF
    P0 = make_template(params, 1.0 / opts.srate, deltaF0, fid)
    _, ex0 = build_data_from_precompute(P0.copy(), dd0, pp0, fid, swh, iwh,
                                        opts.l_max, fmax, analyticPSD_Q=False, verbose=False)
    snr0 = float(ex0["guess_snr"])
    print("baseline signal SNR estimate (psd_scale=1): %.2f" % snr0)

    rows, flow_state, sky_by_snr = [], None, []
    for snr_t in targets:
        psd_scale = (snr0 / snr_t) ** 2          # S -> S*scale ; SNR -> SNR0/sqrt(scale)
        dd, pp, _ = load_real(params, data_dir, frame_dir, opts.srate, psd_scale=psd_scale)
        deltaF = dd[dets[0]].deltaF
        P = make_template(params, 1.0 / opts.srate, deltaF, fid)
        data, exj = build_data_from_precompute(
            P.copy(), dd, pp, fid, swh, iwh, opts.l_max, fmax,
            analyticPSD_Q=False, verbose=False)
        snr_real = float(exj["guess_snr"])
        like = JAXDistanceMarginalizedLikelihood(data, 1.0, 20000.0, n_grid=256)
        t0 = time.time()
        res = samplers.flowmc_sample(like, 1.0, 20000.0, n_prior_pilot=opts.n_prior_pilot,
                                     reuse_state=flow_state, seed=opts.seed)
        flow_state = res.get("flow_state")
        wall = time.time() - t0
        th, lnL = res["theta"], res["lnL"]
        # weighted 90% sky credible area (deg^2) via the local tangent-plane
        # Gaussian covariance -- finite even when the posterior is much narrower
        # than the empirical sample spacing (high SNR).  90% area of a 2D
        # Gaussian = -2 ln(0.1) * pi * sqrt(det Sigma).
        if len(th):
            w = np.exp(lnL - lnL.max()); w /= w.sum()
            ra_m = float((w * th[:, 0]).sum()); dec_m = float((w * th[:, 1]).sum())
            x = (th[:, 0] - ra_m) * np.cos(dec_m); y = th[:, 1] - dec_m
            Sig = np.cov(np.stack([x, y]), aweights=w)
            det = max(float(np.linalg.det(Sig)), 0.0)
            area = float(-2 * np.log(0.1) * np.pi * np.sqrt(det) * (180 / np.pi) ** 2)
            sky_by_snr.append((snr_real, th[:, 0].copy(), th[:, 1].copy(), w.copy()))
        else:
            area = np.nan; ra_m = dec_m = np.nan
        row = [snr_real, psd_scale, float(lnL.max()) if len(lnL) else np.nan,
               res["logZ"], res["neff"], area, ra_m, dec_m, len(th), wall]
        rows.append(row)
        print("SNR=%6.1f (scale %.4g)  maxlnL=%9.1f  logZ=%9.3g  neff=%6.1f  "
              "area90=%.3g deg^2  sky(RA,DEC)=(%.3f,%.3f)  N=%d  %.1fs"
              % (snr_real, psd_scale, row[2], row[3], row[4], area, ra_m, dec_m, len(th), wall))
    cols = "snr psd_scale maxlnL logZ neff area90_deg2 ra_mean dec_mean nsamp wall"
    np.savetxt(opts.out_prefix + "_snr.dat", np.array(rows), header=cols)
    print("wrote %s_snr.dat" % opts.out_prefix)
    _skymap(sky_by_snr, opts.out_prefix + "_skymap.png",
            params.get("event", "event"))


def _skymap(sky_by_snr, path, label):
    """Overlay the recovered sky samples at each SNR (zoomed to the region),
    showing the credible region shrinking with SNR."""
    if not sky_by_snr:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (matplotlib unavailable: %r; skipping skymap)" % e); return
    # zoom window from the lowest-SNR (broadest) sample set
    ra0, dec0 = sky_by_snr[0][1], sky_by_snr[0][2]
    cdec = np.cos(np.median(dec0))
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("viridis")
    for i, (snr, ra, dec, w) in enumerate(sky_by_snr):
        c = cmap(i / max(len(sky_by_snr) - 1, 1))
        ax.scatter(np.degrees(ra), np.degrees(dec), s=4, alpha=0.35, color=c,
                   label="SNR %.0f" % snr)
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
    ax.set_title("GW240925 sky recovery vs SNR (single ILE eval, PSD-scaled)\n"
                 "%s — credible region shrinks with SNR" % label)
    # zoom to +/- a few deg around the lowest-SNR spread
    rc, dc = np.degrees(np.median(ra0)), np.degrees(np.median(dec0))
    span = max(3.0, np.degrees(np.std(ra0) * cdec) * 4, np.degrees(np.std(dec0)) * 4)
    ax.set_xlim(rc - span, rc + span); ax.set_ylim(dc - span, dc + span)
    ax.legend(markerscale=3, fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print("  wrote %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["equality", "snr"])
    ap.add_argument("--data-dir", required=True,
                    help="dir with event_params.json + <IFO>-psd.xml.gz")
    ap.add_argument("--frame-dir", default=None, help="dir with .gwf (default: data-dir/..)")
    ap.add_argument("--ifos", default=None, help="comma list; default from params")
    ap.add_argument("--srate", type=float, default=4096.0)
    ap.add_argument("--l-max", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--integration-window-half", type=float, default=0.075)
    ap.add_argument("--storage-window-half", type=float, default=0.15)
    ap.add_argument("--snr-targets", default="12,24,48,96")
    ap.add_argument("--n-prior-pilot", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="/tmp/jax_ile_demo")
    opts = ap.parse_args()
    params = json.load(open(os.path.join(opts.data_dir, "event_params.json")))
    if opts.ifos:
        params["ifos"] = opts.ifos.split(",")
    frame_dir = opts.frame_dir or os.path.dirname(os.path.abspath(opts.data_dir.rstrip("/")))
    if opts.task == "equality":
        task_equality(params, opts.data_dir, frame_dir, opts)
    else:
        task_snr(params, opts.data_dir, frame_dir, opts)


if __name__ == "__main__":
    main()
