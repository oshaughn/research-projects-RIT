"""
Generate figure DATA for the paper's 3G subsection (8.B):

  1. sky area vs network SNR  (Fig-10 analog) -- one row per SNR:
       snr, area_cov_deg2, area_hist_deg2, sky_ESS, map_dist_deg
  2. full posterior SAMPLES at a representative SNR (Fig-2/Fig-4 analog pair):
       ra, dec, psi, incl, phiref, distMpc (distance drawn from its per-sample
       conditional), plus the injected truth.

Finite-size (Path D) CE+ET injection via slowrot_fs_lib, sampled with the full
high-SNR reparameterization stack (network sky coords + phase rotation + dense
mass + gradient seed-polish).  Writes .npz files to SLOWROT_FIG_DIR.

Run on GPU in the JAX container (pin an idle GPU):
  apptainer exec --nv <sif> env PYTHON_JULIAPKG_OFFLINE=yes JAX_ENABLE_X64=1 \
    JAX_ILE_DISTMARG_GH=64 CUDA_VISIBLE_DEVICES=<idle> \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 SLOWROT_FIG_DIR=<out> \
    PYTHONPATH=<Code>:<paper>/analyses/slowrot_finite-size \
    python test/jax/make_3g_figdata.py
"""
import os
import sys
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

_FSLIB_DIR = os.environ.get(
    "SLOWROT_FS_LIB_DIR",
    os.path.expanduser("~/RIFT_roboto_paper/analyses/slowrot_finite-size"))
sys.path.insert(0, _FSLIB_DIR)
import slowrot_fs_lib as fslib

from RIFT.likelihood.jax_ile.wrapper import (
    build_freqresponse_data_from_precompute, JAXDistanceMarginalizedLikelihood)
from RIFT.likelihood.jax_ile import samplers
from RIFT.likelihood.jax_ile import core as _core

OUT = os.environ.get("SLOWROT_FIG_DIR", "/tmp/slowrot_3g_fig")
os.makedirs(OUT, exist_ok=True)
NETWORK = os.environ.get("SLOWROT_NET", "CE+ET")
QMAX = int(os.environ.get("SLOWROT_QMAX", "4"))
IWH = 0.03
TBUF = 0.12
# SNR ladder for the area-vs-SNR curve; SNR of the representative event.
SNRS = [float(x) for x in os.environ.get("SLOWROT_SNRS", "40,100,200,400,700,1000").split(",")]
SNR_REP = float(os.environ.get("SLOWROT_SNR_REP", "600"))


def _cov_sky_area_90(ra, dec):
    ra0 = np.angle(np.mean(np.exp(1j * ra))); dec0 = float(np.mean(dec))
    x = ((ra - ra0 + np.pi) % (2 * np.pi) - np.pi) * np.cos(dec0)
    y = dec - dec0
    cov = np.cov(np.vstack([x, y])); det = float(np.linalg.det(cov))
    if not np.isfinite(det) or det <= 0:
        return float("nan")
    return float(np.pi * 4.60517 * np.sqrt(det) * (180.0 / np.pi) ** 2)


def _gc_dist(ra, dec, ra0, dec0):
    c = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _draw_distance(data, ra, dec, psi, incl, phiref, d_min, d_max, seed=0):
    """Draw a luminosity distance per angular sample from its conditional.

    The per-(sample,time) distance integrand is exp(K x - 0.5 R x^2) with
    x = dref/d, K = Re(kappa_unit), R = rho_sq_unit (from _accumulate_unit).  At
    the matched-filter time bin (max K^2/R, K>0) we draw x from the normalized
    p(x) ∝ exp(K x - 0.5 R x^2) x^{-4} (the d^2 volumetric prior in x) on a grid,
    by inverse-CDF.  Good enough for the figure's distance posterior.
    """
    rng = np.random.default_rng(seed)
    K, R = _core._accumulate_unit(data, ra, dec, psi, incl, phiref, "cubic", False)
    K = np.asarray(K.real); R = np.maximum(np.asarray(R), 1e-30)   # (S, npts)
    snr2 = np.where(K > 0, K * K / R, -np.inf)
    tb = np.argmax(snr2, axis=1)                                   # best time bin
    S = ra.shape[0]
    Kb = K[np.arange(S), tb]; Rb = R[np.arange(S), tb]
    dref = float(data.distMpcRef)
    x_lo, x_hi = dref / d_max, dref / d_min
    xg = np.linspace(x_lo, x_hi, 512)
    d = np.empty(S)
    for i in range(S):
        lg = Kb[i] * xg - 0.5 * Rb[i] * xg ** 2 - 4.0 * np.log(xg)
        lg -= lg.max()
        w = np.exp(lg); c = np.cumsum(w); c /= c[-1]
        u = rng.random()
        xi = np.interp(u, c, xg)
        d[i] = dref / xi
    return d


def run_one(src, net, target_snr, want_samples=False):
    dist = fslib.distance_for_snr(src, net, target_snr)
    # SELFCONSISTENT (int Qmax): render the injection with the recovery's own b_p*W_p
    # response so truth is the exact global maximum -- combined with an adequately
    # OVERSAMPLED rholm this removes the cubic-interpolation timing systematic that
    # otherwise displaces the razor-sharp high-SNR sky posterior.  The oversampling is
    # slowrot_fs_lib's own knob (deltaT = 1/(2*oversample*fmax), default oversample=4);
    # raising fmax does not do it.  Measured ladders (offset vs oversample, and the
    # stencil sweep): analyses/slowrot_finite-size/DESIGN_sampling.md in the paper repo.
    sc = os.environ.get("SLOWROT_SELFCONSISTENT")
    data_dict, psd_dict, arm_dict, meta = fslib.build_finite_size_data(
        src, net, dist, selfconsistent_Qmax=(int(sc) if sc else None))
    P0 = fslib._base_params(src, dist, meta["deltaT"], meta["deltaF"])
    data, _ = build_freqresponse_data_from_precompute(
        P0, data_dict, psd_dict, fslib.EVENT_TIME, IWH, fslib.LMAX, src.fmax,
        t_window=TBUF, Qmax=QMAX, L_arm=arm_dict, analyticPSD_Q=True, verbose=False)
    d_min = max(1.0, dist * 0.3); d_max = dist * 2.5
    like = JAXDistanceMarginalizedLikelihood(data, d_min, d_max, n_grid=256, interp="cubic")
    n_pilot = int(max(2e4, 50.0 * target_snr))
    # At very high SNR + fine deltaT the true peak is thinner than the pilot can
    # resolve, so seed one chain at the injected truth (production: intrinsic grid +
    # coarse extrinsic pass provides this).  Gated on SLOWROT_SEED_TRUTH so the
    # area-vs-SNR sweep, where the pilot already finds the (broader) modes, is untouched.
    extra = None
    if os.environ.get("SLOWROT_SEED_TRUTH"):
        extra = np.array([[src.ra, src.dec, src.psi, src.incl, src.phiref]])
    res = samplers.multistart_nuts(
        like, d_min, d_max, n_starts=6, num_warmup=300, num_samples=500,
        n_prior_pilot=n_pilot, seed=1, sky_coords="network", rotate_phase=True,
        dense_mass=True, max_tree_depth=(7, 10), polish_seeds=True,
        extra_seeds=extra, verbose=True)
    th = np.asarray(res["theta"]); lnL_all = np.asarray(res["lnL"])
    ra, dec, psi, incl, phiref = (th[:, 0], th[:, 1], th[:, 2], th[:, 3], th[:, 4])
    # Dominant-mode mask: the credible region is the region carrying the posterior
    # mass; sub-dominant multi-start modes many nats below the peak carry none.
    # Keep draws with lnL within DTHR of the peak -- broad at low SNR (one wide
    # mode), compact at high SNR (secondary modes dropped).  (Raw pooled draws are
    # saved; the mask is applied for the area and, in plot_3g, the recovery figure.)
    DTHR = 40.0
    dom = lnL_all > (lnL_all.max() - DTHR)
    tpc = res.get("theta_per_chain")
    from numpyro.diagnostics import effective_sample_size
    sky_ess = 0.0
    for c in np.asarray(tpc):
        sky_ess += float(effective_sample_size(np.sin(c[:, 1])[None, :]))
    # areas over the DOMINANT mode (dom mask)
    area_cov = _cov_sky_area_90(ra[dom], dec[dom])
    area_hist = fslib.sky_area_90(ra[dom], dec[dom], np.ones(dom.sum()), nside_bins=64)
    try:
        area_kde = float(fslib.sky_area_90_kde(ra[dom], dec[dom], np.ones(dom.sum())))
    except Exception:
        area_kde = float("nan")
    imap = int(np.argmax(lnL_all))
    map_d = float(_gc_dist(np.array([ra[imap]]), np.array([dec[imap]]), src.ra, src.dec)[0])
    row = dict(snr=meta["snr"], area_cov=area_cov, area_hist=area_hist,
               area_kde=area_kde, sky_ess=sky_ess, map_dist=map_d, dist_true=dist,
               frac_dom=float(dom.mean()))
    print("  SNR %.0f: sky_ESS=%.0f area_kde=%.3e area_cov=%.3e MAP=%.2f deg dom=%.0f%%" %
          (meta["snr"], sky_ess, area_kde, area_cov, map_d, 100 * dom.mean()))
    # save the sky samples at EVERY SNR (with lnL) so areas can be recomputed/plotted
    np.savez(os.path.join(OUT, "sky_snr%d.npz" % int(round(target_snr))),
             ra=ra, dec=dec, lnL=lnL_all, snr=meta["snr"],
             truth=np.array([src.ra, src.dec]))
    if want_samples:
        dist_s = _draw_distance(data, ra, dec, psi, incl, phiref, d_min, d_max)
        np.savez(os.path.join(OUT, "samples_snr%d.npz" % int(round(target_snr))),
                 ra=ra, dec=dec, psi=psi, incl=incl, phiref=phiref, distMpc=dist_s,
                 lnL=lnL_all,
                 truth=np.array([src.ra, src.dec, src.psi, src.incl, src.phiref, dist]),
                 snr=meta["snr"])
        print("  saved samples_snr%d.npz (%d draws)" % (int(round(target_snr)), len(ra)))
    return row


def main():
    # Representative-event inclination: the area-vs-SNR sweep is orientation-
    # independent (sky localization), but the recovery corner needs an INCLINED
    # source (default 60 deg) so the distance-inclination-polarization degeneracy
    # of a near-face-on dominant-quadrupole source is broken and the orientation
    # sector recovers on truth.  Override with SLOWROT_INCL.
    incl = float(os.environ.get("SLOWROT_INCL", "0.4"))
    # fmax is the ANALYSIS BAND LIMIT only: since paper-repo commit 2445905 the rholm
    # sampling is set independently by slowrot_fs_lib's oversample (deltaT =
    # 1/(2*oversample*fmax), default 4 -> srate 8192 here), so raising fmax no longer
    # refines the time series.  1024 stays as the band choice for this BNS, and at equal
    # sample rate the narrower band is marginally the better one.  SLOWROT_OVERSAMPLE
    # overrides the sampling; 1 reproduces pre-2026-08-26 archived runs bit-for-bit, and
    # is only passed when set so the script still runs against the older library.
    fmax = float(os.environ.get("SLOWROT_FMAX", "1024.0"))
    _ovs = os.environ.get("SLOWROT_OVERSAMPLE")
    src_kw = {"oversample": int(_ovs)} if _ovs else {}
    src = fslib.Source(m1=1.6, m2=1.4, ra=1.2, dec=0.3, psi=0.5, incl=incl,
                       phiref=0.0, fmin=50.0, fmax=fmax, seglen=32.0,
                       approx="IMRPhenomD", **src_kw)
    net = fslib.network(NETWORK)
    print("3G FIGDATA network=%s rep_snr=%.0f snrs=%s" % (NETWORK, SNR_REP, SNRS))
    rows = []
    snr_set = sorted(set(SNRS) | {SNR_REP})
    for snr in snr_set:
        try:
            rows.append(run_one(src, net, snr, want_samples=(snr == SNR_REP)))
        except Exception as e:
            import traceback; traceback.print_exc(); print("  SNR %.0f FAILED: %s" % (snr, e))
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez(os.path.join(OUT, "area_vs_snr.npz"), **arr)
    print("\nsaved area_vs_snr.npz (%d SNRs) to %s" % (len(rows), OUT))
    print("3G FIGDATA DONE")


if __name__ == "__main__":
    main()
