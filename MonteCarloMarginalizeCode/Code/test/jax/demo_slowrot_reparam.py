"""
High-SNR reparameterized sampling: phase-marginalized + Fisher-whitened NUTS on
the finite-size (Path D) extrinsic posterior, vs the naive 5-D multistart NUTS.

The naive 5-D sampler (demo_slowrot_highsnr_nuts.py) samples (ra,dec,psi,incl,
phiref) directly.  Even with a dense mass matrix its posterior ESS collapses at
high SNR (536 -> 34 -> 18 at SNR 100/300/1000) because the sky posterior is a
thin CURVED ring entangled with the psi/phiref degeneracy, which a global
constant metric cannot whiten.

This driver uses the reparameterization instead (samplers.fisher_nuts_sample_phimarg):
  * phi_ref (phase) is MARGINALIZED analytically (JAXDistPhiMargLikelihood),
    removing the curved psi/phi_ref ridge -> a 4-D (ra,dec,psi,incl) target;
  * each discrete sky mode is Fisher-WHITENED (theta = MAP + A y, A from the
    inverse-Fisher), so the ~1/SNR-narrow ring is O(1) scale in y and NUTS keeps
    a healthy step at any SNR.

Reports the POSTERIOR effective sample size (the honest "resolved the posterior"
metric) and sky recovery, to compare against the naive numbers.

Run on GPU in the JAX container (pin an idle GPU on a shared box):
  apptainer exec --nv <jax_sif> \
    env PYTHON_JULIAPKG_OFFLINE=yes JAX_ENABLE_X64=1 JAX_ILE_DISTMARG_GH=64 \
        CUDA_VISIBLE_DEVICES=<idle> XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
        PYTHONPATH=<Code>:<paper>/analyses/slowrot_finite-size \
    python test/jax/demo_slowrot_reparam.py
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

from RIFT.likelihood.jax_ile.wrapper import build_freqresponse_data_from_precompute
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiMargLikelihood
from RIFT.likelihood.jax_ile import samplers

NETWORK = os.environ.get("SLOWROT_NET", "CE+ET")
QMAX = 4
IWH = 0.03
TBUF = 0.12
NPHI = int(os.environ.get("SLOWROT_NPHI", "32"))
SNRS = [float(x) for x in os.environ.get("SLOWROT_SNRS", "100,300,1000").split(",")]


def _gc_dist(ra, dec, ra0, dec0):
    c = (np.sin(dec) * np.sin(dec0)
         + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _posterior_ess_4(theta_per_chain):
    """Total posterior ESS (within-chain, summed over chains) for 4-D (ra,dec,psi,incl).

    MIN over the sampled dims (worst-mixing direction = the honest number).
    """
    if not theta_per_chain:
        return float("nan")
    from numpyro.diagnostics import effective_sample_size
    per_dim = []
    ndim = theta_per_chain[0].shape[-1]
    for j in range(ndim):
        tot = 0.0
        for th in theta_per_chain:
            x = np.asarray(th)[:, j][None, :]
            try:
                tot += float(effective_sample_size(x))
            except Exception:
                tot += float(np.asarray(th).shape[0])
        per_dim.append(tot)
    return float(np.min(per_dim))


def run_one(src, net, target_snr):
    dist = fslib.distance_for_snr(src, net, target_snr)
    data_dict, psd_dict, arm_dict, meta = fslib.build_finite_size_data(src, net, dist)
    print("\n=== target SNR %.0f  ->  dist=%.2f Mpc  actual SNR=%.1f  half<d|d>=%.3e ==="
          % (target_snr, dist, meta["snr"], meta["half_dd"]))

    P0 = fslib._base_params(src, dist, meta["deltaT"], meta["deltaF"])
    data, _ = build_freqresponse_data_from_precompute(
        P0, data_dict, psd_dict, fslib.EVENT_TIME, IWH, fslib.LMAX, src.fmax,
        t_window=TBUF, Qmax=QMAX, L_arm=arm_dict, analyticPSD_Q=True, verbose=False)

    d_min = max(1.0, dist * 0.3)
    d_max = dist * 2.5
    # phi_ref-marginalized 4-D target (ra,dec,psi,incl); works on banded data.
    like4 = JAXDistPhiMargLikelihood(data, d_min, d_max, nphi=NPHI, n_grid=256)

    res = samplers.fisher_nuts_sample_phimarg(
        like4, num_warmup=300, num_samples=500, n_starts=12, n_modes=4,
        n_prior_pilot=int(max(2e4, 50.0 * target_snr)), seed=1, verbose=True)

    th = np.asarray(res["theta"])              # (N,4): ra,dec,psi,incl
    lnLs = np.asarray(res["lnL"])
    ess = _posterior_ess_4(res.get("theta_per_chain"))
    ra, dec = th[:, 0], th[:, 1]
    imap = int(np.argmax(lnLs))
    d_map = float(_gc_dist(np.array([ra[imap]]), np.array([dec[imap]]),
                           src.ra, src.dec)[0])
    d_near = float(np.min(_gc_dist(ra, dec, src.ra, src.dec)))
    nb = int(np.clip(np.sqrt(len(ra)) / 2.0, 64, 256))
    area = fslib.sky_area_90(ra, dec, np.asarray(res["post_weight"]), nside_bins=nb)
    print("  REPARAM(phimarg+Fisher-whiten): posterior_ESS=%.0f (of %d draws)  "
          "evidence_neff=%.1f  logZ=%.2f" % (ess, len(ra), res["neff"], res["logZ"]))
    print("  sky: MAP d=%.2f deg  nearest=%.2f deg  90%% area=%.3e deg^2 (nbin=%d)  "
          "modes=%d  truth=(%.3f,%.3f)"
          % (d_map, d_near, area, nb, len(res["modes"]), src.ra, src.dec))
    return dict(target_snr=target_snr, snr=meta["snr"], ess=float(ess),
                n=len(ra), neff=float(res["neff"]), area=float(area),
                d_map=d_map, d_near=d_near, n_modes=len(res["modes"]))


def main():
    src = fslib.Source(m1=1.6, m2=1.4, ra=1.2, dec=0.3, psi=0.5, incl=0.4,
                       phiref=0.0, fmin=50.0, fmax=1024.0, seglen=32.0,
                       approx="IMRPhenomD")
    print("network=%s  Qmax=%d  nphi=%d  distmarg_gh=%s  (REPARAMETERIZED sampler)" %
          (NETWORK, QMAX, NPHI, os.environ.get("JAX_ILE_DISTMARG_GH", "0")))
    net = fslib.network(NETWORK)
    rows = []
    for snr in SNRS:
        try:
            rows.append(run_one(src, net, snr))
        except Exception as e:
            import traceback; traceback.print_exc()
            print("  SNR %.0f FAILED: %s" % (snr, e))
    print("\n==== SUMMARY reparam (phimarg+Fisher-whiten), network=%s ====" % NETWORK)
    print("  target_snr  actual_snr  post_ESS  evid_neff  90%_area_deg2  MAP_deg  modes")
    for r in rows:
        print("  %8.0f   %8.1f  %8.0f  %8.1f   %12.3e  %7.2f  %4d"
              % (r["target_snr"], r["snr"], r["ess"], r["neff"], r["area"],
                 r["d_map"], r["n_modes"]))
    print("\n  vs naive 5-D dense-mass NUTS post_ESS: 536 / 34 / 18 at SNR 100/300/1000.")
    print("REPARAM DEMO DONE")


if __name__ == "__main__":
    main()
