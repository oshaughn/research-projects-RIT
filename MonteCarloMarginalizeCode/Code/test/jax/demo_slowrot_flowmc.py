"""
flowMC on the finite-size (Path D) high-SNR extrinsic posterior — run in parallel
with the coordinate-rotation NUTS (demo_slowrot_highsnr_nuts.py).

flowMC interleaves local MALA with a normalizing-flow global proposal that LEARNS
the curved multimodal geometry (the sky ring + phase/polarization structure) that
a constant-metric NUTS cannot whiten.  This tests whether flowMC holds up where
naive dense-mass NUTS lost ESS at high SNR.

Requires flowMC importable (installed to ~/flowmc_libs; put it on PYTHONPATH).
Run on GPU in the JAX container (pin an idle GPU):
  apptainer exec --nv <jax_sif> \
    env PYTHON_JULIAPKG_OFFLINE=yes JAX_ENABLE_X64=1 JAX_ILE_DISTMARG_GH=64 \
        CUDA_VISIBLE_DEVICES=<idle> XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
        PYTHONPATH=<Code>:<flowmc_libs>:<paper>/analyses/slowrot_finite-size \
    python test/jax/demo_slowrot_flowmc.py
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
from RIFT.likelihood.jax_ile.wrapper import JAXDistanceMarginalizedLikelihood
from RIFT.likelihood.jax_ile import samplers

NETWORK = os.environ.get("SLOWROT_NET", "CE+ET")
QMAX = 4
IWH = 0.03
TBUF = 0.12
SNRS = [float(x) for x in os.environ.get("SLOWROT_SNRS", "100,300,1000").split(",")]


def _gc_dist(ra, dec, ra0, dec0):
    c = (np.sin(dec) * np.sin(dec0)
         + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


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
    like = JAXDistanceMarginalizedLikelihood(data, d_min, d_max, n_grid=256)

    res = samplers.flowmc_sample(
        like, d_min, d_max, n_chains=20, n_local_steps=20, n_global_steps=20,
        n_training_loops=4, n_production_loops=4, n_epochs=10,
        n_prior_pilot=int(max(2e4, 50.0 * target_snr)), seed=1, verbose=True)

    th = np.asarray(res["theta"]); lnLs = np.asarray(res["lnL"])
    ra, dec = th[:, 0], th[:, 1]
    imap = int(np.argmax(lnLs)) if len(lnLs) else 0
    d_map = float(_gc_dist(np.array([ra[imap]]), np.array([dec[imap]]),
                           src.ra, src.dec)[0]) if len(ra) else float("nan")
    nb = int(np.clip(np.sqrt(max(len(ra), 1)) / 2.0, 64, 256))
    area = fslib.sky_area_90(ra, dec, np.ones_like(ra), nside_bins=nb) if len(ra) else float("nan")
    print("  flowMC: n_draws=%d  evidence_neff=%.1f  logZ=%.2f  90%% area=%.3e deg^2  "
          "MAP d=%.2f deg  truth=(%.3f,%.3f)"
          % (len(ra), res["neff"], res["logZ"], area, d_map, src.ra, src.dec))
    return dict(target_snr=target_snr, snr=meta["snr"], n=len(ra),
                neff=float(res["neff"]), area=float(area), d_map=d_map)


def main():
    src = fslib.Source(m1=1.6, m2=1.4, ra=1.2, dec=0.3, psi=0.5, incl=0.4,
                       phiref=0.0, fmin=50.0, fmax=1024.0, seglen=32.0,
                       approx="IMRPhenomD")
    print("network=%s  Qmax=%d  distmarg_gh=%s  (flowMC)" %
          (NETWORK, QMAX, os.environ.get("JAX_ILE_DISTMARG_GH", "0")))
    net = fslib.network(NETWORK)
    rows = []
    for snr in SNRS:
        try:
            rows.append(run_one(src, net, snr))
        except Exception as e:
            import traceback; traceback.print_exc()
            print("  SNR %.0f FAILED: %s" % (snr, e))
    print("\n==== SUMMARY flowMC (finite-size, network=%s) ====" % NETWORK)
    print("  target_snr  actual_snr  n_draws  evid_neff  90%_area_deg2  MAP_deg")
    for r in rows:
        print("  %8.0f   %8.1f  %7d  %8.1f   %12.3e  %7.2f"
              % (r["target_snr"], r["snr"], r["n"], r["neff"], r["area"], r["d_map"]))
    print("FLOWMC DEMO DONE")


if __name__ == "__main__":
    main()
