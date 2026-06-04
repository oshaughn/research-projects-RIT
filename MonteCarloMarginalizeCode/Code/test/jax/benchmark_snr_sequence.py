"""
High-SNR benchmark sequence for the JAX ILE extrinsic likelihood + samplers.

Builds zero-noise injections at a sequence of network SNRs (default
40,80,160,320,640) by scaling the luminosity distance (network SNR is exactly
proportional to 1/distance), runs a single flowMC extrinsic evaluation per
source, and records sky recovery, evidence, neff and wall time.  This is the
"single ILE evaluation" sky-recovery demonstration (the basis for the
skymap-vs-SNR figure) and the high-SNR efficiency probe where gradient/flow
sampling is expected to beat the adaptive-Cartesian (AV) integrator: as SNR
grows the extrinsic peak becomes extremely narrow, which is hard for brute-force
adaptive MC but tractable for a gradient-trained flow.

Usage:
  PYTHONPATH=<...>/Code python test/jax/benchmark_snr_sequence.py \
      [--snrs 40,80,160,320,640] [--budget small|full] [--seed 0] \
      [--detectors H1,L1,V1] [--out bench_snr.dat] [--mode flowmc|multistart-nuts]

Writes a results table (one row per SNR) to --out and prints a summary.
"""

import argparse
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lalsimutils
from RIFT.likelihood.jax_ile import build_data_from_precompute
from RIFT.likelihood.jax_ile.wrapper import JAXDistanceMarginalizedLikelihood
from RIFT.likelihood.jax_ile import samplers

MSUN, PC = lal.MSUN_SI, lal.PC_SI
FID = 1126259462.0
# injected extrinsic truth (sky + orientation), fixed across the sequence
TRUTH = dict(ra=1.2, dec=-0.4, psi=0.7, incl=0.9, phiref=2.1)


def make_P(dist_mpc, m1=35.0, m2=30.0):
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = m1 * MSUN, m2 * MSUN
    P.s1z, P.s2z = 0.1, -0.2
    P.fmin = P.fref = 30.0
    P.deltaT, P.deltaF = 1.0 / 4096, 0.25
    P.approx = lalsim.IMRPhenomD
    P.radec = True
    P.tref = FID
    P.phi, P.theta = TRUTH["ra"], TRUTH["dec"]
    P.psi, P.incl, P.phiref = TRUTH["psi"], TRUTH["incl"], TRUTH["phiref"]
    P.dist = dist_mpc * 1e6 * PC
    return P


def synth_data(P, detectors):
    dd, pp = {}, {}
    for det in detectors:
        Pd = P.copy(); Pd.detector = det
        dd[det] = lalsimutils.non_herm_hoff(Pd)
        pp[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower
    return dd, pp


def network_snr(dd, pp, fmin=30.0, fmax=1000.0):
    """Optimal network SNR of the (zero-noise) data = sqrt(sum_det <d|d>)."""
    snr2 = 0.0
    for det in dd:
        d = dd[det]
        fNyq = 1.0 / (2.0 * d.deltaT) if hasattr(d, "deltaT") else \
            d.deltaF * d.data.length / 2.0
        IP = lalsimutils.ComplexIP(fLow=fmin, fNyq=fNyq, deltaF=d.deltaF,
                                   psd=pp[det], fMax=fmax, analyticPSD_Q=True)
        snr2 += float(np.real(IP.norm(d))) ** 2
    return np.sqrt(snr2)


def sky_distance(theta, lnL):
    best = theta[np.argmax(lnL)]
    dlon = best[0] - TRUTH["ra"]
    return float(np.arccos(np.clip(
        np.sin(best[1]) * np.sin(TRUTH["dec"])
        + np.cos(best[1]) * np.cos(TRUTH["dec"]) * np.cos(dlon), -1, 1)))


def sky_credible_area(theta, lnL, frac=0.9):
    """Rough 90% sky credible area (deg^2) from importance-weighted draws."""
    w = np.exp(lnL - lnL.max()); w /= w.sum()
    order = np.argsort(w)[::-1]
    csum = np.cumsum(w[order])
    keep = order[:np.searchsorted(csum, frac) + 1]
    ra, dec = theta[keep, 0], theta[keep, 1]
    # crude area: bounding spread on the sphere
    return float(np.std(ra) * np.std(np.sin(dec)) * (180/np.pi)**2 * 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", default="40,80,160,320,640")
    ap.add_argument("--budget", default="small", choices=["small", "full"])
    ap.add_argument("--mode", default="flowmc", choices=["flowmc", "multistart-nuts"])
    ap.add_argument("--detectors", default="H1,L1,V1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-prior-pilot", type=int, default=40000,
                    help="prior pilot draws to seed the sampler (more needed at high SNR)")
    ap.add_argument("--out", default="bench_snr.dat")
    opts = ap.parse_args()
    snrs = [float(x) for x in opts.snrs.split(",")]
    dets = opts.detectors.split(",")

    if opts.budget == "small":
        fkw = dict(n_chains=20, n_local_steps=15, n_global_steps=15,
                   n_training_loops=3, n_production_loops=3, n_epochs=6)
    else:
        fkw = dict(n_chains=50, n_local_steps=30, n_global_steps=30,
                   n_training_loops=6, n_production_loops=6, n_epochs=12)

    rows = []
    flow_state = None  # demonstrate flow re-use down the sequence
    for snr_target in snrs:
        # calibrate distance to hit the target SNR (SNR ∝ 1/d, one step exact)
        d0 = 1000.0
        dd, pp = synth_data(make_P(d0), dets)
        snr0 = network_snr(dd, pp)
        d_target = d0 * snr0 / snr_target
        P = make_P(d_target)
        dd, pp = synth_data(P, dets)
        snr_real = network_snr(dd, pp)

        data, _ = build_data_from_precompute(
            P.copy(), dd, pp, FID, 0.15, 0.075, 2, 1000.0,
            analyticPSD_Q=True, verbose=False)
        like = JAXDistanceMarginalizedLikelihood(data, 1.0, 20000.0, n_grid=256)
        lnL_truth = like.value([TRUTH[k] for k in ("ra", "dec", "psi", "incl", "phiref")])

        t0 = time.time()
        if opts.mode == "flowmc":
            res = samplers.flowmc_sample(
                like, 1.0, 20000.0, n_prior_pilot=opts.n_prior_pilot,
                reuse_state=flow_state, seed=opts.seed, **fkw)
            flow_state = res.get("flow_state")
        else:
            res = samplers.multistart_nuts(
                like, 1.0, 20000.0, n_starts=fkw["n_chains"] // 4,
                num_warmup=150, num_samples=300,
                n_prior_pilot=opts.n_prior_pilot, seed=opts.seed)
        wall = time.time() - t0

        theta, lnL = res["theta"], res["lnL"]
        dsky = sky_distance(theta, lnL) if len(theta) else float("nan")
        area = sky_credible_area(theta, lnL) if len(theta) else float("nan")
        row = dict(snr=snr_real, d_mpc=d_target, maxlnL=float(lnL.max()) if len(lnL) else np.nan,
                   lnL_truth=lnL_truth, logZ=res["logZ"], neff=res["neff"],
                   sky_d=dsky, sky_area=area, nsamp=len(theta), wall=wall)
        rows.append(row)
        print("SNR=%6.1f  d=%8.1f Mpc  maxlnL=%9.1f (truth %9.1f)  logZ=%9.1f  "
              "neff=%6.1f  sky_d=%.4f  area90~%.3g deg^2  N=%d  %.1fs"
              % (row["snr"], row["d_mpc"], row["maxlnL"], row["lnL_truth"],
                 row["logZ"], row["neff"], row["sky_d"], row["sky_area"],
                 row["nsamp"], row["wall"]))

    cols = ["snr", "d_mpc", "maxlnL", "lnL_truth", "logZ", "neff",
            "sky_d", "sky_area", "nsamp", "wall"]
    arr = np.array([[r[c] for c in cols] for r in rows])
    np.savetxt(opts.out, arr, header=" ".join(cols))
    print("\nWrote %s" % opts.out)


if __name__ == "__main__":
    main()
