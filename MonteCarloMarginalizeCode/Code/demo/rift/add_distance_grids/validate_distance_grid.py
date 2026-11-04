"""Stress-test the distance-grid export at realistic n_eff regimes.

Goals
-----
1. Verify round-trip identity: reconstruct_marginal_lnL(grid) == lnL_marginal,
   to machine precision.
2. Verify the *pure* likelihood interpretation: re-integrating against an
   alternative distance prior yields the closed-form answer.
3. Quantify how badly the per-bin shape degrades as n_eff drops, since RIFT
   routinely runs ILE with low n_eff (50-200) and the user is worried Plan A
   may fail catastrophically.

The synthetic problem mimics ILE's actual setup:
  d ~ q(d) (uniform-in-d sampling proposal)
  pi_d(d) = 3 d^2 / (d_max^3 - d_min^3)              (volumetric prior)
  L(d, Omega) = peak * exp(-0.5 ((d - d0)/sigma_d)^2)  (no Omega dependence)
The marginal is exp(lnL) integrated against pi_d.
"""
import numpy as np

from RIFT.misc.distance_grid import (
    build_distance_grid,
    reconstruct_marginal_lnL,
    _logsumexp,
)


def vol_log_prior(d, d_min=1.0, d_max=4000.0):
    norm = (d_max**3 - d_min**3) / 3.0
    return 2.0*np.log(d) - np.log(norm)


def closed_form_marg(d0, sigma, lnL_peak, d_min, d_max):
    """Marginal under volumetric prior, in the wide-support limit."""
    # E[d^2] ~ d0^2 + sigma^2 (for d well inside box)
    norm = (d_max**3 - d_min**3) / 3.0
    return lnL_peak + np.log(np.sqrt(2*np.pi)*sigma * (d0**2 + sigma**2)) - np.log(norm)


def closed_form_flat(d0, sigma, lnL_peak, d_min, d_max):
    """Marginal under flat-in-d prior."""
    return lnL_peak + np.log(np.sqrt(2*np.pi)*sigma / (d_max - d_min))


def synth_trial(n_samp, n_grid, d0, sigma, lnL_peak, d_min, d_max, rng):
    distance = rng.uniform(d_min, d_max, size=n_samp)
    ln_L = lnL_peak - 0.5*((distance-d0)/sigma)**2
    ln_pi = vol_log_prior(distance, d_min, d_max)
    ln_q = -np.log(d_max - d_min)
    ln_w = ln_L + ln_pi - ln_q
    lnL_marg_mc = _logsumexp(ln_w) - np.log(n_samp)
    grid = build_distance_grid(distance, ln_w, lnL_marg_mc, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=n_grid)
    return distance, ln_w, lnL_marg_mc, grid


def neff_from_ln_weights(ln_w):
    p = np.exp(ln_w - _logsumexp(ln_w))
    return 1.0 / np.sum(p**2)


def density_L2_error(grid, d0, sigma, lnL_peak):
    """L2 relative error of exp(grid['lnL']) vs the true pure likelihood
    L(d) at the bin centers (note: this is exp lnL on its own, not the
    posterior density in d)."""
    d_g = grid["dist"]
    truth = np.exp(lnL_peak - 0.5*((d_g - d0)/sigma)**2)
    recov = np.exp(grid["lnL"])
    denom = np.sqrt(np.mean(truth**2))
    return np.sqrt(np.mean((truth - recov)**2)) / max(denom, 1e-300)


def main():
    rng = np.random.default_rng(20260528)
    d_min, d_max = 1.0, 4000.0
    d0, sigma, lnL_peak = 400.0, 80.0, 37.0
    truth_vol = closed_form_marg(d0, sigma, lnL_peak, d_min, d_max)
    truth_flat = closed_form_flat(d0, sigma, lnL_peak, d_min, d_max)

    print(f"closed-form marg (volumetric): {truth_vol:.3f}")
    print(f"closed-form marg (flat in d):  {truth_flat:.3f}")
    print()
    print(f"{'N':>6} {'n_grid':>7} {'n_eff':>7} "
          f"{'lnL_mc':>10} {'lnL_reco_vol':>13} {'lnL_reco_flat':>14} "
          f"{'reco-mc':>9} {'flat-truth':>11} {'L2 shape':>10}")
    for N in (50, 100, 200, 500, 2000, 10000, 50000):
        n_grid = max(8, min(N//2, 200))
        errs, mc_errs, flat_errs, neffs = [], [], [], []
        for trial in range(50):
            _, ln_w, lnL_mc, grid = synth_trial(N, n_grid, d0, sigma, lnL_peak,
                                                d_min, d_max, rng)
            lnL_reco_vol = reconstruct_marginal_lnL(grid)
            flat = lambda d: -np.log(d_max-d_min) * np.ones_like(np.asarray(d, float))
            lnL_reco_flat = reconstruct_marginal_lnL(grid, ln_prior_d=flat)
            errs.append(density_L2_error(grid, d0, sigma, lnL_peak))
            mc_errs.append(lnL_mc - truth_vol)
            flat_errs.append(lnL_reco_flat - truth_flat)
            neffs.append(neff_from_ln_weights(ln_w))
        last = (lnL_mc, lnL_reco_vol, lnL_reco_flat)
        print(f"{N:6d} {n_grid:7d} {np.median(neffs):7.1f} "
              f"{last[0]:10.4f} {last[1]:13.4f} {last[2]:14.4f} "
              f"{last[1]-last[0]:+9.2e} {np.median(flat_errs):+11.3f} "
              f"{np.median(errs):10.3f}")

    print()
    print("Round-trip identity (lnL_reco_vol - lnL_mc, should be ~machine eps):")
    for n_grid in (4, 8, 32, 200):
        _, _, lnL_mc, grid = synth_trial(500, n_grid, d0, sigma, lnL_peak,
                                          d_min, d_max, rng)
        print(f"  n_grid={n_grid:3d}: {reconstruct_marginal_lnL(grid) - lnL_mc:+.3e}")


if __name__ == "__main__":
    main()
