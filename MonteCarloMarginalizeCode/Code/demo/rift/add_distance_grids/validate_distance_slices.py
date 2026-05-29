"""Synthetic stress-test for the B2 distance-slice estimator.

We synthesize an ILE-like Monte Carlo with two parameters: distance d and a
single Omega-proxy x.  Sample (d, x) from a known proposal q_joint(d, x) =
q_d(d) q_x(x); evaluate the joint likelihood L(d, x) = exp(-0.5*((d - d0(x))/
sigma_d)^2 + lnL_peak); record per-sample integrand and the joint prior /
proposal arrays exactly as ILE's mcsamplerEnsemble does.  Then run the
importance-reweight slice estimator and compare to the closed-form

    L_pure(d_target) = integral L(d_target, x) pi_x(x) dx

For a Gaussian in d with mean d0(x) = d0_const + alpha * x and sigma_d
small, increasing alpha makes Omega couple more strongly to d -- that's the
regime where reweighting is expected to break, and where the bias should
appear.
"""
import numpy as np

# Stand-in for sampler._rvs we'll mock up
class MockSampler:
    pass


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m))) if np.isfinite(m) else m


def synth_run(N, alpha, sigma_d=80.0, d0_const=400.0, lnL_peak=30.0,
              d_min=1.0, d_max=4000.0, x_min=-2.0, x_max=2.0, rng=None):
    rng = rng or np.random.default_rng()
    # Proposals: q_d uniform on [d_min,d_max], q_x uniform on [x_min,x_max]
    d = rng.uniform(d_min, d_max, size=N)
    x = rng.uniform(x_min, x_max, size=N)
    q_d = 1.0/(d_max - d_min) * np.ones(N)
    q_x = 1.0/(x_max - x_min) * np.ones(N)
    # Priors: pi_d volumetric (d^2/(d_max^3/3)), pi_x uniform on [-2,2]
    norm_d = (d_max**3 - d_min**3)/3.0
    pi_d = d**2 / norm_d
    pi_x = 1.0/(x_max - x_min) * np.ones(N)
    # Likelihood (no prior): peaks at d=d0(x) for each x
    d0 = d0_const + alpha*x
    lnL_at_sample = lnL_peak - 0.5 * ((d - d0)/sigma_d)**2
    integrand = np.exp(lnL_at_sample)
    joint_prior = pi_d * pi_x
    joint_s_prior = q_d * q_x
    # Standard MC estimator of marg likelihood:
    w_full = integrand * joint_prior / joint_s_prior
    lnL_marg_mc = np.log(np.mean(w_full))
    # Closed-form marg (over both d and x):
    # integral L(d,x) pi_d pi_x dd dx = (1/norm_d)(1/range_x) integral d^2 exp(-0.5((d-d0(x))/sigma_d)^2) dx dd
    # for sigma_d much less than range, gaussian in d concentrates near d0(x);
    # integrate d^2 against gaussian at d0(x) ~ (d0(x)^2 + sigma_d^2)*sqrt(2pi)*sigma_d.
    # Then average over x uniform on [-2,2]: E[(d0_const + alpha*x)^2 + sigma_d^2] = d0_const^2 + alpha^2*var_x + sigma_d^2
    var_x_uniform = (x_max-x_min)**2 / 12.0
    E_d2 = d0_const**2 + alpha**2 * var_x_uniform + sigma_d**2
    marg_truth = np.exp(lnL_peak) * np.sqrt(2*np.pi) * sigma_d * E_d2 / norm_d
    lnL_marg_truth = np.log(marg_truth)

    # For B2-slice we need a "like_to_integrate"-compatible function that
    # takes (x, distance) arrays and returns lnL (mocking return_lnL=True)
    def like_to_integrate(x, distance):
        d0_arr = d0_const + alpha * np.asarray(x)
        return lnL_peak - 0.5*((np.asarray(distance) - d0_arr)/sigma_d)**2

    rvs = dict(distance=d, x=x, integrand=integrand,
               joint_prior=joint_prior, joint_s_prior=joint_s_prior)
    sampler = MockSampler()
    sampler._rvs = rvs
    sampler.prior_pdf = {"distance": lambda dd: dd**2 / norm_d}
    sampler.pdf = {"distance": lambda dd: np.ones_like(dd) / (d_max - d_min)}
    sampler._pdf_norm = {"distance": 1.0}
    return sampler, like_to_integrate, lnL_marg_mc, lnL_marg_truth, dict(
        d_min=d_min, d_max=d_max, sigma_d=sigma_d, d0_const=d0_const,
        alpha=alpha, lnL_peak=lnL_peak,
    )


def test_wing_placement_and_skip():
    """Unit-check the absolute skip cut and the parabolic wing placement."""
    from RIFT.misc import distance_slices as ds
    print("\n-- is_uninformative (absolute peak cut) --")
    assert ds.is_uninformative(np.array([0.1, 0.3, 0.4, 0.2]))          # undetected
    assert not ds.is_uninformative(np.array([49.8, 50.0, 49.9, 49.85])) # hi-SNR flat
    assert ds.is_uninformative(np.array([np.nan, np.nan]))              # all nan
    print("   ok: undetected skipped, high-SNR-flat kept, nan skipped")

    print("-- parabolic wing placement --")
    A2, dpeak, peak = 8.0e6, 400.0, 30.0
    d_core = np.array([300., 350., 400., 450., 500.])
    lnL_core = peak - 0.5 * A2 * (1.0/d_core - 1.0/dpeak)**2
    a, b, c = ds.fit_lnL_parabola_in_inv_d(d_core, lnL_core)
    assert abs(-2*a - A2) / A2 < 1e-6, "A^2 mis-recovered"
    d_min, d_max = 1.0, 4000.0
    w = ds.pick_wing_centers(d_min, d_max, d_core, 6, lnL_core=lnL_core,
                             lnL_peak=peak, delta_lnL_target=7.0)
    hw = np.sqrt(14.0 / A2)
    d_small, d_large = 1.0/(1.0/dpeak + hw), 1.0/(1.0/dpeak - hw)
    assert w.min() >= d_small - 1e-6 and w.max() <= d_large + 1e-6, \
        "wings escaped parabolic bounds"
    assert np.all((w < d_core.min()) | (w > d_core.max())), "wing inside core"
    print("   ok: A^2 recovered, wings within [{:.1f},{:.1f}] outside core".format(
        d_small, d_large))

    # degenerate -> log-uniform fallback spans the full prior range
    w_fb = ds.pick_wing_centers(d_min, d_max, d_core, 6)
    assert w_fb.min() < d_small and w_fb.max() > d_large, "fallback not full-range"
    print("   ok: degenerate input falls back to full-range log-uniform")


def main():
    from RIFT.misc import distance_slices
    test_wing_placement_and_skip()
    rng = np.random.default_rng(20260528)
    print("Mock 1-Omega problem: L(d, x) = peak exp(-0.5*((d - d0(x))/sigma)^2)")
    print("'alpha' is the d-Omega coupling. alpha=0: separable. alpha=80: peak shifts ~1 sigma per unit x.")
    print(f"\n{'alpha':>7} {'N':>6} {'lnL_truth':>10} {'lnL_mc':>10} "
          f"{'B2_marg':>10} {'diff':>8} {'med slice n_eff':>16}")
    for alpha in (0.0, 10.0, 40.0, 80.0, 160.0):
        for N in (2000, 20000):
            sampler, like, lnL_mc, lnL_truth, meta = synth_run(N, alpha, rng=rng)
            # Run B2 slice
            dL_samp = sampler._rvs["distance"]
            # ln_w_full
            rvs = sampler._rvs
            keep = (rvs['integrand'] > 0)
            ln_w_full = np.full(N, -np.inf)
            ln_w_full[keep] = np.log(rvs['integrand'][keep]) + np.log(rvs['joint_prior'][keep]) - np.log(rvs['joint_s_prior'][keep])
            d_slices = distance_slices.quantile_slice_centers(dL_samp, ln_w_full, 20)
            ln_pi_d_samp = np.log(sampler.prior_pdf['distance'](dL_samp))
            ln_q_d_samp = np.log(sampler.pdf['distance'](dL_samp))
            lnL_k, sigmaL_k, neff_k, _ = distance_slices.importance_reweight_slices(
                sampler, like, d_slices, ln_pi_d_samp, ln_q_d_samp,
                manual_overflow=0.0, return_lnL=True,
            )
            # Build slice table to reconstruct marginal
            ln_pi_d_slices = np.log(sampler.prior_pdf['distance'](d_slices))
            t = distance_slices.build_distance_slice_table(
                d_slices, lnL_k, sigmaL_k, neff_k, N,
                distance_slices.METHOD_REWEIGHT, {}, ln_pi_d_slices,
            )
            lnL_marg_b2 = distance_slices.reconstruct_marginal_lnL(t)
            med_neff = np.median(neff_k)
            diff = lnL_marg_b2 - lnL_truth
            print(f"{alpha:7.1f} {N:6d} {lnL_truth:10.4f} {lnL_mc:10.4f} "
                  f"{lnL_marg_b2:10.4f} {diff:+8.3f} {med_neff:16.1f}")


if __name__ == "__main__":
    main()
