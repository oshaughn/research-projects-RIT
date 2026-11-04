"""Warm-started distance slices: known-answer integrals for the adaptive importance sampler.

Targets are Gaussians (correlated, displaced from the seed, cut by the box) whose integrals are known,
and a family whose mode moves with d, run through WarmSliceSampler and the walk-out.
"""
import numpy as np
import pytest

from RIFT.misc import distance_slices as ds

P = 5
LO, HI = np.zeros(P), np.ones(P)
MU = np.array([0.3, 0.6, 0.5, 0.2, 0.8])
S = np.array([0.02, 0.05, 0.03, 0.04, 0.02])
RHO = 0.9
LNZ = 0.5 * P * np.log(2 * np.pi) + np.sum(np.log(S)) + 0.5 * np.log(1 - RHO ** 2)


def ln_gauss(x, mu=MU):
    z = (x - mu) / S
    z0 = (z[:, 0] - RHO * z[:, 1]) / np.sqrt(1 - RHO ** 2)
    return -0.5 * (z0 ** 2 + np.sum(z[:, 1:] ** 2, 1))


def seed_at(mu, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return mu + S * rng.standard_normal((n, P)), np.zeros(n)


def run(ln_f, seed, reps, **kw):
    out = []
    for r in range(reps):
        res = ds.warm_integrate_slice(ln_f, LO, HI, seed[0], seed[1], n_max=200000,
                                      n_eff_target=30, rng=np.random.default_rng(10 + r), n_block=2000,
                                      khat_max=0.7, **kw)
        out.append((res["lnZ"], res["sigma"], res["neff"], res["ntotal"], res["converged"] and res["khat"] <= 0.7))
    return np.array(out)


def test_kde_density_matches_an_explicit_gaussian_mixture():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(1)
    x, lw = seed_at(MU)
    kde = ds.fit_kde_proposal(x, lw, LO, HI, n_kernel=50, rng=rng)
    H = kde["L"] @ kde["L"].T
    y = ds.kde_draw(kde, 7, rng)
    ref = np.log(np.mean([multivariate_normal(c, H).pdf(y) for c in kde["centers"]], axis=0))
    np.testing.assert_allclose(ds.kde_logpdf(kde, y), ref, rtol=0, atol=1e-8)


def test_good_seed_is_unbiased_and_calibrated_at_two_blocks():
    r = run(ln_gauss, seed_at(MU), 20)
    err = r[:, 0] - LNZ
    assert abs(err.mean()) < 4 * err.std() / np.sqrt(len(err)) + 0.01
    assert 0.6 < np.std(err / r[:, 1]) < 1.5
    assert np.median(r[:, 3]) <= 3 * 2000


@pytest.mark.parametrize("disp", [1.0, 2.0])
def test_displaced_seed_costs_samples_not_bias(disp):
    seed = seed_at(MU + disp * S * np.array([1, 1, -1, 1, -1]))
    r = run(ln_gauss, seed, 20)
    err = r[:, 0] - LNZ
    assert abs(err.mean()) < 0.1, err.mean()
    assert np.std(err / r[:, 1]) < 2.0


def test_no_seed_falls_back_to_uniform_box():
    wide = lambda x: -0.5 * np.sum(((x - 0.5) / 0.2) ** 2, 1)
    truth = P * np.log(0.2 * np.sqrt(2 * np.pi) * (2 * 0.9938790 - 1))   # erf(2.5/sqrt2) per axis
    r = run(wide, (None, None), 5)
    assert np.all(np.abs(r[:, 0] - truth) < 5 * r[:, 1] + 0.02), (r[:, 0] - truth, r[:, 1])


def test_box_edge_cuts_the_integral():
    mu = MU.copy()
    mu[4] = 1.0                      # mode on the upper edge of axis 4: half the mass is outside
    r = run(lambda x: ln_gauss(x, mu), seed_at(mu), 10)
    err = r[:, 0] - (LNZ + np.log(0.5))
    assert abs(err.mean()) < 0.05, err.mean()


# --- the slice sampler on a d-family -------------------------------------------------------

class FakeSampler:
    def __init__(self):
        names = ["right_ascension", "declination"]
        self.llim = {q: 0.0 for q in names}
        self.rlim = {q: 1.0 for q in names}
        self.prior_pdf = {q: (lambda x: np.ones_like(x)) for q in names}


SD = 0.03


def mode(d):
    return 0.5 + 0.08 * np.log(np.asarray(d, float) / 1000.0)      # the conditional moves with d


def lnc(d):
    return 40.0 - 0.5 * ((1000.0 / np.asarray(d, float) - 1.0) / 0.2) ** 2


def like(right_ascension, declination, distance):
    return (lnc(distance) - 0.5 * ((right_ascension - mode(distance)) / SD) ** 2
            - 0.5 * ((declination - 0.4) / SD) ** 2)


def truth(d):
    return lnc(d) + np.log(2 * np.pi * SD ** 2)


def make_warm(rng=1):
    r = np.random.default_rng(rng)
    seed = {"right_ascension": mode(1000.0) + SD * r.standard_normal(500),
            "declination": 0.4 + SD * r.standard_normal(500)}
    return ds.WarmSliceSampler(FakeSampler(), like, seed, np.zeros(500), seed_distance=np.full(500, 1000.0),
                               rng=np.random.default_rng(rng), verbose=False, n_block=2000)


def test_slices_follow_a_moving_conditional():
    d = np.array([600.0, 800.0, 1000.0, 1250.0, 1600.0])     # mode moves by ~3 SD across the set
    w = make_warm()
    lnL, sig, ne, nt = w.run(d, n_max=100000, n_eff_target=30)
    assert np.all(np.abs(lnL - truth(d)) < 4 * sig + 0.02), (lnL - truth(d), sig)
    assert np.all(ne >= 30) and np.all(nt <= 4 * 2000)
    assert len(w.done) == len(d)


def test_walkout_runner_uses_the_warm_sampler():
    w = make_warm(2)
    d = np.linspace(900.0, 1100.0, 9)
    lnL = w.run(d, n_max=100000, n_eff_target=30)[0]
    why = []
    added = ds.walk_out_fresh_slices(None, None, d, lnL, lambda x: np.zeros_like(np.asarray(x, float)),
                                     100.0, 1e4, lnL_offset=0.0, delta_lnL=5.0, max_steps=8, n_max=100000,
                                     n_eff_target=30, n_chunk=2000, reasons=why,
                                     runner=lambda dd: w.run(dd, n_max=100000, n_eff_target=30))
    assert len(added[0]) > 0 and why == ["target", "target"]
    assert np.all(np.abs(added[1] - truth(added[0])) < 4 * added[2] + 0.02)


class PeriodicSampler:
    def __init__(self):
        self.llim = {"right_ascension": 0.0, "declination": -1.0}
        self.rlim = {"right_ascension": 2 * np.pi, "declination": 1.0}
        self.prior_pdf = {"right_ascension": lambda x: np.full_like(x, 1 / (2 * np.pi)),
                          "declination": lambda x: np.full_like(x, 0.5)}


def test_mode_at_the_periodic_wrap_point():
    # sky mode at RA = 0 = 2 pi: half the seed sits near 0, half near 2 pi
    s_ra, s_dec = 0.05, 0.05

    def like_ra(right_ascension, declination, distance):
        dra = np.angle(np.exp(1j * right_ascension))           # wrapped offset from RA = 0
        return -0.5 * (dra / s_ra) ** 2 - 0.5 * ((declination - 0.2) / s_dec) ** 2

    truth = np.log(2 * np.pi * s_ra * s_dec / (2 * np.pi * 2))
    r = np.random.default_rng(3)
    ra = np.mod(s_ra * r.standard_normal(500), 2 * np.pi)
    seed = {"right_ascension": ra, "declination": 0.2 + s_dec * r.standard_normal(500)}
    w = ds.WarmSliceSampler(PeriodicSampler(), like_ra, seed, np.zeros(500), rng=np.random.default_rng(4),
                            verbose=False, n_block=2000)
    lnL, sig, ne, nt = w.run(np.array([1000.0]), n_max=100000, n_eff_target=30)
    assert abs(lnL[0] - truth) < 4 * sig[0] + 0.02, (lnL[0] - truth, sig[0])
    assert nt[0] <= 4 * 2000 and len(w.done) == 1


def test_running_neff_matches_concatenated():
    r = np.random.default_rng(5)
    blocks = [r.normal(0, s, 300) for s in (0.5, 3.0, 1.0)] + [np.array([-np.inf, 9.0])]
    run = ds._RunningNeff()
    for b in blocks:
        run.add(b)
    assert abs(run.neff - ds._neff(np.concatenate(blocks))) < 1e-9 * run.neff


def test_local_kernel_density_matches_an_explicit_mixture():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(6)
    x, lw = seed_at(MU)
    kde = ds.fit_kde_proposal(x, lw, LO, HI, n_kernel=40, rng=rng, local_k=15, n_pool=200)
    assert "Ls" in kde and kde["Ls"].shape == (40, P, P)
    y = ds.kde_draw(kde, 6, rng)
    ref = np.log(np.mean([multivariate_normal(c, L @ L.T).pdf(y) for c, L in zip(kde["centers"], kde["Ls"])], axis=0))
    np.testing.assert_allclose(ds.kde_logpdf(kde, y), ref, rtol=0, atol=1e-8)


def test_local_kernels_fall_back_when_the_seed_collapses():
    # all the weight on three of 300 rows: the pool has at most three distinct points in five dimensions
    rng = np.random.default_rng(7)
    x, _ = seed_at(MU)
    lw = np.full(len(x), -1000.0)
    lw[:3] = 0.0
    kde = ds.fit_kde_proposal(x, lw, LO, HI, n_kernel=40, rng=rng, local_k=30)
    assert "L" in kde and "Ls" not in kde
    ref = ds.fit_kde_proposal(x, lw, LO, HI, n_kernel=40, rng=np.random.default_rng(7))   # shared kernel
    np.testing.assert_allclose(kde["L"], ref["L"], rtol=1e-12)
    assert kde["ln_norm"] == ref["ln_norm"] and kde["neff"] == ref["neff"]
    np.testing.assert_array_equal(kde["centers"], ref["centers"])
    # many copies of three distinct values: the pool dedups by value and also falls back
    xd = np.repeat(x[:3], 100, axis=0)
    kd = ds.fit_kde_proposal(xd, np.zeros(len(xd)), LO, HI, n_kernel=40, rng=rng, local_k=30)
    assert "L" in kd and "Ls" not in kd
    y = ds.kde_draw(kde, 100, rng)
    assert np.all(np.isfinite(ds.kde_logpdf(kde, y)))


def test_collapsed_seed_is_never_trusted():
    # Slice seed's weight on three rows, broad component healthy: without the collapse flag this run
    # converges (block agreement at the 2nd block), which is the case the #511 review showed can be
    # 1-3 nats low.  It must report not converged so WarmSliceSampler redoes it cold.
    x, _ = seed_at(MU)
    lw = np.full(len(x), -1000.0)
    lw[:3] = 0.0
    res = ds.warm_integrate_slice(ln_gauss, LO, HI, x, lw, n_max=200000, n_eff_target=30,
                                  rng=np.random.default_rng(0), local_k=30, broad=(x, np.zeros(len(x))))
    assert res["collapsed"] and not res["converged"]
    # a collapsed broad component marks the slice too; a healthy seed and broad do not
    res = ds.warm_integrate_slice(ln_gauss, LO, HI, x, np.zeros(len(x)), n_max=200000, n_eff_target=30,
                                  rng=np.random.default_rng(0), local_k=30, broad=(x, lw))
    assert res["collapsed"] and not res["converged"]
    good = ds.warm_integrate_slice(ln_gauss, LO, HI, x, np.zeros(len(x)), n_max=200000, n_eff_target=30,
                                   rng=np.random.default_rng(0), local_k=30, broad=(x, np.zeros(len(x))))
    assert not good["collapsed"] and good["converged"]
