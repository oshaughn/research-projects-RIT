#!/usr/bin/env python
"""
AV (mcsamplerAdaptiveVolume.integrate_log) evidence normalization.

The retained set's sampling density used to be 1/(V * box volume), with V the product
of per-cycle survival fractions nrec/ninj.  That product is biased high: each cycle's
occupied bins miss part of the previous live region -- its low-likelihood rim -- so new
draws over-sample the interior and survive the next threshold too often.  The bias
compounds over cycles, so lnZ came out high on any sharply peaked integrand.

integrate_log now gives each retained point the density of the mixture of every grid it
drew from (log_retained_density).  These tests pin that on a target with a closed-form
evidence, and the mixture density itself on hand-built grids.
Measurements: RIFT_roboto_paper analyses/av_lnz_normalization_20261006 (2026-10).
"""

import numpy as np
import pytest

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV
from RIFT.integrators.mcsamplerAdaptiveVolume import log_retained_density


def _gaussian_lnZ_error(seed, sig=0.003, ndim=4, n_chunk=2000, nmax=200000, pinned=None):
    """lnZ(AV) - lnZ(exact) for an isotropic Gaussian of width sig centred in the unit box."""
    np.random.seed(seed)
    names = ['x%d' % i for i in range(ndim)]

    def lnF(*xs):
        X = np.array(xs).T - 0.5
        if pinned is not None:
            X = X[:, 1:]          # the pinned coordinate does not enter the integrand
        return -0.5 * np.sum(X**2, axis=1) / sig**2

    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    s.xpy = np
    s.identity_convert = lambda x: x
    for name in names:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    kw = {} if pinned is None else {names[0]: pinned}
    out = s.integrate_log(lnF, *names, nmax=nmax, neff=50, n=n_chunk,
                          no_protect_names=True, verbose=False, **kw)
    n_free = ndim if pinned is None else ndim - 1
    return out[0] - n_free * np.log(np.sqrt(2 * np.pi) * sig)


def test_narrow_gaussian_lnZ_matches_closed_form():
    """Fails with the scalar-V normalization, which is biased high on this target."""
    err = np.array([_gaussian_lnZ_error(seed) for seed in (1, 2, 3)])
    assert abs(err.mean()) < 0.15, "lnZ - exact = {} (mean {:.3f})".format(err, err.mean())
    assert np.all(np.abs(err) < 0.3), "lnZ - exact = {}".format(err)


def test_prior_narrower_than_box_keeps_normalization():
    """Prior zero on x0 >= 0.5 (density 2 below).  Draws there return -inf and still count as
    draws; the scalar-V normalization dropped them and came out high by about ln 2."""
    sig, ndim = 0.01, 4
    names = ['x%d' % i for i in range(ndim)]
    mu = np.array([0.25] + [0.5] * (ndim - 1))
    exact = ndim * np.log(np.sqrt(2 * np.pi) * sig) + np.log(2.0)
    err = []
    for seed in (1, 2, 3):
        np.random.seed(seed)
        s = mcsamplerAV.MCSampler(n_chunk=4000)
        s.xpy = np
        s.identity_convert = lambda x: x
        for i, name in enumerate(names):
            pdf = (lambda x: 2.0 * (np.asarray(x) < 0.5)) if i == 0 else (lambda x: np.ones(np.shape(x)))
            s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0, prior_pdf=pdf,
                            adaptive_sampling=True)
        out = s.integrate_log(lambda *xs: -0.5 * np.sum((np.array(xs).T - mu)**2, axis=1) / sig**2,
                              *names, nmax=400000, neff=50, n=4000, no_protect_names=True, verbose=False)
        err.append(out[0] - exact)
    err = np.array(err)
    assert abs(err.mean()) < 0.15, "lnZ - exact = {} (mean {:.3f})".format(err, err.mean())


def test_pinned_coordinate_keeps_normalization():
    """A pinned coordinate is drawn at one value; with a unit-width uniform prior on it the
    evidence equals that of the remaining free coordinates."""
    err = _gaussian_lnZ_error(1, ndim=4, pinned=0.37)
    assert abs(err) < 0.3, "pinned lnZ - exact = {:.3f}".format(err)


def test_density_single_full_box_grid():
    """One cycle drawn uniformly over a box of volume 2*3: density n/(N_ret * 6) everywhere."""
    X = np.array([[0.1, 0.2], [1.9, 2.9], [1.0, 1.5]])
    grids = [(np.array([2.0, 3.0]), np.array([[0, 0]]), 100)]
    out = log_retained_density(X, np.zeros(3, dtype=int), grids, np.zeros(2), np.array([2.0, 3.0]))
    assert np.allclose(out, np.log(100) - np.log(3) - np.log(6.0))


def test_density_sums_the_grids_that_cover_each_point():
    """Box grid (n=10, volume 1) plus a 2x2 grid with one occupied bin (n=30, volume 1/4)."""
    X = np.array([[0.2, 0.2],    # inside the occupied fine bin (0, 0)
                  [0.8, 0.8]])   # outside it: only the box grid covers this point
    grids = [(np.array([1.0, 1.0]), np.array([[0, 0]]), 10),
             (np.array([0.5, 0.5]), np.array([[0, 0]]), 30)]
    out = log_retained_density(X, np.array([1, 0]), grids, np.zeros(2), np.ones(2))
    expect = np.log([10 / 1.0 + 30 / 0.25, 10 / 1.0]) - np.log(2)
    assert np.allclose(out, expect)


def test_density_counts_a_point_in_its_own_grid_at_a_bin_edge():
    """A draw that rounds onto its bin's upper edge still carries its own grid's density."""
    X = np.array([[0.5, 0.25]])          # floor(0.5/0.5) = 1: outside occupied bin (0, 0)
    grids = [(np.array([0.5, 0.5]), np.array([[0, 0]]), 8)]
    out = log_retained_density(X, np.array([0]), grids, np.zeros(2), np.ones(2))
    assert np.isfinite(out[0])
    assert np.isclose(out[0], np.log(8 / 0.25))


def test_density_pinned_dimension_uses_full_width():
    """Pinned dims are excluded from membership and enter with dx0."""
    X = np.array([[0.2, 1.0]])           # second coordinate pinned at the box's upper edge
    grids = [(np.array([0.5, 1.0]), np.array([[0, 0]]), 4)]
    out = log_retained_density(X, np.array([0]), grids, np.zeros(2), np.array([1.0, 1.0]),
                               pinned_dims=[1])
    assert np.isclose(out[0], np.log(4 / 0.5))


def test_bin_set_matches_brute_force_on_both_storage_paths():
    """Sorted-key storage, and the row fallback used when keys would overflow int64."""
    from RIFT.integrators.mcsamplerAdaptiveVolume import _BinSet
    rng = np.random.default_rng(3)
    for scale in (1, 2**40):
        bins = rng.integers(0, 6, size=(50, 3)) * scale
        idx = np.vstack([bins[:20], rng.integers(-1, 7, size=(200, 3)) * scale])
        bs = _BinSet(bins)
        assert (bs.rows is not None) == (scale > 1)
        want = np.array([tuple(r) in set(map(tuple, bins.tolist())) for r in idx.tolist()])
        np.testing.assert_array_equal(bs.contains(idx), want)


def _one_cycle_constant(prior_x, warm=None, pin=None, seed=1, n=4000):
    """One draw cycle of a constant likelihood on the unit square; returns (lnZ, rel sigma)."""
    np.random.seed(seed)
    s = mcsamplerAV.MCSampler(n_chunk=n)
    s.xpy = np
    s.identity_convert = lambda x: x
    s.add_parameter('x', pdf=None, left_limit=0.0, right_limit=1.0, prior_pdf=prior_x, adaptive_sampling=True)
    s.add_parameter('y', pdf=None, left_limit=0.0, right_limit=1.0,
                    prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    if warm is not None:
        s._warm = warm
    kw = {} if pin is None else {'x': pin}
    out = s.integrate_log(lambda *xs: np.zeros(len(np.atleast_1d(xs[0]))), 'x', 'y', nmax=n + 100,
                          neff=10, n=n, no_protect_names=True, verbose=False, **kw)
    return out[0], np.sqrt(np.exp(out[1] - 2 * out[0]))


def test_warm_grid_binned_in_a_later_pinned_dimension():
    """Warm bins (0,0), (1,0), (1,1) with x pinned: y bin 0 carries 2 bins' draws, y bin 1 one.
    With L = 1[y < 1/2] the exact lnZ is ln(1/2).  Counting each projected bin once gives
    ln 2 too high on the full grid; dividing by the unique projected count gives ln(2/3)."""
    for seed in (1, 2):
        np.random.seed(seed)
        s = mcsamplerAV.MCSampler(n_chunk=4000)
        s.xpy = np
        s.identity_convert = lambda x: x
        for name in ('x', 'y'):
            s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                            prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
        s._warm = dict(binunique=np.array([[0, 0], [1, 0], [1, 1]]), dx=np.array([0.5, 0.5]),
                       nbins=np.array([2, 2]), V=1.0, loglkl_thr=-1e15)
        out = s.integrate_log(lambda x, y: np.where(np.asarray(y) < 0.5, 0.0, -np.inf), 'x', 'y',
                              nmax=4100, neff=10, n=4000, no_protect_names=True, verbose=False, x=0.3)
        assert out[0] == pytest.approx(np.log(0.5), abs=1e-9), out[0]


def test_rejected_draws_enter_the_reported_variance():
    """Prior 2 on x < 0.5: every retained weight is equal, but the estimate 2*N_ret/N_drawn is
    binomial, relative sigma sqrt((1-p)/(p N)) = 1/sqrt(N) at p = 1/2.  Reported ~1e-10 before."""
    lnZ, rel_sigma = _one_cycle_constant(lambda x: 2.0 * (np.asarray(x) < 0.5))
    assert abs(lnZ) < 0.1
    assert rel_sigma == pytest.approx(1 / np.sqrt(4001), rel=0.05)


def test_rejected_draws_counted_over_every_cycle():
    """Two cycles (box, then the bins of the x < 1/2 survivors).  Every retained point lies in
    both grids, so all weights are equal and rel var = 1/N_retained - 1/N_drawn exactly, with
    N_drawn summed over both cycles."""
    np.random.seed(1)
    n = 4000
    s = mcsamplerAV.MCSampler(n_chunk=n)
    s.xpy = np
    s.identity_convert = lambda x: x
    # Priors vanish outside the box: cycle-2 bins overhang it, and a draw retained out there
    # would be covered by cycle 2 only.
    s.add_parameter('x', pdf=None, left_limit=0.0, right_limit=1.0, adaptive_sampling=True,
                    prior_pdf=lambda x: 2.0 * ((np.asarray(x) >= 0) & (np.asarray(x) < 0.5)))
    s.add_parameter('y', pdf=None, left_limit=0.0, right_limit=1.0, adaptive_sampling=True,
                    prior_pdf=lambda x: 1.0 * ((np.asarray(x) >= 0) & (np.asarray(x) <= 1)))
    out = s.integrate_log(lambda *xs: np.zeros(len(np.atleast_1d(xs[0]))), 'x', 'y', nmax=n + 2,
                          neff=1e9, n=n, no_protect_names=True, verbose=False, dict_return=True)
    n_ret = out[3]['n_live_final']
    n_drawn = s.last_stopping_statistics['total_draws']
    assert n_drawn > n + 1, 'the run must draw a second cycle'
    rel_var = np.exp(out[1] - 2 * out[0])
    assert rel_var == pytest.approx(1.0 / n_ret - 1.0 / n_drawn, rel=1e-6)


def test_bootstrap_resamples_the_survivor_count():
    """10 equal weights out of 4001 draws: resampling the survivors alone gives a zero-width
    interval; with n_drawn the survivor count varies and so does lnZ."""
    from RIFT.integrators.statutils import bootstrap_lnZ_quantiles
    lw = np.zeros(10)
    fixed = bootstrap_lnZ_quantiles(lw, n_total=10, rng_seed=1)
    assert fixed[-1] - fixed[0] == 0.0
    q = bootstrap_lnZ_quantiles(lw, n_total=10, rng_seed=1, n_drawn=4001)
    assert q[0] < 0.0 < q[-1] and q[-1] - q[0] > 0.5, q


def test_av_interval_counts_rejected_draws():
    """Constant likelihood, prior on x < 0.0025: 11 of 4001 draws survive (seed 1), relative
    sigma 0.30 triggers the bootstrap.  The interval had zero width."""
    f = 0.0025
    np.random.seed(1)
    s = mcsamplerAV.MCSampler(n_chunk=4000)
    s.xpy = np
    s.identity_convert = lambda x: x
    s.add_parameter('x', pdf=None, left_limit=0.0, right_limit=1.0, adaptive_sampling=True,
                    prior_pdf=lambda x: (np.asarray(x) < f) / f)
    s.add_parameter('y', pdf=None, left_limit=0.0, right_limit=1.0, adaptive_sampling=True,
                    prior_pdf=lambda x: np.ones(np.shape(x)))
    out = s.integrate_log(lambda *xs: np.zeros(len(np.atleast_1d(xs[0]))), 'x', 'y', nmax=4100,
                          neff=5, n=4000, no_protect_names=True, verbose=False)
    q = out[3].get('lnZ_ci90')
    assert q is not None, 'the bootstrap did not run: premise of this test broke'
    assert q[0] < out[0] < q[-1] and q[-1] - q[0] > 0.5, (q, out[0])


# --- dilation (dilate_layers) ----------------------------------------------------------------
# Each cycle's bins come from the retained points.  On a thin, curved live region that leaves
# parts of the region with no bin, and integrate_log never draws there again, so lnZ comes out
# low.  dilate_bins adds the axis neighbours of the occupied bins before the next draw.

def test_dilate_bins_adds_axis_neighbours_inside_the_grid():
    from RIFT.integrators.mcsamplerAdaptiveVolume import dilate_bins
    out = dilate_bins(np.array([[0, 0], [2, 3]]), np.array([3.0, 4.0]), [0, 1])
    want = {(0, 0), (1, 0), (0, 1),                 # (0,0): -1 steps fall off the grid
            (2, 3), (1, 3), (2, 2)}                 # (2,3): +1 steps fall off the grid
    assert set(map(tuple, out.tolist())) == want


def test_dilate_bins_only_on_given_axes_and_layers():
    from RIFT.integrators.mcsamplerAdaptiveVolume import dilate_bins
    b = np.array([[2, 2]])
    nb = np.array([5.0, 5.0])
    assert np.array_equal(dilate_bins(b, nb, [0, 1], layers=0), b)
    assert set(map(tuple, dilate_bins(b, nb, [1]).tolist())) == {(2, 1), (2, 2), (2, 3)}
    two = set(map(tuple, dilate_bins(b, nb, [0, 1], layers=2).tolist()))
    assert len(two) == 13 and (4, 2) in two and (3, 3) in two and (4, 3) not in two


def _thin_ring_lnZ_error(seed, dilate_layers, sig=0.0005, n_chunk=8000):
    """lnZ(AV) - exact for a ring of radius 0.3 and width sig in (x0, x1), Gaussian of width
    0.05 in four more coordinates, centred in the unit box."""
    np.random.seed(seed)
    names = ['x%d' % i for i in range(6)]
    r0, sb = 0.3, 0.05
    exact = np.log(2 * np.pi * r0) + np.log(np.sqrt(2 * np.pi) * sig) + 4 * np.log(np.sqrt(2 * np.pi) * sb)

    def lnF(*xs):
        Y = np.array(xs).T - 0.5
        r = np.hypot(Y[:, 0], Y[:, 1])
        return -0.5 * ((r - r0) / sig)**2 - 0.5 * np.sum((Y[:, 2:] / sb)**2, axis=1)

    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    s.xpy = np
    s.identity_convert = lambda x: x
    for name in names:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    out = s.integrate_log(lnF, *names, nmax=4000000, neff=50, n=n_chunk, no_protect_names=True,
                          verbose=False, enforce_bounds=True, dilate_layers=dilate_layers)
    return out[0] - exact


def test_dilation_keeps_normalization():
    """The dilated bins enter the mixture density, so lnZ stays exact on a Gaussian."""
    err = []
    for seed in (1, 2, 3):
        np.random.seed(seed)
        names = ['x%d' % i for i in range(4)]
        s = mcsamplerAV.MCSampler(n_chunk=2000)
        s.xpy = np
        s.identity_convert = lambda x: x
        for name in names:
            s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                            prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
        out = s.integrate_log(lambda *xs: -0.5 * np.sum((np.array(xs).T - 0.5)**2, axis=1) / 0.003**2,
                              *names, nmax=400000, neff=50, n=2000, no_protect_names=True,
                              verbose=False, dilate_layers=1)
        err.append(out[0] - 4 * np.log(np.sqrt(2 * np.pi) * 0.003))
    err = np.array(err)
    assert abs(err.mean()) < 0.15, "lnZ - exact = {} (mean {:.3f})".format(err, err.mean())


def test_dilation_recovers_thin_ring_evidence():
    """Without dilation AV loses parts of the ring during contraction and lnZ is low."""
    seeds = (1, 2, 3, 4)
    off = np.array([_thin_ring_lnZ_error(s, 0) for s in seeds])
    on = np.array([_thin_ring_lnZ_error(s, 1) for s in seeds])
    assert off.mean() < -0.1, "undilated lnZ - exact = {}".format(off)
    assert abs(on.mean()) < 0.1, "dilated lnZ - exact = {}".format(on)
