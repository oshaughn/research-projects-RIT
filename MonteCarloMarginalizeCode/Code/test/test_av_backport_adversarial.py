"""Adversarial O4c coverage for the #412/#430 density and #455 support contract."""
from pathlib import Path
import numpy as np
import pytest
from scipy.special import logsumexp
from RIFT.integrators import mcsamplerAdaptiveVolume as av

xp = av.xpy_default
host = av.identity_convert


def sampler(bounds, n=4000, adaptive=True):
    s = av.MCSampler(n_chunk=n)
    s.xpy = xp
    s.identity_convert = host
    for i, (lo, hi) in enumerate(bounds):
        s.add_parameter('x'+str(i), pdf=None, left_limit=lo, right_limit=hi,
                        prior_pdf=lambda x: xp.ones(x.shape), adaptive_sampling=adaptive)
    return s


def seed(value):
    np.random.seed(value)
    xp.random.seed(value)


@pytest.mark.parametrize('scale', [1, 2**40])
def test_projected_bin_multiplicity_matches_brute_force(scale):
    rng = np.random.default_rng(193)
    bins = rng.integers(0, 6, size=(100, 3)) * scale
    bins = np.vstack([bins, bins[:20], bins[:20]])
    idx = np.vstack([bins, rng.integers(-1, 7, size=(100, 3)) * scale])
    expected = np.array([np.sum(np.all(bins == row, axis=1)) for row in idx])
    bs = av._BinSet(bins)
    np.testing.assert_array_equal(bs.multiplicity(idx), expected)


@pytest.mark.parametrize('pinned', [(), (0,), (0, 1)])
def test_mixture_matches_direct_bin_sum_with_pins(pinned):
    X = np.array([[.2, .4], [.7, .8]])
    grids = [(np.array([1., 2.]), np.array([[0, 0]]), 100),
             (np.array([.5, 1.]), np.array([[0, 0], [1, 0], [1, 1]]), 60)]
    keep = [i for i in range(2) if i not in pinned]
    direct = np.zeros(len(X))
    for dx, bins, n in grids:
        count = np.sum(np.all(np.floor(X[:, None, keep]/dx[keep]) == bins[None, :, keep], axis=2), axis=1)
        volume = len(bins)*np.prod(dx[keep])*np.prod(np.array([1., 2.])[list(pinned)])
        direct += n*count/volume
    got = av.log_retained_density(X, np.array([0, 0]), grids, [0, 0], [1, 2], pinned)
    np.testing.assert_allclose(got, np.log(direct/len(X)), rtol=1e-13)


@pytest.mark.parametrize('pin', [0., .37, 1.])
def test_pinned_nonunit_box_and_upper_edge(pin):
    seed(311)
    s = sampler([(0., 1.), (-2., 4.)])
    out = s.integrate_log(lambda x0, x1: xp.zeros(len(x0)), n=4000, nmax=8001,
                          neff=1e9, enforce_bounds=True, x0=pin)
    assert out[0] == pytest.approx(np.log(6), abs=1e-10)
    assert np.all(s.binunique[:, 0] == 0)
    assert np.all(s._rvs['x0'] == pin)
    assert out[3]['total_draws'] > 4000


@pytest.mark.parametrize('pin', [0., 1.])
def test_every_coordinate_pinned(pin):
    seed(439)
    s = sampler([(0., 1.), (0., 2.)])
    out = s.integrate_log(lambda x0, x1: xp.zeros(len(x0)), n=100, nmax=250,
                          neff=1e9, x0=pin, x1=pin)
    assert out[0] == pytest.approx(np.log(2), abs=1e-10)
    assert out[2] == pytest.approx(out[3]['total_draws'])


@pytest.mark.parametrize('named', [False, True])
def test_overhanging_proposals_are_counted_zeros_without_evaluating_physics(named):
    seed(455)
    s = sampler([(0., 1.)], n=20000)
    setup = s.setup
    def overhang_setup():
        setup()
        s.dx = np.array([.6])
        s.binunique = np.array([[0], [1]])
        s.ninbin = [10000, 10000]
    s.setup = overhang_setup
    def checked_prior(x):
        assert bool(xp.all((x >= 0) & (x <= 1)))
        return xp.ones(len(x))
    s.prior_pdf['x0'] = checked_prior
    def checked_likelihood(x0):
        assert bool(xp.all((x0 >= 0) & (x0 <= 1)))
        return xp.zeros(len(x0))
    kw = {} if named else {'no_protect_names': True}
    out = s.integrate_log(checked_likelihood, n=20000, nmax=1, neff=1e9,
                          enforce_bounds=False, **kw)
    nr = out[3]['n_live_final']
    assert nr < 19000 and out[3]['total_draws'] == 20000
    assert out[0] == pytest.approx(np.log(1.2*nr/20000), abs=1e-10)
    assert abs(out[0]) < .02
    assert np.exp(out[1]-2*out[0]) == pytest.approx(1/nr-1/20000, rel=1e-8)
    assert np.all((s._rvs['x0'] >= 0) & (s._rvs['x0'] <= 1))


def test_eos_log_wrapper_fairdraw_and_cached_weights():
    seed(412)
    s = sampler([(-2., 3.), (0., 1.)])
    z, v, ne, out = s.integrate(lambda *xs: -.5*(xs[0]/.1)**2,
                               n=4000, nmax=40000, neff=50,
                               use_lnL=True, no_protect_names=True,
                               igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=40)
    assert abs(z - np.log(np.sqrt(2*np.pi)*.1)) < .15
    assert len(s._rvs['x0']) == 40
    assert all(len(values) == 40 for values in s._rvs.values())
    np.testing.assert_array_equal(s._rvs['integrand'], s._rvs['log_integrand'])
    assert np.isfinite(s._rvs['log_joint_s_prior']).all()


def test_nonadaptive_nonuniform_prior_and_reconstructible_evidence():
    seed(430)
    s = sampler([(0., 2.)], n=20000, adaptive=False)
    s.prior_pdf['x0'] = lambda x: x/2  # normalized triangular prior
    z, v, ne, out = s.integrate_log(lambda x0: xp.log(x0), n=20000, nmax=1, neff=1e9)
    # E_prior[x] = 4/3.
    assert abs(z-np.log(4/3)) < .025
    lw = s._rvs['log_integrand']+s._rvs['log_joint_prior']-s._rvs['log_joint_s_prior']
    assert z == pytest.approx(logsumexp(lw)-np.log(len(lw)), abs=1e-12)
    assert out['sigma_lnZ_mc'] > 0


def test_eos_driver_explicitly_requests_bounded_bins():
    path = Path(__file__).parents[1]/'bin/util_ConstructEOSPosterior.py'
    text = path.read_text()
    assert 'if opts.sampler_method == "AV":\n    extra_args["enforce_bounds"] = True' in text


@pytest.mark.parametrize('pin', [-0.5, 1.5])
def test_pin_outside_sampling_limits_is_integrated(pin):
    # A pin is the caller's value, not a draw: only drawn coordinates are tested against the box.
    seed(523)
    s = sampler([(0., 1.), (-2., 4.)])
    out = s.integrate_log(lambda x0, x1: xp.zeros(len(x0)), n=4000, nmax=8001,
                          neff=1e9, enforce_bounds=True, x0=pin)
    assert out[0] == pytest.approx(np.log(6), abs=1e-10)


def test_enforce_bounds_defaults_off():
    # rift_O4d keeps integer bins opt-in; ILE, CIP and the EOS driver request them.
    seed(17)
    s = sampler([(0., 1.), (0., 1.)], n=2000)
    s.integrate_log(lambda x0, x1: -0.5*((x0 - .3)**2 + (x1 - .6)**2)/.01**2,
                    n=2000, nmax=6001, neff=1e9)
    assert np.any(s.nbins != np.floor(s.nbins))
