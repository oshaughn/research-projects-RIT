"""The JAX ILE driver's --save-samples export must be a FAIR DRAW.

Downstream tooling (and every consumer of ``*_samples.dat``) treats the exported
rows as equal-weight draws from the conditional extrinsic posterior: the file has
no weight column, exactly like production ILE's ``--fairdraw-extrinsic-output``
export, which multinomial-resamples against ``w = L p / p_s`` inside
``RIFT/integrators/mcsampler.py::integrate`` before writing.

Several JAX-driver modes produce samples that do NOT follow the posterior --
``laplace-is`` (the DEFAULT mode) draws from a Gaussian proposal, ``prior-mc``
draws from the prior, and the flowMC modes sample a tempered target when
``--adapt-weight-exponent != 1`` -- while computing the correcting importance
weights and, before this test's change, discarding them.

The tests below drive the shipped ``write_samples`` (not a helper in isolation)
on a target whose exact posterior moments are known analytically, so a regression
that drops the reweighting again shows up as a wrong exported distribution.

Run:
  PYTHONPATH=<...>/Code  python -m pytest -q test/jax/test_jax_fairdraw_export.py
"""

import importlib.machinery
import importlib.util
import os
import types

import numpy as np
import pytest


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DRIVER = os.path.join(CODE_DIR, "bin", "integrate_likelihood_extrinsic_jax")


def load_driver():
    """Import the driver script (no .py suffix) as a module."""
    loader = importlib.machinery.SourceFileLoader("_ile_jax_driver", DRIVER)
    spec = importlib.util.spec_from_loader("_ile_jax_driver", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


drv = load_driver()


# ---------------------------------------------------------------------------
# A 4-D target with analytically known posterior moments.
#
#   proposal   q(x) = N(0, s_q^2 I)          <- what the sampler actually drew
#   prior      p(x) = N(0, s_p^2 I)          (flat enough over the support)
#   likelihood L(x) = N(x; mu_L, s_L^2 I)
#
# so the posterior is Gaussian with
#   var_post = 1 / (1/s_p^2 + 1/s_L^2),  mean_post = var_post * mu_L / s_L^2
# and  ln w = ln L + ln p - ln q  is the exact importance weight.
# ---------------------------------------------------------------------------
S_Q, S_P, S_L = 3.0, 5.0, 0.8
MU_L = np.array([1.4, -0.9, 0.6, 2.0])
NDIM = 4
VAR_POST = 1.0 / (1.0 / S_P ** 2 + 1.0 / S_L ** 2)
MEAN_POST = VAR_POST * MU_L / S_L ** 2
SD_POST = np.sqrt(VAR_POST)


def _logN(x, mu, s):
    return (-0.5 * np.sum((x - mu) ** 2, axis=1) / s ** 2
            - x.shape[1] * np.log(s * np.sqrt(2 * np.pi)))


def make_cloud(n=400000, seed=7):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((n, NDIM)) * S_Q
    lnL = _logN(theta, MU_L, S_L)
    logw = lnL + _logN(theta, np.zeros(NDIM), S_P) - _logN(theta, np.zeros(NDIM), S_Q)
    return theta, lnL, logw


def fake_opts(tmpdir, **kw):
    o = types.SimpleNamespace(
        output_file=os.path.join(str(tmpdir), "OUT"), save_samples=True,
        mode="flowmc-phimarg", seed=11,
        fairdraw_extrinsic_output=False, fairdraw_extrinsic_output_n_max=5,
        n_fairdraw_extrinsic_samples=None)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def read_export(opts, idx=0):
    f = opts.output_file + "_" + str(idx) + "_samples.dat"
    assert os.path.exists(f), "write_samples wrote nothing"
    with open(f) as fh:
        hdr = fh.readline()
    return np.loadtxt(f), hdr


# write_samples' 4-D branch emits columns (ra, dec, incl, psi, lnL), i.e.
# theta columns 0, 1, 3, 2.  Map back so we compare like with like.
COL_OF_THETA = {0: 0, 1: 1, 2: 3, 3: 2}


def test_export_is_a_fair_draw_of_the_posterior(tmp_path):
    """With importance weights supplied, the exported rows must follow the
    POSTERIOR -- not the proposal the sampler drew from."""
    theta, lnL, logw = make_cloud()
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=logw,
                      neff=np.inf, rng=np.random.default_rng(3))
    got, hdr = read_export(opts)

    # unchanged file format: no weight column, same header as before
    assert got.shape[1] == 5
    assert hdr.split()[1:] == ["right_ascension", "declination", "inclination",
                               "psi", "loglikelihood"]

    for j in range(NDIM):
        col = got[:, COL_OF_THETA[j]]
        # MC tolerance from the fair draw's own ESS, generously padded
        tol_mean = 6.0 * SD_POST / np.sqrt(1.0 / np.sum(
            (np.exp(logw - logw.max()) / np.exp(logw - logw.max()).sum()) ** 2))
        assert abs(col.mean() - MEAN_POST[j]) < max(tol_mean, 0.02), (
            "coord %d exported mean %.4f, posterior mean %.4f "
            "(proposal mean 0.0) -- the cloud was NOT reweighted"
            % (j, col.mean(), MEAN_POST[j]))
        assert 0.85 < col.std() / SD_POST < 1.15, (
            "coord %d exported sd %.4f vs posterior sd %.4f (proposal sd %.4f)"
            % (j, col.std(), SD_POST, S_Q))


def test_unreweighted_export_would_fail_the_above(tmp_path):
    """Mutation control, scored the SAME way: passing logw=None (the pre-fix
    behaviour -- write the raw sampler cloud) must NOT satisfy the assertions
    above.  If this ever passes, the test above proves nothing."""
    theta, lnL, _ = make_cloud()
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=None)
    got, _ = read_export(opts)
    ok = all(abs(got[:, COL_OF_THETA[j]].mean() - MEAN_POST[j]) < 0.02
             and 0.85 < got[:, COL_OF_THETA[j]].std() / SD_POST < 1.15
             for j in range(NDIM))
    assert not ok, ("the RAW proposal cloud passed the fair-draw assertions; "
                    "the test target is too weak to detect a dropped reweight")


def test_uniform_weights_are_a_no_op(tmp_path):
    """A converged, untempered MCMC chain already targets the posterior and
    reports uniform post_weight; the export must then be the chain itself, not a
    resampled (duplicate-ridden) version of it."""
    rng = np.random.default_rng(5)
    theta = MEAN_POST[None, :] + rng.standard_normal((5000, NDIM)) * SD_POST
    lnL = _logN(theta, MU_L, S_L)
    logw = np.log(np.ones(len(theta)) / len(theta))
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=logw,
                      neff=np.inf, rng=np.random.default_rng(3))
    got, _ = read_export(opts)
    assert len(got) == len(theta)
    assert len(np.unique(got[:, 0])) == len(theta), \
        "uniform weights triggered a resample (duplicates in the export)"


def test_fairdraw_count_options_are_live(tmp_path):
    """--n-fairdraw-extrinsic-samples / --fairdraw-extrinsic-output-n-max must
    CHANGE the number of exported rows (a parsed-and-logged knob is not a live
    one)."""
    theta, lnL, logw = make_cloud(n=20000)
    for kw, want in ((dict(n_fairdraw_extrinsic_samples=137), 137),
                     (dict(fairdraw_extrinsic_output=True,
                           fairdraw_extrinsic_output_n_max=9), 9)):
        opts = fake_opts(tmp_path / str(want), **kw)
        os.makedirs(str(tmp_path / str(want)), exist_ok=True)
        drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=logw,
                          neff=np.inf, rng=np.random.default_rng(3))
        got, _ = read_export(opts)
        assert len(got) == want, "requested %d fair draws, got %d" % (want, len(got))


def test_ess_clamp_prevents_manufactured_draws(tmp_path):
    """A low-ESS cloud must not be resampled up to its original length: that
    writes a file that looks like N independent draws but holds ~ESS distinct
    points.  ILE clamps at 1.5*ESS; so must this.  (Observed for real: --mode
    laplace-is on a BNS gave ESS=97 out of 200000 samples.)"""
    rng = np.random.default_rng(2)
    n = 50000
    theta = rng.standard_normal((n, NDIM)) * 4.0
    # a deliberately terrible proposal -> a handful of points carry the weight
    logw = _logN(theta, MU_L, 0.05)
    w = np.exp(logw - logw.max()); w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    assert ess < n / 100.0, "the test cloud is not actually low-ESS (ESS=%.1f)" % ess
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, _logN(theta, MU_L, 0.05), with_distance=False,
                      logw=logw, neff=np.nan, rng=np.random.default_rng(3))
    got, _ = read_export(opts)
    assert len(got) <= np.ceil(1.5 * ess), (
        "exported %d rows from an ESS=%.1f cloud (cap %d)"
        % (len(got), ess, int(np.ceil(1.5 * ess))))
    assert len(got) < n


def test_tempered_flowmc_weights_are_not_uniform():
    """The flowMC modes sample L^inv_T; post_weight = L^(1-inv_T) is the
    correction.  Guard the invariant that a tempered run yields NON-uniform
    weights, so silently dropping them is a real (not cosmetic) error."""
    lnL = np.linspace(-30.0, 30.0, 1000)
    for inv_T, uniform_expected in ((1.0, True), (0.8, False), (0.5, False)):
        lw = (1.0 - inv_T) * lnL
        w = np.exp(lw - lw.max()); w /= w.sum()
        assert np.allclose(w, w[0]) is uniform_expected, \
            "inv_T=%g: uniformity of post_weight is %s" % (inv_T, not uniform_expected)
        idx, note = drv.fairdraw_indices(np.log(w), 500, np.random.default_rng(1))
        assert (idx is None) is uniform_expected
        assert not note.startswith("FAILED"), note


def test_exported_lnL_belongs_to_its_own_row(tmp_path):
    """The resample must carry theta and lnL through the SAME index.  A version
    that permutes one relative to the other writes a loglikelihood that does not
    belong to the parameters on its row -- invisible to every distributional
    check, because both marginals stay correct."""
    theta, lnL, logw = make_cloud(n=120000)
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=logw,
                      neff=np.inf, rng=np.random.default_rng(3))
    got, _ = read_export(opts)
    th_out = np.empty((len(got), NDIM))
    for j in range(NDIM):
        th_out[:, j] = got[:, COL_OF_THETA[j]]
    recomputed = _logN(th_out, MU_L, S_L)          # lnL implied by the row's theta
    err = np.abs(recomputed - got[:, -1])
    assert np.max(err) < 1e-9, (
        "exported lnL does not match the exported theta on the same row "
        "(max |dlnL| = %.4g, mean %.4g) -- theta/lnL pairing was broken"
        % (np.max(err), np.mean(err)))


def test_degenerate_weights_fail_loudly_not_silently(tmp_path):
    """Weights that cannot be normalized must be reported as FAILED, not
    silently returned as 'uniform, nothing to do' -- otherwise the raw,
    unreweighted cloud is written under a header promising a fair draw."""
    rng = np.random.default_rng(4)
    theta = rng.standard_normal((5000, NDIM)) * 3.0
    for bad, why in ((np.full(5000, -np.inf), "all -inf"),
                     (np.where(np.arange(5000) == 0, 0.0, -np.inf), "one finite")):
        idx, note = drv.fairdraw_indices(bad, 100, rng)
        assert idx is None
        assert note.startswith("FAILED"), "%s reported as %r" % (why, note)
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, np.full(5000, -np.inf), with_distance=False,
                      logw=np.full(5000, -np.inf), neff=np.nan,
                      rng=np.random.default_rng(5))
    with open(opts.output_file + "_0_samples.dat") as fh:
        head = [fh.readline() for _ in range(2)]
    assert "FAILED" in head[1], "failure not recorded in the export header: %r" % head[1]


def test_weights_are_stabilized_at_realistic_lnL(tmp_path):
    """Real extrinsic lnL runs to several hundred (this BNS peaked at 266; ILE
    routinely sees >1000).  exp(logw) without subtracting the max overflows to
    inf there, which the fail-open path used to swallow.  Exercise the range the
    driver actually operates in, not the O(1) range of a toy target."""
    rng = np.random.default_rng(6)
    n = 20000
    theta = rng.standard_normal((n, NDIM)) * 2.0
    logw = 800.0 + _logN(theta, MU_L, 1.5)     # ~ +800, well past exp() overflow
    assert logw.max() > 700.0
    idx, note = drv.fairdraw_indices(logw, 2000, rng)
    assert idx is not None, "fair draw refused at realistic lnL: %s" % note
    assert not note.startswith("FAILED"), note
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, logw, with_distance=False, logw=logw,
                      neff=np.inf, rng=np.random.default_rng(7))
    got, _ = read_export(opts)
    assert len(got) > 1 and np.isfinite(got).all()
    assert len(np.unique(got[:, 0])) > 1, "export collapsed to a single point"


def test_export_header_records_ess_and_mode(tmp_path):
    """The artifact must be self-describing: export ESS was previously recorded
    nowhere, so a 200000-sample file with ESS 97 looked like any other."""
    theta, lnL, logw = make_cloud(n=120000)
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=logw,
                      neff=np.inf, rng=np.random.default_rng(3))
    with open(opts.output_file + "_0_samples.dat") as fh:
        cols_line, prov_line = fh.readline(), fh.readline()
    assert cols_line.split()[1] == "right_ascension", "column line moved: %r" % cols_line
    assert "ESS=" in prov_line and "mode=" in prov_line, prov_line


def test_export_rng_is_independent_of_the_science_stream(tmp_path):
    """--save-samples is an OUTPUT flag and must not move any number.  The
    export draw is keyed by (seed, out_index), so it is reproducible no matter
    what else has consumed randomness, and it cannot perturb the sampler's
    stream."""
    theta, lnL, logw = make_cloud(n=20000)
    outs = []
    for burn in (0, 10000):
        shared = np.random.default_rng(fake_opts(tmp_path).seed)
        shared.standard_normal(burn)                    # unrelated consumption
        d = tmp_path / ("burn%d" % burn)
        os.makedirs(str(d), exist_ok=True)
        opts = fake_opts(d)
        drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=logw,
                          neff=np.inf)                  # rng=None -> derived
        outs.append(read_export(opts)[0])
    assert np.array_equal(outs[0], outs[1]), \
        "export depends on how much the shared RNG was consumed"
    # and different events must not reuse the same draw
    o2 = fake_opts(tmp_path / "ev1"); os.makedirs(str(tmp_path / "ev1"), exist_ok=True)
    drv.write_samples(o2, 1, theta, lnL, with_distance=False, logw=logw, neff=np.inf)
    assert not np.array_equal(read_export(o2, 1)[0], outs[0])


def test_mode_sets_exclude_non_importance_weights():
    """multistart-nuts / nuts-phimarg report post_weight as a per-chain Laplace
    MODE-EVIDENCE weight (samplers.py: np.full(n_per[k], mass[k]/n_per[k])), not
    L*p/p_s.  They must not be fair-drawn against it."""
    assert "multistart-nuts" not in drv._TEMPERED_MODES
    assert "nuts-phimarg" not in drv._TEMPERED_MODES
    assert "multistart-nuts" not in drv._FAIRDRAW_MODES
    assert "nuts-phimarg" not in drv._FAIRDRAW_MODES
    for m in ("flowmc", "flowmc-phimarg", "flowmc-phipsimarg", "flowmc-dpsimarg"):
        assert m in drv._TEMPERED_MODES and m in drv._FAIRDRAW_MODES
    for m in ("prior-mc", "laplace-is"):
        assert m in drv._FAIRDRAW_MODES and m not in drv._TEMPERED_MODES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
