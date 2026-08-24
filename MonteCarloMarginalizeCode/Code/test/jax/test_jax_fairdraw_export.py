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
import ast
import inspect
import textwrap
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
                      neff=np.inf)
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
                      neff=np.inf)
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
                          neff=np.inf)
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
                      logw=logw, neff=np.nan)
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
        # n_out < n, so BOTH paths return indices -- but for different reasons,
        # and the distinction is what the export header must record:
        #   uniform     -> subsample WITHOUT replacement (no duplicates)
        #   non-uniform -> resample WITH replacement against w
        idx, note = drv.fairdraw_indices(np.log(w), np.random.default_rng(1))
        assert not note.startswith("FAILED"), note
        # fairdraw_indices does REWEIGHTING only: uniform weights are a genuine
        # no-op there, and the export count is the caller's job.
        assert (idx is None) is uniform_expected
        assert ("weights uniform" in note) is uniform_expected, note
        assert "ESS=" in note, note
        if not uniform_expected:
            assert len(np.unique(idx)) < len(idx), "resampling must be WITH replacement"


def test_exported_lnL_belongs_to_its_own_row(tmp_path):
    """The resample must carry theta and lnL through the SAME index.  A version
    that permutes one relative to the other writes a loglikelihood that does not
    belong to the parameters on its row -- invisible to every distributional
    check, because both marginals stay correct."""
    theta, lnL, logw = make_cloud(n=120000)
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=logw,
                      neff=np.inf)
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


def test_exported_lnL_stays_paired_through_the_count_subsample(tmp_path):
    """The pairing test above uses default opts, so it never enters the count
    path -- and the count subsample is a SECOND place theta and lnL are indexed
    together.  Re-check it with a count requested."""
    theta, lnL, logw = make_cloud(n=120000)
    opts = fake_opts(tmp_path, n_fairdraw_extrinsic_samples=311)
    drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=logw,
                      neff=np.inf)
    got, _ = read_export(opts)
    assert len(got) == 311
    th_out = np.empty((len(got), NDIM))
    for j in range(NDIM):
        th_out[:, j] = got[:, COL_OF_THETA[j]]
    err = np.abs(_logN(th_out, MU_L, S_L) - got[:, -1])
    assert np.max(err) < 1e-9, (
        "count subsample broke the theta/lnL pairing (max |dlnL| = %.4g)"
        % np.max(err))


def test_count_flags_are_inert_for_modes_reported_as_ignoring_them(tmp_path):
    """Report and behaviour must agree.  check_critical_and_report gates these
    flags on _FAIRDRAW_MODES; applying them anyway under a NUTS mode printed
    "IGNORED" and then wrote 5 rows instead of 300.  --fairdraw-extrinsic-output
    is in ILE_extr.sub, so that is a real production line losing 60x of its
    export under a banner saying the flag did nothing."""
    rng = np.random.default_rng(21)
    theta = MEAN_POST[None, :] + rng.standard_normal((300, NDIM)) * SD_POST
    lnL = _logN(theta, MU_L, S_L)
    for mode, expect in (("nuts", 300), ("multistart-nuts", 300),
                         ("nuts-phimarg", 300), ("flowmc-phimarg", 5)):
        d = tmp_path / mode; os.makedirs(str(d), exist_ok=True)
        opts = fake_opts(d, mode=mode, fairdraw_extrinsic_output=True,
                         fairdraw_extrinsic_output_n_max=5)
        drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=None,
                          neff=np.inf)
        got, _ = read_export(opts)
        assert len(got) == expect, (
            "--mode %s: wrote %d rows, expected %d (%s)"
            % (mode, len(got), expect,
               "count must be inert where it is reported ignored"
               if expect == 300 else "count must act where it is reported implemented"))


def test_provenance_n_out_matches_the_file(tmp_path):
    """n_out was computed before non-finite lnL were dropped, so the header
    over-stated the file exactly when the likelihood misbehaved."""
    rng = np.random.default_rng(22)
    n = 1000
    theta = MEAN_POST[None, :] + rng.standard_normal((n, NDIM)) * SD_POST
    lnL = _logN(theta, MU_L, S_L)
    lnL[rng.choice(n, size=37, replace=False)] = np.nan
    for kw in ({}, dict(n_fairdraw_extrinsic_samples=137)):
        d = tmp_path / str(len(kw)); os.makedirs(str(d), exist_ok=True)
        opts = fake_opts(d, **kw)
        drv.write_samples(opts, 0, theta, lnL, with_distance=False,
                          logw=np.log(np.ones(n) / n), neff=np.inf)
        got, _ = read_export(opts)
        with open(opts.output_file + "_0_samples.dat") as fh:
            fh.readline(); prov = fh.readline()
        n_out = int(prov.split("n_out=")[1].split()[0])
        assert n_out == len(got), (
            "header says n_out=%d, file holds %d rows" % (n_out, len(got)))
        assert np.isfinite(got).all()


def test_every_path_reports_ess_and_n_in(tmp_path):
    """F-E in full: the logw=None path reported neither ESS= nor n_in=, so the
    self-describing header was blank on one of the three paths that write.  (The
    FAILED path writes nothing at all -- see
    test_failed_fairdraw_writes_no_samples_file.)"""
    rng = np.random.default_rng(23)
    theta = MEAN_POST[None, :] + rng.standard_normal((500, NDIM)) * SD_POST
    lnL = _logN(theta, MU_L, S_L)
    cases = {"none": None,
             "uniform": np.log(np.ones(500) / 500),
             "weighted": _logN(theta, MU_L, 1.2)}
    for name, lw in cases.items():
        d = tmp_path / name; os.makedirs(str(d), exist_ok=True)
        opts = fake_opts(d)
        drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=lw,
                          neff=np.inf)
        with open(opts.output_file + "_0_samples.dat") as fh:
            fh.readline(); prov = fh.readline()
        for field in ("ESS=", "n_in=", "n_out="):
            assert field in prov, "%s path header lacks %s: %r" % (name, field, prov)
    # and the uniform path must not pass its ROW COUNT off as an ESS
    opts = fake_opts(tmp_path / "uniform")
    with open(opts.output_file + "_0_samples.dat") as fh:
        fh.readline(); prov = fh.readline()
    assert "ESS=n/a" in prov, "uniform path reports a fabricated ESS: %r" % prov


def test_degenerate_weights_fail_loudly_not_silently():
    """Weights that cannot be normalized must be reported as FAILED, not
    silently returned as 'uniform, nothing to do' -- otherwise the raw,
    unreweighted cloud is written under a header promising a fair draw."""
    rng = np.random.default_rng(4)
    for bad, why in ((np.full(5000, -np.inf), "all -inf"),
                     (np.where(np.arange(5000) == 0, 0.0, -np.inf), "one finite")):
        idx, note = drv.fairdraw_indices(bad, rng)
        assert idx is None
        assert note.startswith("FAILED"), "%s reported as %r" % (why, note)


def test_failed_fairdraw_writes_no_samples_file(tmp_path):
    """A FAILED fair draw must produce NO export.  Writing the raw cloud with
    'FAILED' in the provenance line was still a non-posterior cloud sitting under
    the standard weightless product name: consumers read `*_samples.dat` rows as
    equal-weight posterior draws and are not obliged to parse the second header
    line, so the contract was violated exactly on the collapsed integrations
    where proposal and posterior differ most."""
    rng = np.random.default_rng(4)
    theta = rng.standard_normal((5000, NDIM)) * 3.0
    opts = fake_opts(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        drv.write_samples(opts, 0, theta, np.full(5000, -np.inf),
                          with_distance=False, logw=np.full(5000, -np.inf),
                          neff=np.nan)
    assert "fair draw failed" in str(exc.value)
    assert not os.path.exists(opts.output_file + "_0_samples.dat"), \
        "a non-posterior cloud was exported after the fair draw failed"


def test_failed_fairdraw_removes_a_stale_export(tmp_path):
    """Refusing to write is not enough on a re-run: a samples file left from an
    earlier run sits at exactly the path the pipeline reads for THIS one, so the
    refusal must also clear it rather than silently endorsing stale rows."""
    rng = np.random.default_rng(9)
    theta = rng.standard_normal((5000, NDIM)) * 3.0
    opts = fake_opts(tmp_path)
    stale = opts.output_file + "_0_samples.dat"
    with open(stale, "w") as fh:
        fh.write("# right_ascension declination inclination psi loglikelihood\n"
                 "# mode=flowmc-phimarg fairdraw: reweighted ESS=900.0 n_in=1 n_out=1\n"
                 "0.1 0.2 0.3 0.4 -5.0\n")
    with pytest.raises(RuntimeError):
        drv.write_samples(opts, 0, theta, np.full(5000, -np.inf),
                          with_distance=False, logw=np.full(5000, -np.inf),
                          neff=np.nan)
    assert not os.path.exists(stale), \
        "the previous run's export survived a failed fair draw"


def test_failed_event_is_skippable_but_never_exported(tmp_path):
    """The refusal must be an ordinary Exception, so main's per-event handler
    (--soft-fail-event-range) can carry a batch past a collapsed event, and it
    must not disturb the other events' exports."""
    src = inspect.getsource(drv.main)
    assert "except Exception" in src and "soft_fail_event_range" in src, \
        "main lost the per-event guard that makes a refused export skippable"
    theta, lnL, logw = make_cloud(n=20000)
    bad = fake_opts(tmp_path / "bad"); os.makedirs(str(tmp_path / "bad"), exist_ok=True)
    with pytest.raises(RuntimeError):
        drv.write_samples(bad, 0, theta, lnL, with_distance=False,
                          logw=np.full(len(theta), -np.inf), neff=np.nan)
    good = fake_opts(tmp_path / "good"); os.makedirs(str(tmp_path / "good"), exist_ok=True)
    drv.write_samples(good, 1, theta, lnL, with_distance=False, logw=logw,
                      neff=np.inf)
    assert not os.path.exists(bad.output_file + "_0_samples.dat")
    assert len(read_export(good, 1)[0]) > 1


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
    idx, note = drv.fairdraw_indices(logw, rng)
    assert idx is not None, "fair draw refused at realistic lnL: %s" % note
    assert not note.startswith("FAILED"), note
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, logw, with_distance=False, logw=logw,
                      neff=np.inf)
    got, _ = read_export(opts)
    assert len(got) > 1 and np.isfinite(got).all()
    assert len(np.unique(got[:, 0])) > 1, "export collapsed to a single point"


def test_export_header_records_ess_and_mode(tmp_path):
    """The artifact must be self-describing: export ESS was previously recorded
    nowhere, so a 200000-sample file with ESS 97 looked like any other."""
    theta, lnL, logw = make_cloud(n=120000)
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, lnL, with_distance=False, logw=logw,
                      neff=np.inf)
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
                          neff=np.inf)   # export rng derived from (seed, out_index)
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


def _write_samples_call():
    """The ast.Call node for write_samples(...) inside analyze_one."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(drv.analyze_one)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "write_samples"]
    assert len(calls) == 1, "expected exactly one write_samples call, found %d" % len(calls)
    return calls[0]


def test_write_samples_never_receives_the_science_generator():
    """STRUCTURAL guard for F2, checked by AST rather than by substring.

    The export RNG is derived inside write_samples from (seed, out_index).  If a
    caller can hand it the generator that feeds the samplers, --save-samples --
    an OUTPUT flag -- moves the science again.

    A name-based guard is not enough: `write_samples(..., generator=rng)` defeats
    both a signature test that looks for the literal "rng" and a source check for
    the substring "rng=" (that text contains "=rng").  So instead: no argument
    expression of the call may be the bare Name `rng`, whatever keyword it wears,
    and the callee must not name a Generator-ish parameter at all."""
    call = _write_samples_call()
    args = list(call.args) + [k.value for k in call.keywords]
    for a in args:
        assert not (isinstance(a, ast.Name) and a.id == "rng"), (
            "analyze_one passes the shared `rng` to write_samples (as %s)"
            % (next((k.arg for k in call.keywords if k.value is a), "positional")))
        # `opts.rng`-style smuggling: any attribute access ending in .rng
        assert not (isinstance(a, ast.Attribute) and a.attr == "rng"), \
            "analyze_one smuggles an rng in via an attribute"
    params = list(inspect.signature(drv.write_samples).parameters)
    for bad in ("rng", "generator", "random_state", "prng", "bitgen"):
        assert bad not in params, (
            "write_samples grew a %r parameter -- a caller can now pass the "
            "science generator (%s)" % (bad, params))


def test_post_weight_is_gated_on_tempered_modes_at_the_call_site():
    """F1 lives at the CALL SITE, not in the frozensets.  Asserting the sets are
    correct cannot see the guard being deleted from analyze_one -- the same
    'test the helper, not the wiring' defect fixed for the export RNG.

    Require that the expression which reads res["post_weight"] is guarded by a
    test mentioning _TEMPERED_MODES."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(drv.analyze_one)))
    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        body = ast.dump(node.body) + ast.dump(node.orelse)
        if "post_weight" in body:
            guarded.append("_TEMPERED_MODES" in ast.dump(node.test))
    assert guarded, ("no conditional expression reads post_weight in analyze_one "
                     "-- the F1 guard was removed or restructured")
    assert all(guarded), ("post_weight is read without a _TEMPERED_MODES guard: "
                          "multistart-nuts / nuts-phimarg would be fair-drawn "
                          "against a per-chain Laplace mode-evidence weight")


def test_count_option_dests_are_stable():
    """fairdraw_size reads the options through getattr(..., None), which FAILS
    OPEN: rename an option's dest and the count silently stops being applied
    while everything still passes.  Drive the real parser and pin the dests."""
    optp = drv.build_parser()
    dests = {o.dest for o in optp._get_all_options() if o.dest}
    for d in ("n_fairdraw_extrinsic_samples", "fairdraw_extrinsic_output",
              "fairdraw_extrinsic_output_n_max", "mode", "seed", "save_samples"):
        assert d in dests, "option dest %r vanished -- fairdraw_size fails open" % d
    opts, _ = optp.parse_args(["--n-fairdraw-extrinsic-samples", "137"])
    assert opts.n_fairdraw_extrinsic_samples == 137
    opts2, _ = optp.parse_args(["--fairdraw-extrinsic-output"])
    assert opts2.fairdraw_extrinsic_output is True
    # unset -n-max must stay None so the ignored-option report does not claim
    # the user passed it; the ILE default of 5 is resolved downstream
    assert opts2.fairdraw_extrinsic_output_n_max is None
    assert drv.fairdraw_size(opts2, 10000, np.inf) == drv._FAIRDRAW_N_MAX_DEFAULT


def test_count_options_act_when_weights_are_uniform(tmp_path):
    """THE default configuration has uniform weights: --adapt-weight-exponent is
    1.0, so the flowMC modes report post_weight uniform and there is nothing to
    reweight.  The count options are a COUNT contract, not a reweight contract
    -- ILE applies the count regardless -- so they must still bound the export.
    Returning early on uniform weights made them a silent no-op in exactly the
    configuration people actually run."""
    rng = np.random.default_rng(11)
    theta = MEAN_POST[None, :] + rng.standard_normal((3200, NDIM)) * SD_POST
    lnL = _logN(theta, MU_L, S_L)
    uniform = np.log(np.ones(len(theta)) / len(theta))
    for kw, want in ((dict(n_fairdraw_extrinsic_samples=137), 137),
                     (dict(fairdraw_extrinsic_output=True,
                           fairdraw_extrinsic_output_n_max=5), 5)):
        d = tmp_path / str(want); os.makedirs(str(d), exist_ok=True)
        opts = fake_opts(d, **kw)
        drv.write_samples(opts, 0, theta, lnL, with_distance=False,
                          logw=uniform, neff=np.inf)
        got, _ = read_export(opts)
        assert len(got) == want, (
            "uniform weights: asked for %d rows, wrote %d -- the count contract "
            "was skipped" % (want, len(got)))
        # equal weights -> subsample WITHOUT replacement, so no duplicates
        assert len(np.unique(got[:, 0])) == want, "uniform subsample duplicated rows"


def test_uniform_export_header_still_reports_ess(tmp_path):
    """The self-describing header must not go blank on the uniform path -- that
    is where the flowMC modes live, and the README tells users to check the ESS
    before trusting a file."""
    rng = np.random.default_rng(12)
    theta = MEAN_POST[None, :] + rng.standard_normal((2000, NDIM)) * SD_POST
    lnL = _logN(theta, MU_L, S_L)
    opts = fake_opts(tmp_path)
    drv.write_samples(opts, 0, theta, lnL, with_distance=False,
                      logw=np.log(np.ones(len(theta)) / len(theta)), neff=np.inf)
    with open(opts.output_file + "_0_samples.dat") as fh:
        fh.readline(); prov = fh.readline()
    assert "ESS=" in prov and "n_in=" in prov and "n_out=" in prov, prov


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
