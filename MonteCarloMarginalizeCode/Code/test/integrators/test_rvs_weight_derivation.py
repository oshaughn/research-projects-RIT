#!/usr/bin/env python
"""`log_weights` in _rvs does not mean the same thing in every sampler.

- `mcsamplerPortfolio` stores the true importance weight:  lnL + ln p - ln p_s
- `mcsamplerGPU`       stores the ADAPTATION weight:       e*lnL + ln p - ln p_s

with `e` = the adapt-weight-exponent.  `e` is NOT 1 in production -- helper_LDG_Events sets it
from the SNR (helper_LDG_Events.py:1472/1477) -- and `--no-adapt` drives it to 0, which removes
the likelihood from the column entirely.  Any consumer that preferred the cached column therefore
reweighted its output by L^(e-1) whenever the GPU/AC sampler was in use.

The `.dgrid` and calibration-posterior exporters both did exactly that.  These tests pin the
canonical derivation they now share.

Run:  python test_rvs_weight_derivation.py
"""
import os
import re
import types

import numpy


def _driver_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "bin",
                                         "integrate_likelihood_extrinsic_batchmode"))


def _driver_source():
    return open(_driver_path()).read()


def _load():
    src = _driver_source()
    mod = types.ModuleType("drv")
    mod.numpy = numpy
    for fn in ("_rvs_lnL_convention", "ln_weights_from_rvs"):
        m = re.search(r"^def %s\(.*?(?=\n\ndef |\n\nclass )" % fn, src, re.S | re.M)
        assert m, "%s not found in the driver" % fn
        exec(compile(m.group(0), "<drv>", "exec"), mod.__dict__)
    return mod


DRV = _load()


def _rec(n=500, e=0.1, seed=0):
    """A GPU/AC-style record: canonical components PLUS a tempered `log_weights` cache."""
    rng = numpy.random.RandomState(seed)
    lnL = rng.normal(20.0, 3.0, size=n)
    lp = rng.normal(-1.0, 0.1, size=n)
    lps = rng.normal(-0.5, 0.1, size=n)
    return dict(log_integrand=lnL, log_joint_prior=lp, log_joint_s_prior=lps,
                log_weights=e * lnL + lp - lps), lnL, lp, lps


def test_derives_from_components_not_the_cache():
    rec, lnL, lp, lps = _rec(e=0.1)
    got = DRV.ln_weights_from_rvs(rec)
    assert numpy.allclose(got, lnL + lp - lps), "did not derive the true importance weight"
    # and the cache it ignored was materially different -- otherwise this proves nothing
    assert not numpy.allclose(got, rec['log_weights']), \
        "the tempered cache happened to equal the truth; test does not demonstrate the hazard"
    spread = numpy.ptp(got) / max(numpy.ptp(rec['log_weights']), 1e-12)
    assert spread > 5, "expected the tempered cache to be much flatter, got ratio {:.2f}".format(spread)


def test_no_adapt_cache_loses_the_likelihood_entirely():
    """--no-adapt sets the exponent to 0, so the cached column carries no likelihood at all."""
    rec, lnL, lp, lps = _rec(e=0.0)
    assert numpy.allclose(rec['log_weights'], lp - lps)          # likelihood absent, by construction
    got = DRV.ln_weights_from_rvs(rec)
    assert numpy.allclose(got, lnL + lp - lps)


def test_portfolio_style_cache_agrees_but_is_still_not_read():
    """The portfolio's cache IS the importance weight, so agreement here is expected -- the point
    is that correctness no longer depends on which sampler wrote the record."""
    rec, lnL, lp, lps = _rec(e=1.0)
    assert numpy.allclose(DRV.ln_weights_from_rvs(rec), rec['log_weights'])


def test_linear_form_and_out_of_support_rows():
    """mcsamplerEnsemble stores raw (non-log) columns; zero integrand/prior rows must go to -inf,
    not to a NaN that would poison a sum."""
    ig = numpy.array([2.0, 0.0, 3.0])
    jp = numpy.array([1.0, 1.0, 0.0])
    js = numpy.array([1.0, 1.0, 1.0])
    got = DRV.ln_weights_from_rvs(dict(integrand=ig, joint_prior=jp, joint_s_prior=js))
    assert numpy.isneginf(got[1]) and numpy.isneginf(got[2])
    assert numpy.isclose(got[0], numpy.log(2.0))
    assert not numpy.any(numpy.isnan(got))


def test_log_convention_does_not_take_log_of_lnL():
    """mcsamplerEnsemble under return_lnI stores lnL in the SAME 'integrand' field.

    Logging it again turns tens of nats into log(tens): the weight vector goes nearly flat, the
    likelihood drops out, and the exported posterior is prior-dominated.
    """
    rng = numpy.random.RandomState(5)
    lnL = rng.normal(18.0, 4.0, size=400)
    jp = numpy.exp(rng.normal(-1.0, 0.1, size=400))
    js = numpy.exp(rng.normal(-0.5, 0.1, size=400))
    rec = dict(integrand=lnL, joint_prior=jp, joint_s_prior=js)

    got = DRV.ln_weights_from_rvs(rec, use_lnL=True)
    assert numpy.allclose(got, lnL + numpy.log(jp) - numpy.log(js))

    # and the old unconditional reading really was catastrophic, not a rounding detail
    old = DRV.ln_weights_from_rvs(rec, use_lnL=False)
    assert numpy.ptp(got) > 10 * numpy.ptp(old), (
        "expected log(lnL) to collapse the dynamic range; got ptp {:.2f} vs {:.2f}".format(
            numpy.ptp(got), numpy.ptp(old)))


def test_log_convention_retains_rows_with_nonpositive_lnL():
    """`ig > 0` is the right cut for a raw likelihood -- non-positive means REJECTED.

    Against a log it means an ordinary low-likelihood point, so the old cut silently deleted every
    sample with lnL <= 0.  Those rows must survive with finite weights.
    """
    lnL = numpy.array([-12.0, -0.5, 0.0, 3.0, 25.0])
    jp = numpy.array([1.0, 2.0, 0.5, 1.0, 1.0])
    js = numpy.array([1.0, 1.0, 1.0, 4.0, 1.0])
    rec = dict(integrand=lnL, joint_prior=jp, joint_s_prior=js)

    got = DRV.ln_weights_from_rvs(rec, use_lnL=True)
    assert numpy.all(numpy.isfinite(got)), "lnL <= 0 rows were discarded: {}".format(got)
    assert numpy.allclose(got, lnL + numpy.log(jp) - numpy.log(js))

    # the pre-fix reading dropped 3 of the 5 rows entirely
    old = DRV.ln_weights_from_rvs(rec, use_lnL=False)
    assert numpy.sum(numpy.isneginf(old)) == 3


def test_log_convention_still_sends_out_of_support_rows_to_minus_inf():
    """A zero prior or zero sampling prior is out of support in EITHER convention, and a NaN in a
    log-sum-exp poisons the whole record."""
    lnL = numpy.array([5.0, -3.0, 7.0, numpy.nan])
    jp = numpy.array([1.0, 0.0, 1.0, 1.0])
    js = numpy.array([1.0, 1.0, 0.0, 1.0])
    got = DRV.ln_weights_from_rvs(dict(integrand=lnL, joint_prior=jp, joint_s_prior=js),
                                  use_lnL=True)
    assert numpy.isclose(got[0], 5.0)
    assert numpy.isneginf(got[1]) and numpy.isneginf(got[2]) and numpy.isneginf(got[3])
    assert not numpy.any(numpy.isnan(got))


def test_linear_storage_is_unchanged_under_an_accepted_lnL_method():
    """THE P2 REGRESSION GUARD.

    --internal-use-lnL is accepted for every method in ok_lnL_methods, which includes
    'adaptive_cartesian'.  That sampler (RIFT/integrators/mcsampler.py) has NO use_lnL /
    return_lnI handling at all and always stores LINEAR L.  So the predicate cannot be the CLI
    option: keying off it would compute L + ln p - ln p_s on a record where log(L) + ln p - ln p_s
    is right -- a new wrong answer where the pre-fix code was correct.
    """
    rng = numpy.random.RandomState(6)
    L = numpy.exp(rng.normal(18.0, 4.0, size=300))
    jp = numpy.exp(rng.normal(-1.0, 0.1, size=300))
    js = numpy.exp(rng.normal(-0.5, 0.1, size=300))
    rec = dict(integrand=L, joint_prior=jp, joint_s_prior=js)

    got = DRV.ln_weights_from_rvs(rec, use_lnL=False)
    assert numpy.allclose(got, numpy.log(L) + numpy.log(jp) - numpy.log(js))
    # the default must remain the linear reading, so an un-updated caller cannot silently flip
    assert numpy.allclose(DRV.ln_weights_from_rvs(rec), got)


def test_convention_predicate_is_the_stored_convention_not_the_cli_option():
    """Pins WHERE the predicate comes from, in the driver source.

    `opts.internal_use_lnL` says what the user asked for; `return_lnI` in pinned_params is what was
    actually handed to the sampler, and only the latter identifies the stored representation.  The
    combination that breaks the first reading is real: 'adaptive_cartesian' is in ok_lnL_methods
    but gets no use_lnL/return_lnI wiring, so its _rvs stays linear.
    """
    src = _driver_source()
    assert re.search(r"^rvs_integrand_is_lnL\s*=\s*bool\(\s*pinned_params\.get\(\s*[\"']return_lnI[\"']",
                     src, re.M), "the convention variable is no longer derived from return_lnI"
    assert re.search(r"^ok_lnL_methods\s*=.*adaptive_cartesian", src, re.M), \
        "ok_lnL_methods no longer accepts a linear-storage method; re-check this guard"
    # no call site may key the interpretation off the CLI option
    for m in re.finditer(r"ln_weights_from_rvs\((?:[^()]|\([^()]*\))*\)", src):
        assert "opts.internal_use_lnL" not in m.group(0), \
            "call site keys off the CLI option, not the stored convention: {}".format(m.group(0))


def test_convention_default_is_linear_when_the_driver_globals_are_absent():
    """_rvs_lnL_convention is read through globals() on purpose: several callers wrap it in a bare
    `except Exception: return None`, so a NameError would become a silent None rather than a
    diagnosable failure."""
    assert DRV._rvs_lnL_convention() is False
    assert DRV._rvs_lnL_convention(True) is True
    DRV.rvs_integrand_is_lnL = True
    try:
        assert DRV._rvs_lnL_convention() is True
        assert DRV._rvs_lnL_convention(False) is False      # explicit still wins
    finally:
        del DRV.rvs_integrand_is_lnL


def test_missing_components_raise_rather_than_guess():
    """A record with ONLY the ambiguous cache must fail loudly: an explicit error beats a
    plausible wrong number in a science output."""
    try:
        DRV.ln_weights_from_rvs(dict(log_weights=numpy.zeros(3)))
    except Exception as e:
        assert "cannot build importance weights" in str(e), str(e)
        return
    raise AssertionError("derived weights from a cache-only record instead of raising")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("_rvs weight derivation is canonical")
