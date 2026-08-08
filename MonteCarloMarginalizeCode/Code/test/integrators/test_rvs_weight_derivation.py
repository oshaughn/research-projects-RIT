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


def _load():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "..", "bin",
                                         "integrate_likelihood_extrinsic_batchmode"))
    src = open(path).read()
    mod = types.ModuleType("drv")
    mod.numpy = numpy
    m = re.search(r"^def ln_weights_from_rvs\(.*?(?=\n\ndef |\n\nclass )", src, re.S | re.M)
    assert m, "ln_weights_from_rvs not found in the driver"
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
