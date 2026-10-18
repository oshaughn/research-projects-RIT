#!/usr/bin/env python
"""
test_integrate_log_neff_negative_lnZ.py

integrate_log() in mcsamplerGPU, mcsamplerPortfolio and mcsamplerNFlow reports
n_eff = sum(w)/max(w) as exp(ln mean(w) + ln N - maxval), with maxval the running
max of the LOG weight.  maxval used to start at 0, so whenever every weight was < 1
the reported n_eff was low by a factor 1/max(w) -- it could fall below 1, and the
`eff_samp < neff` loop ran to nmax.

The integrand here is a normalized 4-D Gaussian times exp(LNZ_SHIFT), so every log
weight is below LNZ_SHIFT < 0.  The reported n_eff must equal sum(w)/max(w)
recomputed from _rvs.  The +500 arm is the production regime (lnL ~ 1e2-1e3), where
the old and new initializers agree.
"""
from __future__ import print_function

import sys
import types

import numpy as np
import pytest

D = 4
LO, HI = -5.0, 5.0
SIG = 0.7
LNZ_SHIFT = -20.0


def _lnG(*args, **kwargs):
    # mcsamplerGPU/NFlow pass the parameters by name, the portfolio positionally
    r2 = sum(np.asarray(x, dtype=float) ** 2 for x in list(args) + list(kwargs.values()))
    return -r2 / (2 * SIG ** 2) - 0.5 * D * np.log(2 * np.pi * SIG ** 2)


def _add_params(s):
    for i in range(D):
        s.add_parameter("x%d" % i, pdf=np.vectorize(lambda x: 1.0 / (HI - LO)),
                        prior_pdf=np.vectorize(lambda x: 1.0 / (HI - LO)),
                        left_limit=LO, right_limit=HI, adaptive_sampling=True)
    return s


def _kish_from_rvs(s):
    """sum(w)/max(w) over every stored sample, w = integrand * prior / p_s."""
    lw = (np.asarray(s._rvs["log_integrand"]) + np.asarray(s._rvs["log_joint_prior"])
          - np.asarray(s._rvs["log_joint_s_prior"]))
    return float(np.sum(np.exp(lw - np.max(lw)))), float(np.max(lw)), len(lw)


def _run(s, shift, **kw):
    np.random.seed(7)
    args = dict(neff=50, n=2000, nmax=40000, save_intg=True, verbose=False)
    args.update(kw)
    out = s.integrate_log(lambda *a, **k: _lnG(*a, **k) + shift, *["x%d" % i for i in range(D)], **args)
    return float(np.asarray(out[2]))


def _gpu():
    from RIFT.integrators import mcsamplerGPU
    return _add_params(mcsamplerGPU.MCSampler())


def _portfolio():
    import RIFT.integrators.mcsamplerPortfolio as mcsP
    import RIFT.integrators.mcsamplerAdaptiveVolume as mcsAV
    import RIFT.integrators.mcsamplerEnsemble as mcsGMM
    s = _add_params(mcsP.MCSampler(portfolio=[mcsAV, mcsGMM]))
    s.setup()
    return s


_NF_STUBS = ["torch", "torch.optim", "torch.optim.lr_scheduler", "torch.utils", "torch.utils.data",
             "nflows", "nflows.flows", "nflows.flows.base", "nflows.utils",
             "nflows.distributions", "nflows.distributions.normal",
             "nflows.transforms", "nflows.transforms.normalization", "nflows.transforms.base",
             "nflows.transforms.autoregressive", "nflows.transforms.permutations",
             "nflows.transforms.standard", "nflows.transforms.lu",
             "nflows.nn", "nflows.nn.nets"]


def _stub(name):
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: type(str(attr), (object,), {
        "__init__": lambda self, *a, **k: None, "__call__": lambda self, *a, **k: None})
    return mod


@pytest.fixture
def nflow_module():
    """mcsamplerNFlow with real torch/nflows if present, else import-time stubs.  The
    untrained flow (n_adapt=0) samples uniformly and never calls into torch."""
    try:
        import torch  # noqa: F401
        import nflows  # noqa: F401
        names = []
    except Exception:
        names = _NF_STUBS
    saved = {n: sys.modules.get(n) for n in names + ["RIFT.integrators.mcsamplerNFlow"]}
    for n in names:
        sys.modules[n] = _stub(n)
    sys.modules.pop("RIFT.integrators.mcsamplerNFlow", None)
    try:
        import RIFT.integrators.mcsamplerNFlow as NF
        yield NF
    finally:
        for n, prev in saved.items():
            if prev is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = prev


def _check(s, eff, shift):
    kish, lwmax, n = _kish_from_rvs(s)
    assert (lwmax < 0) == (shift < 0), "premise: max log weight {} for shift {}".format(lwmax, shift)
    assert eff >= 1.0, "n_eff={} < 1 is impossible for sum(w)/max(w) (max lnw={})".format(eff, lwmax)
    assert np.isclose(eff, kish, rtol=1e-8), \
        "reported n_eff {} != sum(w)/max(w)={} from {} stored samples".format(eff, kish, n)


@pytest.mark.parametrize("shift", [LNZ_SHIFT, 500.0])
def test_gpu_neff_is_kish_ratio(shift):
    s = _gpu()
    _check(s, _run(s, shift, n_adapt=0), shift)


@pytest.mark.parametrize("shift", [LNZ_SHIFT, 500.0])
def test_portfolio_neff_is_kish_ratio(shift):
    s = _portfolio()
    _check(s, _run(s, shift), shift)


@pytest.mark.parametrize("shift", [LNZ_SHIFT, 500.0])
def test_nflow_loop_neff_stops_on_true_kish_ratio(nflow_module, shift):
    """NFlow recomputes its RETURNED n_eff from _rvs, so only the loop's stopping value
    carried the defect.  With a reachable target the run must stop at the first chunk
    whose true n_eff passes it, not at nmax."""
    s = _add_params(nflow_module.MCSampler(n_chunk=2000))
    eff = _run(s, shift, n_adapt=0, neff=10, nmax=40000)
    _check(s, eff, shift)
    assert len(s._rvs["log_integrand"]) < 40000, "ran to nmax although sum(w)/max(w)={}".format(eff)


def _run_f(s, f, **kw):
    np.random.seed(7)
    args = dict(neff=10, n=2000, nmax=40000, save_intg=True, verbose=False)
    args.update(kw)
    out = s.integrate_log(f, *["x%d" % i for i in range(D)], **args)
    return float(np.asarray(out[2]))


def test_gpu_nan_still_raises():
    """With maxval starting at -inf, max(-inf, nan) is -inf.  A NaN in the first chunk must
    still reach the NanOrInf check, not be reported as a finite or zero n_eff."""
    from RIFT.integrators import mcsamplerGPU
    s = _gpu()

    def f(*a, **k):
        out = _lnG(*a, **k) + LNZ_SHIFT
        out[0] = np.nan
        return out
    with pytest.raises(mcsamplerGPU.NanOrInf):
        _run_f(s, f, n_adapt=0)
