#!/usr/bin/env python
"""Regression test: the GMM member of a default-configured portfolio must
actually TRAIN.

Bug (2026-07-22, found by the PR#28 freeze-territory probe): portfolio setup()
forwarded kwargs that lack n_comp, mcsamplerEnsemble.setup() defaulted
n_comp=None, and update_sampling_prior silently no-opped for n_comp=None --
so in default wiring the GMM member never trained and every 'portfolio' was
effectively AV-only, with no error or warning.  n_comp=0 remains the
intentional off-switch and must stay off.

Run:  python test_portfolio_gmm_member_trains.py   (exit 0 = pass)
"""
from __future__ import print_function

import numpy as np
from scipy.stats import multivariate_normal

# CPU nodes with cupy installed but no GPU: work around the import-time
# cupy binding in gaussian_mixture_model (fixed separately); harmless once
# that fix lands.
import RIFT.integrators.gaussian_mixture_model as _gmmmod
if hasattr(_gmmmod, "_xpy_eigvals"):
    _gmmmod._xpy_eigvals = np.linalg.eigvalsh
if hasattr(_gmmmod, "_xpy_eig"):
    _gmmmod._xpy_eig = np.linalg.eig

from RIFT.integrators import (mcsamplerAdaptiveVolume, mcsamplerEnsemble,
                              mcsamplerPortfolio)

rng = np.random.RandomState(31415)
NDIM = 2
LLIM, RLIM = -5.0, 5.0
MU = rng.uniform(-1.5, 1.5, NDIM)
COV = 0.3 * np.identity(NDIM)
_mvn = multivariate_normal(MU, COV)


def ln_f(*cols):
    X = np.array([np.asarray(c, dtype=float) for c in cols]).T
    return 100.0 + np.log(_mvn.pdf(np.atleast_2d(X)) + 1e-300)


def _gmm_trained(gmm_sampler):
    integ = getattr(gmm_sampler, "integrator", None)
    if integ is None:
        return False
    return any(m is not None for m in integ.gmm_dict.values())


def _build_portfolio(portfolio_args=None):
    try:
        av = mcsamplerAdaptiveVolume.MCSampler(n_chunk=5000)
    except TypeError:
        av = mcsamplerAdaptiveVolume.MCSampler()
    gmm = mcsamplerEnsemble.MCSampler()
    port = mcsamplerPortfolio.MCSampler(portfolio=[av, gmm])
    params = ["x{}".format(i) for i in range(NDIM)]
    for p in params:
        pdf = np.vectorize(lambda x: 1.0 / (RLIM - LLIM))
        port.add_parameter(p, pdf, prior_pdf=pdf, left_limit=LLIM,
                           right_limit=RLIM, adaptive_sampling=True)
    if portfolio_args is None:
        port.setup()
    else:
        port.setup(portfolio_args=portfolio_args)
    return port, gmm, params


def test_default_portfolio_gmm_member_trains():
    # NO explicit n_comp anywhere: this is exactly the default production wiring.
    port, gmm, params = _build_portfolio()
    port.integrate_log(ln_f, *params, no_protect_names=True, nmax=60000,
                       n=5000, neff=50000, n_adapt=100, tempering_exp=0.1,
                       save_intg=True, verbose=False)
    assert _gmm_trained(gmm), (
        "portfolio GMM member never trained (n_comp default regression): "
        "gmm_dict models are all None")
    print("PASS: portfolio GMM member trained under default configuration")


def test_n_comp_zero_remains_off_switch():
    port, gmm, params = _build_portfolio(portfolio_args=[{}, dict(n_comp=0)])
    port.integrate_log(ln_f, *params, no_protect_names=True, nmax=30000,
                       n=5000, neff=50000, n_adapt=100, tempering_exp=0.1,
                       save_intg=True, verbose=False)
    assert not _gmm_trained(gmm), "n_comp=0 must remain the off-switch"
    print("PASS: n_comp=0 off-switch still honored")


if __name__ == "__main__":
    test_default_portfolio_gmm_member_trains()
    test_n_comp_zero_remains_off_switch()
