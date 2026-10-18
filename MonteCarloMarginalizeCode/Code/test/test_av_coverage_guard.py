#!/usr/bin/env python
"""
AV integrate_log must cover the live region when a narrow peak sits inside one bin.

Two cosine axes carry a Gaussian of width SIG; three flat axes (RA, psi, phase) span their
whole range.  The narrow axes stay unresolved, the live set stays near nsel, and the flat axes
hold more bins than it has points.  Without the coverage guard, bins left empty drop out of
later grids, log_retained_density over-counts where points remain, and lnZ reads about 2 nats
low.  Backport of rift_O4d #458.  Measurements: RIFT_roboto_paper
development/OPEN_av_412_narrow_flat_axes.md.
"""

import numpy as np

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV

BOUNDS = {"declination": (-1.0, 1.0), "inclination": (-1.0, 1.0),
          "right_ascension": (0.0, 2 * np.pi), "psi": (0.0, np.pi), "phi_orb": (0.0, 2 * np.pi)}
MU = {"declination": 0.2, "inclination": -0.3}
SIG = 0.001
LNZ_EXACT = 2 * np.log(np.sqrt(2 * np.pi) * SIG / 2)


def _run(names, seed, **kw):
    np.random.seed(seed)
    mcsamplerAV.xpy_default.random.seed(seed)   # draws use the active backend (cupy on a GPU host)
    s = mcsamplerAV.MCSampler()
    for p in names:
        lo, hi = BOUNDS[p]
        f = lambda x, n=1 / (hi - lo): np.full(np.shape(x), n)
        s.add_parameter(p, pdf=f, prior_pdf=f, left_limit=lo, right_limit=hi,
                        adaptive_sampling=True)

    def lnF(**x):
        return -0.5 * ((x["declination"] - MU["declination"]) ** 2
                       + (x["inclination"] - MU["inclination"]) ** 2) / SIG ** 2
    r = s.integrate_log(lnF, *names, nmax=4000000, neff=300, n=10000, tempering_exp=0.1,
                        enforce_bounds=True, verbose=False, dict_return=True, **kw)
    return float(r[0]) - LNZ_EXACT, r[3]


def test_narrow_peak_with_flat_axes():
    # unguarded: -2.18 +/- 0.07 over 9 seeds; guarded: -0.009 +/- 0.009 over 20
    errs = [_run(list(BOUNDS), seed)[0] for seed in (11, 12, 13)]
    assert np.all(np.abs(errs) < 0.3), errs
    assert abs(np.mean(errs)) < 0.15, errs


def test_guard_idle_when_every_axis_is_resolved():
    names = ["declination", "inclination"]
    err_on, d_on = _run(names, 11)
    err_off, d_off = _run(names, 11, coverage_unseen_tol=0)
    assert d_on["n_coarsen_cycles"] == 0
    assert err_on == err_off
    assert abs(err_on) < 0.3
