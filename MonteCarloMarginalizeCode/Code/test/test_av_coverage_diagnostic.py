#!/usr/bin/env python
"""
AV's final-grid coverage diagnostic (unseen_frac_final) must flag a live set that misses bins.

Two cosine axes carry a Gaussian of width SIG; three flat axes (RA, psi, phase) span their whole
range.  The live set leaves flat-axis bins empty and lnZ reads about 2 nats low (a known defect,
not fixed here).  The diagnostic must read above COVERAGE_UNSEEN_WARN there and below it with no
flat axes.  Measurements: RIFT_roboto_paper development/OPEN_av_412_narrow_flat_axes.md.
"""

import numpy as np

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV

BOUNDS = {"declination": (-1.0, 1.0), "inclination": (-1.0, 1.0),
          "right_ascension": (0.0, 2 * np.pi), "psi": (0.0, np.pi), "phi_orb": (0.0, 2 * np.pi)}
MU = {"declination": 0.2, "inclination": -0.3}
SIG = 0.001


def _unseen(names, seed):
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
                        enforce_bounds=True, verbose=False, dict_return=True)
    return r[3]["unseen_frac_final"]


def test_diagnostic_flags_narrow_peak_with_flat_axes():
    # measured 0.33-0.39 at these settings
    u = [_unseen(list(BOUNDS), seed) for seed in (11, 12)]
    assert min(u) > mcsamplerAV.COVERAGE_UNSEEN_WARN, u


def test_diagnostic_quiet_when_every_axis_is_resolved():
    u = _unseen(["declination", "inclination"], 11)
    assert u < mcsamplerAV.COVERAGE_UNSEEN_WARN, u
