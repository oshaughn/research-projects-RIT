#!/usr/bin/env python
"""
fresh_sample_slices must integrate over the prior box only.

Without enforce_bounds, AV's edge bins overhang the box; like_at_pinned_d used to clip
out-of-box draws to the edge, so they carried real likelihood values and each slice's
lnZ came out high.  The bias grows with the number of AV cycles and is largest when an
Omega direction is unconstrained (sky azimuth, phase, polarization).  On S240426s it
moved the slice conditional's distance quantiles outward by 3% (before research-projects-RIT
#412) and 10% (after).  Measurements: RIFT_roboto_paper
analyses/distance_gp_interp/checkpoints/2026-10-16-bisect-412.md.
"""

import numpy as np
import pytest

from RIFT.misc.distance_slices import fresh_sample_slices
import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV

BOUNDS = {
    "right_ascension": (0.0, 2 * np.pi),
    "declination": (-1.0, 1.0),
    "inclination": (-1.0, 1.0),
    "psi": (0.0, np.pi),
    "phi_orb": (0.0, 2 * np.pi),
    "distance": (10.0, 4000.0),
}
SIG = 0.01   # narrow enough that AV contracts over several cycles
MU = {"declination": 0.2, "inclination": -0.3}


class _RefSampler(object):
    """The attributes fresh_sample_slices reads off ILE's extrinsic sampler."""

    def __init__(self):
        self.params_ordered = list(BOUNDS)
        self.llim = {p: lo for p, (lo, _) in BOUNDS.items()}
        self.rlim = {p: hi for p, (_, hi) in BOUNDS.items()}
        self.pdf, self.prior_pdf = {}, {}
        for p, (lo, hi) in BOUNDS.items():
            n = 1.0 / (hi - lo)
            # constant everywhere, like ILE's uniform priors: zero support outside the box
            # must come from fresh_sample_slices, not from the prior
            self.pdf[p] = (lambda x, _n=n: np.full(np.shape(x), _n))
            self.prior_pdf[p] = (lambda x, _n=n: np.full(np.shape(x), _n))


def likelihood(right_ascension, declination, inclination, psi, phi_orb, distance):
    """Flat in right_ascension, psi and phi_orb, Gaussian in the two cosine coordinates."""
    q = ((declination - MU["declination"]) ** 2 + (inclination - MU["inclination"]) ** 2) / SIG ** 2
    return -0.5 * q + 0.0 * (right_ascension + psi + phi_orb) + 400.0 / distance


def exact_lnZ(d):
    # prior density 1/2 on each cosine coordinate, Gaussians far from the edges; flat axes integrate to 1
    return 2 * np.log(np.sqrt(2 * np.pi) * SIG / 2.0) + 400.0 / d


def test_slice_lnZ_matches_closed_form_with_a_flat_direction():
    d_slices = np.array([200.0, 800.0, 3000.0])
    err = []
    for seed in (11, 12, 13):
        np.random.seed(seed)
        if mcsamplerAV.cupy_ok:
            mcsamplerAV.xpy_default.random.seed(seed)
        lnL, _, _, _ = fresh_sample_slices(_RefSampler(), likelihood, d_slices,
                                           n_max=400000, n_eff_target=30, n_chunk=10000)
        err.append(lnL - exact_lnZ(d_slices))
    err = np.array(err)
    assert abs(err.mean()) < 0.1, "slice lnZ - exact = {} (mean {:.3f})".format(err, err.mean())
    assert np.all(np.abs(err) < 0.35), "slice lnZ - exact = {}".format(err)


def test_draws_outside_the_box_get_minus_inf(monkeypatch):
    """The integrand itself must exclude out-of-box points, whatever AV's grid does."""
    seen = {}

    def fake_integrate_log(self, lnF, *params, **kw):
        n = 7
        x = {p: np.full(n, 0.5 * (BOUNDS[p][0] + BOUNDS[p][1])) for p in params}
        x["declination"][:3] = [1.2, -1.0 - 1e-6, 1.0]     # out, out, on the edge
        x["right_ascension"][3] = 2 * np.pi + 0.1           # out
        seen["lnL"] = np.asarray(lnF(**x))
        seen["enforce_bounds"] = kw.get("enforce_bounds")
        return 0.0, 0.0, 1.0, {}

    monkeypatch.setattr(mcsamplerAV.MCSampler, "integrate_log", fake_integrate_log)
    fresh_sample_slices(_RefSampler(), likelihood, [1000.0], n_max=10000, n_chunk=10000)
    lnL = seen["lnL"]
    assert np.all(np.isneginf(lnL[[0, 1, 3]])), lnL
    assert np.all(np.isfinite(lnL[[2, 4, 5, 6]])), lnL
    assert seen["enforce_bounds"] is True
