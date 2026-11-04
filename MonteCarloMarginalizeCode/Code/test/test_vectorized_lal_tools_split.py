"""The source-only / per-detector split of the vectorized LAL tools is bitwise exact.

`DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop` used to rebuild the detector
response basis and the source propagation direction once per detector, although
neither depends on the detector.  Those are now built once and handed to a
per-detector half.  The split is only worth having if it changes nothing, so pin
that with exact equality rather than a tolerance.

WHAT THE REFERENCE IS, AND WHY IT IS NOT THE WRAPPER.  An earlier version of this
file compared `ComputeDetAMResponse(...)` against
`ComputeDetAMResponsePrecomputed(SourcePolarizationBasis(...))`.  After the split the
wrapper IS that composition, so the comparison was tautological -- it passed with a
sign flipped in the source-only half, with the response matrix doubled in the
per-detector half, and with the speed of light wrong by 0.1%.  The references below
are instead the PRE-SPLIT bodies, frozen here verbatim, so the test compares the
refactor against what it replaced rather than against itself.
"""
import numpy as np
import pytest

import lalsimulation as lalsim

from RIFT.likelihood.vectorized_lal_tools import (
    ComputeDetAMResponse,
    ComputeDetAMResponsePrecomputed,
    SourcePolarizationBasis,
    SourcePropagationDirection,
    TimeDelayFromEarthCenter,
    TimeDelayFromEarthCenterPrecomputed,
)

# Three real interferometer geometries, so the test would catch an axis or
# transpose error that a symmetric toy matrix would hide.
DETECTORS = ["H1", "L1", "V1"]


def _frozen_detector_response(R, ra, dec, psi, gmst):
    """The body of ComputeDetAMResponse as it stood BEFORE the split (verbatim)."""
    X = np.empty(ra.shape + (3,), dtype=np.float64)
    Y = np.empty(ra.shape + (3,), dtype=np.float64)
    gha = gmst - ra
    cos_gha, sin_gha = np.cos(gha), np.sin(gha)
    cos_dec, sin_dec = np.cos(dec), np.sin(dec)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    X[..., 0] = -cos_psi*sin_gha - sin_psi*cos_gha*sin_dec
    X[..., 1] = -cos_psi*cos_gha + sin_psi*sin_gha*sin_dec
    X[..., 2] = sin_psi*cos_dec
    Y[..., 0] = sin_psi*sin_gha - cos_psi*cos_gha*sin_dec
    Y[..., 1] = sin_psi*cos_gha + cos_psi*sin_gha*sin_dec
    Y[..., 2] = cos_psi*cos_dec
    F_plus = (X*np.inner(X, R) - Y*np.inner(Y, R)).sum(axis=-1)
    F_cross = (X*np.inner(Y, R) + Y*np.inner(X, R)).sum(axis=-1)
    return F_plus + 1.0j*F_cross


def _frozen_time_delay(loc, ra, dec, gmst):
    """The body of TimeDelayFromEarthCenter as it stood BEFORE the split (verbatim)."""
    negative_speed_of_light = np.asarray(-299792458.0)
    cos_dec = np.cos(dec)
    gha = gmst - ra
    ehat = np.empty(ra.shape + (3,), dtype=np.float64)
    ehat[..., 0] = cos_dec * np.cos(gha)
    ehat[..., 1] = -cos_dec * np.sin(gha)
    ehat[..., 2] = np.sin(dec)
    neg_separation = np.inner(loc, ehat)
    return np.divide(neg_separation, negative_speed_of_light, out=neg_separation)


def _samples(n=257, seed=20260905):
    rng = np.random.RandomState(seed)
    return (
        rng.uniform(0.0, 2.0 * np.pi, n),          # right ascension
        np.arcsin(rng.uniform(-1.0, 1.0, n)),      # declination
        rng.uniform(0.0, np.pi, n),                # polarization
    )


@pytest.mark.parametrize("det", DETECTORS)
def test_detector_response_split_matches_frozen_pre_split_body(det):
    ra, dec, psi = _samples()
    gmst = 4.371829
    R = np.asarray(lalsim.DetectorPrefixToLALDetector(det).response)

    want = _frozen_detector_response(R, ra, dec, psi, gmst)
    X, Y = SourcePolarizationBasis(ra, dec, psi, gmst, xpy=np)
    got_split = ComputeDetAMResponsePrecomputed(R, X, Y, xpy=np)
    got_wrapper = ComputeDetAMResponse(R, ra, dec, psi, gmst, xpy=np)

    assert np.array_equal(got_split, want), det
    assert np.array_equal(got_wrapper, want), det


@pytest.mark.parametrize("det", DETECTORS)
def test_time_delay_split_matches_frozen_pre_split_body(det):
    ra, dec, _ = _samples()
    gmst = 4.371829
    loc = np.asarray(lalsim.DetectorPrefixToLALDetector(det).location)

    want = _frozen_time_delay(loc, ra, dec, gmst)
    ehat = SourcePropagationDirection(ra, dec, gmst, xpy=np)
    got_split = TimeDelayFromEarthCenterPrecomputed(loc, ehat, xpy=np)
    got_wrapper = TimeDelayFromEarthCenter(loc, ra, dec, gmst, xpy=np)

    assert np.array_equal(got_split, want), det
    assert np.array_equal(got_wrapper, want), det


def test_shared_inputs_are_not_mutated_by_the_per_detector_halves():
    """The whole point of the split is that one prologue serves every detector.

    If a per-detector half wrote into ehat_src, X or Y -- the time-delay half divides
    in place, into the result of `inner`, which is a fresh array, but that is a
    one-character edit away from being wrong -- the second detector would silently be
    computed from corrupted geometry.
    """
    ra, dec, psi = _samples(n=64)
    gmst = 1.25
    ehat = SourcePropagationDirection(ra, dec, gmst, xpy=np)
    X, Y = SourcePolarizationBasis(ra, dec, psi, gmst, xpy=np)
    ehat0, X0, Y0 = ehat.copy(), X.copy(), Y.copy()

    for det in DETECTORS:
        d = lalsim.DetectorPrefixToLALDetector(det)
        TimeDelayFromEarthCenterPrecomputed(np.asarray(d.location), ehat, xpy=np)
        ComputeDetAMResponsePrecomputed(np.asarray(d.response), X, Y, xpy=np)

    assert np.array_equal(ehat, ehat0)
    assert np.array_equal(X, X0)
    assert np.array_equal(Y, Y0)
