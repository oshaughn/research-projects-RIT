"""The source-only / per-detector split of the vectorized LAL tools is bitwise exact.

`DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop` used to rebuild the detector
response basis and the source propagation direction once per detector, although
neither depends on the detector.  Those are now built once and handed to a
per-detector half.  The split is only worth having if it changes nothing, so pin
that with exact equality rather than a tolerance: the per-detector functions must
perform the same contractions, in the same order, on the same inputs.
"""
import numpy as np

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
import lalsimulation as lalsim

DETECTORS = ["H1", "L1", "V1"]


def _samples(n=257, seed=20260905):
    rng = np.random.RandomState(seed)
    return (
        rng.uniform(0.0, 2.0 * np.pi, n),          # right ascension
        np.arcsin(rng.uniform(-1.0, 1.0, n)),      # declination
        rng.uniform(0.0, np.pi, n),                # polarization
    )


def test_detector_response_split_is_bitwise_identical():
    ra, dec, psi = _samples()
    gmst = 4.371829

    X, Y = SourcePolarizationBasis(ra, dec, psi, gmst, xpy=np)
    for det in DETECTORS:
        response = np.asarray(
            lalsim.DetectorPrefixToLALDetector(det).response)
        combined = ComputeDetAMResponse(response, ra, dec, psi, gmst, xpy=np)
        split = ComputeDetAMResponsePrecomputed(response, X, Y, xpy=np)
        assert np.array_equal(combined, split), det


def test_time_delay_split_is_bitwise_identical():
    ra, dec, _ = _samples()
    gmst = 4.371829

    ehat = SourcePropagationDirection(ra, dec, gmst, xpy=np)
    for det in DETECTORS:
        location = np.asarray(
            lalsim.DetectorPrefixToLALDetector(det).location)
        combined = TimeDelayFromEarthCenter(location, ra, dec, gmst, xpy=np)
        split = TimeDelayFromEarthCenterPrecomputed(location, ehat, xpy=np)
        assert np.array_equal(combined, split), det


def test_time_delay_is_not_secretly_shared_state():
    """The per-detector half must not consume or mutate the shared ehat_src.

    It divides in place into the result of `inner`, which is a fresh array; if that
    ever became an in-place write into ehat_src, the second detector would silently
    get a delay computed from a scaled direction vector.
    """
    ra, dec, _ = _samples(n=64)
    gmst = 1.25
    ehat = SourcePropagationDirection(ra, dec, gmst, xpy=np)
    before = ehat.copy()
    for det in DETECTORS:
        location = np.asarray(
            lalsim.DetectorPrefixToLALDetector(det).location)
        TimeDelayFromEarthCenterPrecomputed(location, ehat, xpy=np)
    assert np.array_equal(ehat, before)
