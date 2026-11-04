"""Import smoke tests for dormant LISA support modules.

These tests deliberately avoid waveform generation and data-file access.  The
goal is to keep the parachuted LISA response code import-safe for CI while the
larger likelihood and pipeline integration lands in later stages.
"""

import numpy as np


def test_lisa_response_import_and_pure_helper():
    import RIFT.LISA.response.LISA_response as lisa_response

    idx = lisa_response.get_closest_index(np.array([0.0, 2.0, 5.0]), 1.7)
    assert idx == 1


def test_lisa_factored_likelihood_import():
    import RIFT.likelihood.factored_likelihood_LISA as lisa_likelihood

    assert hasattr(lisa_likelihood, "PrecomputeAlignedSpinLISA")
    assert hasattr(lisa_likelihood, "FactoredLogLikelihoodAlignedSpinLISA")

