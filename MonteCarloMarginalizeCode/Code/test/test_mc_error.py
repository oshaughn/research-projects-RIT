import math

import pytest

from RIFT.misc.mc_error import relative_mc_error


def test_relative_mc_error_linear_space():
    assert relative_mc_error(4.0, 0.04) == pytest.approx(0.05)


def test_relative_mc_error_log_space():
    log_z = -1.9780425516285014
    sigma = 0.0020773022158222486
    log_variance = math.log(sigma**2) + 2 * log_z
    assert relative_mc_error(log_z, log_variance, log_space=True) == pytest.approx(sigma)


def test_relative_mc_error_rejects_negative_linear_variance():
    with pytest.raises(ValueError, match="non-negative"):
        relative_mc_error(1.0, -1.0)


def test_relative_mc_error_propagates_non_finite_inputs():
    assert math.isnan(relative_mc_error(math.nan, 1.0))
    assert relative_mc_error(0.0, -math.inf, log_space=True) == 0.0
    assert relative_mc_error(1.0, math.inf) == math.inf
