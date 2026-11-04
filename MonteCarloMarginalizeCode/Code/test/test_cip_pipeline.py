"""Tests for RIFT.misc.cip_pipeline (posterior export draw + CIP arg-list rewrite).

Imported by file path so the tests run without RIFT's heavy import chain
(lal/glue), which the pure-numpy module under test does not need.
"""
import importlib.util
import os

import numpy as np

_MOD_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                         "RIFT", "misc", "cip_pipeline.py")
_spec = importlib.util.spec_from_file_location("cip_pipeline", _MOD_PATH)
cip_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cip_pipeline)

systematic_resample = cip_pipeline.systematic_resample
unique_draw_bound = cip_pipeline.unique_draw_bound
flag_final_group_unique = cip_pipeline.flag_final_group_unique
FLAG = cip_pipeline.POSTERIOR_UNIQUE_FLAG


def test_unique_draw_bound_is_sum_over_max():
    w = np.array([10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # sum 19, max 10
    assert unique_draw_bound(w) == 1
    assert unique_draw_bound(np.ones(500)) == 500


def test_systematic_resample_expected_counts_at_any_n():
    # Expected count of index i must be N*w_i/sum(w) even when N >> n_eff --
    # this is exactly the property the old weighted choice(replace=False) lacked.
    np.random.seed(4)
    w = np.array([10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    n = 200000
    counts = np.bincount(systematic_resample(w, n), minlength=len(w))
    expected = n * w / np.sum(w)
    # systematic resampling: each count is within 1 of expectation per stratum pass;
    # across strata the deviation stays O(1) regardless of n
    assert np.all(np.abs(counts - expected) <= np.ceil(expected * 0.01) + 1)


def test_systematic_resample_counts_never_exceed_ceiling():
    np.random.seed(5)
    w = np.random.uniform(size=300) ** 4
    n = 5000
    counts = np.bincount(systematic_resample(w, n), minlength=len(w))
    assert np.all(counts <= np.ceil(n * w / np.sum(w)))


def test_systematic_resample_unique_at_the_bound():
    np.random.seed(6)
    w = np.ones(1000)
    w[0] = 3.0
    n = unique_draw_bound(w)
    indx = systematic_resample(w, n)
    assert len(np.unique(indx)) == len(indx)


def test_systematic_resample_output_is_shuffled():
    # The export truncates the head of the draw, so draw order must carry no signal.
    np.random.seed(7)
    indx = systematic_resample(np.ones(1000), 500)
    assert np.any(np.diff(indx) < 0)


def test_flag_lands_on_final_group_only():
    configured = flag_final_group_unique([
        "2 --fit-method gp --parameter mc\n",
        "3 --fit-method rf --parameter mc\n",
    ])
    assert configured == [
        "2 --fit-method gp --parameter mc",
        "3 --fit-method rf --parameter mc {}".format(FLAG),
    ]


def test_flag_rides_into_terminal_convergence_group():
    configured = flag_final_group_unique([
        "2 --fit-method gp",
        "Z --fit-method rf",
    ])
    assert configured[0] == "2 --fit-method gp"
    assert configured[-1] == "Z --fit-method rf {}".format(FLAG)


def test_final_gaussian_group_is_left_untouched():
    # The G executable has a strict parser without the flag; flagging it kills the job.
    lines = ["2 --fit-method gp", "G3 --fit-method quadratic"]
    assert flag_final_group_unique(lines) == lines


def test_rewrite_is_idempotent():
    once = flag_final_group_unique(["1 --fit-method rf"])
    assert flag_final_group_unique(once) == once


def test_blank_lines_dropped_and_empty_input_ok():
    assert flag_final_group_unique([]) == []
    assert flag_final_group_unique(["\n", "1 --fit-method rf\n", "   \n"]) == [
        "1 --fit-method rf {}".format(FLAG)]


def test_extended_precision_weights_beyond_float64_range():
    # RIFT builds export weights as longdouble on x86_64; finite values above
    # float64's range must not overflow to inf inside the helpers.
    import pytest
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no extra range")
    big = np.longdouble(10.0) ** 400
    w = np.full(10, big, dtype=np.longdouble)
    w[0] *= 10.0  # sum/max = 19/10 -> bound 1
    assert unique_draw_bound(w) == 1
    assert unique_draw_bound(np.full(93, big)) == 93  # exact bound, beyond-DBL_MAX case
    np.random.seed(8)
    n = 100000
    counts = np.bincount(systematic_resample(w, n), minlength=len(w))
    assert counts.sum() == n
    assert abs(counts[0] - n * 10.0 / 19.0) < n * 0.01


def test_invalid_weights_are_rejected():
    import pytest
    for bad in ([], [0.0, 0.0], [1.0, -1.0], [1.0, np.inf], [1.0, np.nan]):
        with pytest.raises(ValueError):
            unique_draw_bound(np.array(bad, dtype=float))


def test_unique_draw_bound_is_exact_for_equal_weights():
    # floor(1/max(p)) in float64 gives 92 for 93 equal weights (reciprocal
    # roundoff); the bound must be computed from the scaled sum instead.
    for n in (93, 3, 1000):
        assert unique_draw_bound(np.ones(n)) == n
