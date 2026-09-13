"""Regression tests for memory-bounded weighting in streamed V contraction."""

import numpy as np
import pytest

from RIFT.likelihood import gpu_precompute as gpu

from conftest import to_host


def _problem(seed=6061):
    rng = np.random.default_rng(seed)
    nfreq, nmodes, nresponse = 38, 2, 3
    delta_f = 0.125
    delta_t = 1.0 / (nfreq * delta_f)
    base = (rng.normal(size=(nmodes, nfreq))
            + 1j * rng.normal(size=(nmodes, nfreq)))
    base_conj = (rng.normal(size=(nmodes, nfreq))
                 + 1j * rng.normal(size=(nmodes, nfreq)))
    response = (rng.normal(size=(nresponse, nfreq))
                + 1j * rng.normal(size=(nresponse, nfreq)))
    # Deliberately nonsymmetric compound rows and a partial final a-block.
    a_list = [(0, 0, -2), (2, 1, 1), (1, 0, 3),
              (0, 2, -1), (2, 1, 2)]
    weights = rng.uniform(0.01, 2.0, size=nfreq)
    weights[[0, 3, nfreq // 2 + 1, nfreq - 1]] = 0.0
    epoch = -13.37
    f_sidereal = 0.019
    return (base, base_conj, response, a_list, weights,
            delta_f, delta_t, epoch, f_sidereal)


@pytest.mark.parametrize("a_block,frequency_chunk", [(1, 1), (2, 7), (4, 64)])
def test_streamed_weighted_left_matches_dense_complex_oracle(
        backend, a_block, frequency_chunk):
    (base, base_conj, response, a_list, weights,
     delta_f, delta_t, epoch, f_sidereal) = _problem()
    primary = gpu.build_compound_basis(
        backend.asarray(base), backend.asarray(response), a_list,
        delta_f, delta_t, epoch, f_sidereal=f_sidereal,
        backend=backend, fft_batch=2)
    conjugate = gpu.build_compound_basis(
        backend.asarray(base_conj), backend.asarray(response), a_list,
        delta_f, delta_t, epoch, f_sidereal=f_sidereal,
        backend=backend, fft_batch=3)

    a_count, mode_count, nfreq = primary.shape
    primary_flat = to_host(primary).reshape(a_count * mode_count, nfreq)
    conjugate_flat = to_host(conjugate).reshape(a_count * mode_count, nfreq)
    expected = ((np.conj(conjugate_flat) * weights[None, :])
                @ primary_flat.T) * (2.0 * delta_f)
    expected = expected.reshape(
        a_count, mode_count, a_count, mode_count).transpose(0, 2, 1, 3)

    got = gpu.streamed_v_matrix(
        backend.asarray(base_conj), backend.asarray(response), a_list, primary,
        backend.asarray(weights), delta_f, delta_t, epoch,
        f_sidereal=f_sidereal, backend=backend, a_block=a_block,
        fft_batch=min(2, a_block), frequency_chunk=frequency_chunk,
        return_device=True)
    np.testing.assert_allclose(
        to_host(got), expected, rtol=3e-11, atol=3e-10)


class _TrackingArray(np.ndarray):
    """NumPy view that records every broadcast multiply by a row weight."""

    def __new__(cls, value, tracker):
        obj = np.asarray(value).view(cls)
        obj.tracker = tracker
        return obj

    def __array_finalize__(self, source):
        self.tracker = getattr(source, "tracker", None)

    def __mul__(self, other):
        other_shape = np.shape(other)
        if len(other_shape) == 2 and other_shape[0] == 1:
            self.tracker.append((tuple(self.shape), tuple(other_shape)))
        return _TrackingArray(
            np.asarray(self) * np.asarray(other), self.tracker)

    def __rmul__(self, other):
        return self.__mul__(other)


class _TrackingNumpy:
    """Small backend shim used only to audit temporary working-set shapes."""

    __name__ = "tracking_numpy"
    complex128 = np.complex128

    def __init__(self):
        self.weighted_multiplies = []

    def asarray(self, value, dtype=None):
        if isinstance(value, _TrackingArray) and dtype is None:
            return value
        return _TrackingArray(
            np.asarray(value, dtype=dtype), self.weighted_multiplies)

    def zeros(self, shape, dtype=None):
        return _TrackingArray(
            np.zeros(shape, dtype=dtype), self.weighted_multiplies)

    def conj(self, value):
        return _TrackingArray(
            np.conj(np.asarray(value)), self.weighted_multiplies)


def test_each_v_block_weights_only_its_small_left_working_set(monkeypatch):
    """Never allocate a weighted (A*M, frequency_chunk) primary temporary."""
    rng = np.random.default_rng(9902)
    a_count, mode_count, nfreq, a_block = 5, 2, 37, 2
    a_list = [(i, 0, 0) for i in range(a_count)]
    primary = (rng.normal(size=(a_count, mode_count, nfreq))
               + 1j * rng.normal(size=(a_count, mode_count, nfreq)))
    conjugate = (rng.normal(size=(a_count, mode_count, nfreq))
                 + 1j * rng.normal(size=(a_count, mode_count, nfreq)))
    backend = _TrackingNumpy()

    positions = {a: i for i, a in enumerate(a_list)}

    def fake_build(unused_base, unused_response, block, *unused_args, **unused_kwargs):
        rows = [positions[tuple(a)] for a in block]
        return backend.asarray(conjugate[rows])

    monkeypatch.setattr(gpu, "build_compound_basis", fake_build)
    got = gpu.streamed_v_matrix(
        np.zeros((mode_count, nfreq), dtype=complex),
        np.zeros((a_count, nfreq), dtype=complex), a_list, primary,
        np.linspace(0.0, 1.0, nfreq), 0.25, 1.0 / (nfreq * 0.25), 0.0,
        backend=backend, a_block=a_block, fft_batch=1, frequency_chunk=11,
        return_device=True)

    assert got.shape == (a_count, a_count, mode_count, mode_count)
    weighted_shapes = backend.weighted_multiplies
    assert weighted_shapes
    assert max(shape[0][0] for shape in weighted_shapes) == a_block * mode_count
    assert all(shape[0][0] < a_count * mode_count for shape in weighted_shapes)
    # Across all blocks/chunks, every streamed-left element is weighted once.
    assert sum(left[0] * left[1] for left, unused_weight in weighted_shapes) == \
        a_count * mode_count * nfreq
