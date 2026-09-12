"""Device-resident precompute for the rotating finite-response likelihood.

This module is intentionally optional.  It keeps the long two-sided spectra on a
CuPy device, builds all ``(b,p,n,l,m)`` elementary templates with batched FFTs,
and reduces Q/U/V there.  Only the short Q time window and the small U/V matrices
need to return to the host for the conventional ILE interface.

The low-level array API also has a NumPy backend.  That is useful for convention
and failure-mode tests on machines without CUDA; production callers should leave
``backend=None`` so CuPy is required.
"""
from __future__ import division, print_function

from dataclasses import dataclass, field
import hashlib
import math
import threading
import time

import numpy as np


def _resolve_backend(backend=None):
    if backend is not None:
        return backend
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on CUDA installation
        raise RuntimeError(
            "GPU precompute requested but CuPy could not be imported") from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy reports no CUDA devices")
    except Exception as exc:  # pragma: no cover - depends on CUDA installation
        raise RuntimeError(
            "GPU precompute requested but no usable CUDA device is available") from exc
    return cp


def _is_numpy(xp):
    return xp is np or getattr(xp, "__name__", "") == "numpy"


def _to_host(value, xp):
    return np.asarray(value) if _is_numpy(xp) else xp.asnumpy(value)


def _device_asarray(value, xp, dtype=None):
    """Convert arrays without routing a JAX device buffer through the host."""
    if _is_numpy(xp):
        return np.asarray(value, dtype=dtype)
    module = type(value).__module__.split(".")[0]
    # NumPy also exposes __dlpack__, but its capsule is a CPU device and cannot
    # be imported by CuPy.  DLPack is specifically the JAX GPU handoff here;
    # ordinary host arrays take the explicit H2D path below.
    if module in ("jax", "jaxlib"):
        try:
            converted = xp.from_dlpack(value)
        except (AttributeError, TypeError):  # older CuPy/JAX DLPack API
            try:
                import jax.dlpack
                converted = xp.fromDlpack(jax.dlpack.to_dlpack(value))
            except Exception as exc:
                raise RuntimeError("cannot transfer waveform device array to CuPy via DLPack") from exc
        return converted.astype(dtype, copy=False) if dtype is not None else converted
    return xp.asarray(value, dtype=dtype)


def _content_digest(array):
    """A mutation-safe cache digest (deliberately hashes every input byte)."""
    a = np.ascontiguousarray(np.asarray(array))
    h = hashlib.blake2b(digest_size=20)
    h.update(str(a.shape).encode("ascii"))
    h.update(a.dtype.str.encode("ascii"))
    h.update(memoryview(a).cast("B"))
    return h.digest()


@dataclass
class GPUPrecomputeContext:
    """Reusable data/PSD device cache for multiple intrinsic points in one worker.

    Cache keys include a full content digest, so an in-place mutation cannot silently
    reuse stale device data.  ``clear`` may be used between events to release memory.
    """
    backend: object = None
    _arrays: dict = field(default_factory=dict)
    _role_keys: dict = field(default_factory=dict)
    _lock: object = field(default_factory=threading.RLock)
    uploads: int = 0
    upload_bytes: int = 0
    cache_hits: int = 0

    def __post_init__(self):
        self.backend = _resolve_backend(self.backend)
        self._device_id = (None if _is_numpy(self.backend) else
                           int(self.backend.cuda.runtime.getDevice()))

    def check_device(self):
        """Do not reuse a worker's cached buffers on another CUDA device."""
        if self._device_id is not None and int(
                self.backend.cuda.runtime.getDevice()) != self._device_id:
            raise RuntimeError("GPU precompute context belongs to CUDA device %d; "
                               "select that device or use its default context" %
                               self._device_id)

    def array(self, role, host_array, dtype=None):
        self.check_device()
        a = np.asarray(host_array, dtype=dtype)
        key = (str(role), _content_digest(a), a.dtype.str, tuple(a.shape))
        with self._lock:
            cached = self._arrays.get(key)
            if cached is None:
                previous = self._role_keys.get(str(role))
                if previous is not None and previous != key:
                    self._arrays.pop(previous, None)
                cached = (self.backend.array(a, copy=True) if _is_numpy(self.backend)
                          else self.backend.asarray(a))
                self._arrays[key] = cached
                self._role_keys[str(role)] = key
                self.uploads += 1
                self.upload_bytes += int(a.nbytes)
            else:
                self.cache_hits += 1
            return cached

    def clear(self):
        self.check_device()
        with self._lock:
            self._arrays.clear()
            self._role_keys.clear()
        if not _is_numpy(self.backend):  # pragma: no cover - CUDA only
            self.backend.get_default_memory_pool().free_all_blocks()

    def stats(self):
        return dict(uploads=int(self.uploads), upload_bytes=int(self.upload_bytes),
                    cache_hits=int(self.cache_hits), retained_arrays=len(self._arrays))


_DEFAULT_CONTEXTS = {}
_DEFAULT_CONTEXT_LOCK = threading.Lock()


def default_context(backend=None):
    xp = _resolve_backend(backend)
    key = (id(xp), None if _is_numpy(xp) else int(xp.cuda.runtime.getDevice()))
    with _DEFAULT_CONTEXT_LOCK:
        if key not in _DEFAULT_CONTEXTS:
            _DEFAULT_CONTEXTS[key] = GPUPrecomputeContext(xp)
        return _DEFAULT_CONTEXTS[key]


def lal_frequency_axis(n, delta_f, xp=np):
    """RIFT/LAL two-sided order: +Nyquist, ..., 0, ..., -Nyquist+df."""
    if int(n) != n or n < 2 or n % 2:
        raise ValueError("a two-sided LAL frequency grid must have positive even length")
    if not np.isfinite(delta_f) or delta_f <= 0:
        raise ValueError("delta_f must be finite and positive")
    return float(delta_f) * (float(n) / 2.0 - xp.arange(int(n)))


def _lal_reverse(spectrum, xp):
    """LAL COMPLEX16 frequency-to-time transform using an FFT backend."""
    n = spectrum.shape[-1]
    alternating = 1 - 2 * (xp.arange(n) & 1)
    return xp.fft.ifft(spectrum, axis=-1) * alternating


def _lal_forward(series, xp):
    """LAL COMPLEX16 time-to-frequency transform using an FFT backend."""
    n = series.shape[-1]
    alternating = 1 - 2 * (xp.arange(n) & 1)
    return xp.fft.fft(series * alternating, axis=-1)


def _derivative_weight(frequency, p, xp):
    from .factored_likelihood_with_rotation import FT_SIGN
    if p == 0:
        return xp.ones_like(frequency, dtype=xp.complex128)
    out = (FT_SIGN * 2.0j * math.pi * frequency) ** int(p)
    if p % 2 and frequency.size > 1:
        # The LAL packing has one unpaired +Nyquist bin at index zero.
        out = out.copy()
        out[0] = 0
    return out


def modulate_fd(spectrum, n, delta_f, epoch=0.0, f_sidereal=None,
                backend=None):
    """Apply exp(i n Omega t_abs) with the exact LAL FFT conventions."""
    xp = _resolve_backend(backend)
    x = xp.asarray(spectrum)
    if x.ndim < 1:
        raise ValueError("spectrum must have a frequency axis")
    if int(n) == 0:
        return x.copy()
    if f_sidereal is None:
        from .factored_likelihood_with_rotation import F_SIDEREAL
        f_sidereal = F_SIDEREAL
    count = x.shape[-1]
    delta_t = 1.0 / (count * float(delta_f))
    time = float(epoch) + xp.arange(count) * delta_t
    phase = xp.exp(2.0j * math.pi * int(n) * float(f_sidereal) * time)
    return _lal_forward(_lal_reverse(x, xp) * phase, xp)


def build_compound_basis(base_modes_fd, response_weights, a_list, delta_f,
                         delta_t, epoch, *, base_conjugate_fd=None,
                         f_sidereal=None, backend=None, fft_batch=8):
    """Build a dense device bank with shape ``(A,M,N)``.

    ``base_modes_fd`` is ``(M,N)`` and ``response_weights`` is ``(B,N)``.
    Each ``a=(b,p,n)`` selects one response row, a time derivative, and a
    sidereal modulation.  This function is public mainly for validation; the
    full precompute fails before allocating a bank that cannot safely reside on
    the device; it does not silently spill the derived bank to the host.
    """
    xp = _resolve_backend(backend)
    base = xp.asarray(base_modes_fd)
    weights = xp.asarray(response_weights)
    if base.ndim != 2 or weights.ndim != 2 or base.shape[1] != weights.shape[1]:
        raise ValueError("base_modes_fd and response_weights must be (M,N) and (B,N)")
    if not bool(_to_host(xp.all(xp.isfinite(base)), xp)):
        raise ValueError("base_modes_fd contains non-finite values")
    if not bool(_to_host(xp.all(xp.isfinite(weights)), xp)):
        raise ValueError("response_weights contains non-finite values")
    nfreq = base.shape[1]
    expected_dt = 1.0 / (nfreq * float(delta_f))
    if not np.isclose(float(delta_t), expected_dt, rtol=5e-13, atol=0):
        raise ValueError("delta_t is inconsistent with N and delta_f")
    indices = [tuple(map(int, a)) for a in a_list]
    if not indices:
        raise ValueError("a_list is empty")
    if any(len(a) != 3 or a[0] < 0 or a[0] >= weights.shape[0] or a[1] < 0
           for a in indices):
        raise ValueError("invalid (b,p,n) compound index")
    if fft_batch < 1:
        raise ValueError("fft_batch must be positive")
    fft_batch = min(int(fft_batch), len(indices))
    # One retained bank plus raw/TD/phased/FFT block and the phase table.  Reduce
    # the block automatically; fail before allocating a bank that cannot fit at all.
    bytes_per_a = int(base.shape[0]) * int(nfreq) * np.dtype(np.complex128).itemsize
    free_bytes = _device_free_bytes(xp)
    while fft_batch > 1 and (len(indices) * bytes_per_a +
                             5 * fft_batch * bytes_per_a) > 0.88 * free_bytes:
        fft_batch = max(1, fft_batch // 2)
    if (len(indices) * bytes_per_a + 5 * fft_batch * bytes_per_a) > 0.95 * free_bytes:
        raise MemoryError(
            "compound basis cannot fit safely: retained %.2f GiB, free %.2f GiB" %
            (len(indices) * bytes_per_a / 2.0**30, free_bytes / 2.0**30))
    frequency = lal_frequency_axis(nfreq, delta_f, xp=xp)
    out = xp.empty((len(indices), base.shape[0], nfreq), dtype=xp.complex128)
    # The reverse transform depends on (b,p), not on the sidereal index n.
    # Reuse it across all n without retaining a second full compound bank.
    groups = {}
    for index, (b, p, n) in enumerate(indices):
        groups.setdefault((b, p), []).append((index, n))
    time = float(epoch) + xp.arange(nfreq) * float(delta_t)
    sidereal = float(f_sidereal if f_sidereal is not None else _sidereal_default())
    for (b, p), members in groups.items():
        raw = base * weights[b][None, :] * _derivative_weight(frequency, p, xp)[None, :]
        td = _lal_reverse(raw, xp)
        del raw
        for start in range(0, len(members), fft_batch):
            block = members[start:start + fft_batch]
            phases = xp.stack([
                xp.exp(2.0j * math.pi * n * sidereal * time)
                for unused_index, n in block], axis=0)
            transformed = _lal_forward(td[None, :, :] * phases[:, None, :], xp)
            out[xp.asarray([index for index, unused_n in block])] = transformed
            del phases, transformed
        del td
    if base_conjugate_fd is None:
        return out
    conj_bank = build_compound_basis(
        base_conjugate_fd, response_weights, indices, delta_f, delta_t, epoch,
        f_sidereal=f_sidereal, backend=xp, fft_batch=fft_batch)
    return out, conj_bank


def _sidereal_default():
    from .factored_likelihood_with_rotation import F_SIDEREAL
    return F_SIDEREAL


def compound_precompute_arrays(basis_fd, basis_conj_fd, data_fd, weights2side,
                               delta_f, delta_t, n_shift, n_window, *,
                               backend=None, frequency_chunk=1 << 18,
                               row_block=8, return_device=False, context=None,
                               timing_callback=None):
    """Reduce a compound basis to Q/U/V on the device.

    Inputs use shape ``basis_fd=(A,M,N)`` and exact LAL frequency ordering.
    Outputs are ``Q=(A,M,n_window)`` and ``U,V=(A,A,M,M)``.  Definitions are

    ``U[a,b,i,j] = 2 df sum_f conj(chi[a,i]) chi[b,j] / Sn``
    ``V[a,b,i,j] = 2 df sum_f conj(chi_conj[a,i]) chi[b,j] / Sn``.
    """
    xp = _resolve_backend(backend)
    basis = xp.asarray(basis_fd)
    basis_c = None if basis_conj_fd is None else xp.asarray(basis_conj_fd)
    data = xp.asarray(data_fd)
    weight = xp.asarray(weights2side)
    if basis.ndim != 3 or (basis_c is not None and basis_c.shape != basis.shape):
        raise ValueError("basis_fd and basis_conj_fd must have identical (A,M,N) shape")
    a_count, mode_count, nfreq = basis.shape
    if data.shape != (nfreq,) or weight.shape != (nfreq,):
        raise ValueError("data and weights must have length N")
    if nfreq % 2 or not np.isclose(float(delta_t), 1.0/(nfreq*float(delta_f)),
                                   rtol=5e-13, atol=0):
        raise ValueError("unsupported or inconsistent LAL FFT grid")
    n_shift = int(n_shift); n_window = int(n_window)
    if n_window < 1 or n_window > nfreq:
        raise ValueError("n_window must lie in [1,N]")
    if frequency_chunk < 1 or row_block < 1:
        raise ValueError("frequency_chunk and row_block must be positive")
    # Check before FFT/GEMM: NaNs otherwise turn an entire likelihood into NaN.
    checked = [("basis", basis), ("data", data), ("weights", weight)]
    if basis_c is not None:
        checked.append(("conjugate basis", basis_c))
    for name, arr in checked:
        if not bool(_to_host(xp.all(xp.isfinite(arr)), xp)):
            raise ValueError("%s contains non-finite values" % name)
    if bool(_to_host(xp.any(weight < 0), xp)):
        raise ValueError("weights2side must be nonnegative")

    flat = basis.reshape(a_count * mode_count, nfreq)
    flat_c = None if basis_c is None else basis_c.reshape(a_count * mode_count, nfreq)
    row_count = flat.shape[0]

    # Q: batched inverse FFT.  Roll semantics match lalsimutils.DataRollBins.
    detail_stamp = time.perf_counter()
    q_rows = []
    for start in range(0, row_count, int(row_block)):
        stop = min(start + int(row_block), row_count)
        integrand = (2.0 * xp.conj(flat[start:stop]) * data[None, :] *
                     weight[None, :])
        # LAL COMPLEX16FreqTimeFFT applies the physical frequency-series scale
        # N*delta_f in addition to the normalized inverse DFT.
        full = _lal_reverse(integrand, xp) * (nfreq * float(delta_f))
        full = xp.roll(full, -n_shift, axis=-1)
        q_rows.append(full[:, :n_window].copy())
    q = xp.concatenate(q_rows, axis=0).reshape(a_count, mode_count, n_window)
    detail_stamp = _timed("Q_fft", xp, timing_callback, detail_stamp,
                          rows=row_count, frequencies=nfreq, window=n_window)

    # U,V: accumulate frequency slabs.  The output is tiny compared with the spectra.
    u = xp.zeros((row_count, row_count), dtype=xp.complex128)
    v = None if flat_c is None else xp.zeros_like(u)
    for f0 in range(0, nfreq, int(frequency_chunk)):
        f1 = min(f0 + int(frequency_chunk), nfreq)
        rhs = flat[:, f0:f1] * weight[None, f0:f1]
        u += xp.conj(flat[:, f0:f1]) @ rhs.T
        if v is not None:
            v += xp.conj(flat_c[:, f0:f1]) @ rhs.T
    u *= 2.0 * float(delta_f)
    if v is not None:
        v *= 2.0 * float(delta_f)
    u = u.reshape(a_count, mode_count, a_count, mode_count).transpose(0, 2, 1, 3)
    if v is not None:
        v = v.reshape(a_count, mode_count, a_count, mode_count).transpose(0, 2, 1, 3)
    _timed("U_gram", xp, timing_callback, detail_stamp,
           rows=row_count, frequencies=nfreq,
           includes_v=bool(flat_c is not None))
    if return_device:
        return q, u, v
    return (_to_host(q, xp), _to_host(u, xp),
            None if v is None else _to_host(v, xp))


def streamed_v_matrix(base_conjugate_fd, response_weights, a_list, primary_basis,
                      weights2side, delta_f, delta_t, epoch, *, f_sidereal=None,
                      backend=None, a_block=2, fft_batch=2,
                      frequency_chunk=1 << 18, return_device=False,
                      timing_callback=None):
    """Build conjugate elementary templates in blocks and reduce V immediately.

    This is the production memory bound: at A=40, M=2, N=8388608 the retained
    primary bank is 10.0 GiB, while only ``a_block*M`` conjugate spectra exist at
    once.  The second 10.0 GiB bank is never allocated.
    """
    xp = _resolve_backend(backend)
    primary = xp.asarray(primary_basis)
    base_c = xp.asarray(base_conjugate_fd)
    response = xp.asarray(response_weights)
    weight = xp.asarray(weights2side)
    if primary.ndim != 3 or base_c.ndim != 2:
        raise ValueError("primary_basis and base_conjugate_fd must be (A,M,N) and (M,N)")
    a_count, mode_count, nfreq = primary.shape
    if base_c.shape != (mode_count, nfreq) or response.shape[1] != nfreq \
            or weight.shape != (nfreq,) or len(a_list) != a_count:
        raise ValueError("V inputs have inconsistent shapes")
    if a_block < 1:
        raise ValueError("a_block must be positive")
    primary_flat = primary.reshape(a_count * mode_count, nfreq)
    out = xp.zeros((a_count * mode_count, a_count * mode_count), dtype=xp.complex128)
    basis_seconds = 0.0
    gram_seconds = 0.0
    for a0 in range(0, a_count, int(a_block)):
        a1 = min(a0 + int(a_block), a_count)
        if timing_callback is not None:
            _synchronize(xp)
            block_stamp = time.perf_counter()
        conjugate = build_compound_basis(
            base_c, response, a_list[a0:a1], delta_f, delta_t, epoch,
            f_sidereal=f_sidereal, backend=xp, fft_batch=fft_batch)
        if timing_callback is not None:
            _synchronize(xp)
            now = time.perf_counter()
            basis_seconds += now - block_stamp
            block_stamp = now
        left = conjugate.reshape((a1-a0)*mode_count, nfreq)
        block_out = xp.zeros((left.shape[0], primary_flat.shape[0]), dtype=xp.complex128)
        for f0 in range(0, nfreq, int(frequency_chunk)):
            f1 = min(f0 + int(frequency_chunk), nfreq)
            # Weight the streamed (small) conjugate block, not the retained
            # (A*M,N) primary bank.  The contraction is algebraically identical,
            # while the per-frequency-slab temporary shrinks by A/a_block.
            weighted_left = (xp.conj(left[:, f0:f1])
                             * weight[None, f0:f1])
            block_out += weighted_left @ primary_flat[:, f0:f1].T
            del weighted_left
        r0, r1 = a0 * mode_count, a1 * mode_count
        out[r0:r1] = block_out * (2.0 * float(delta_f))
        if timing_callback is not None:
            _synchronize(xp)
            gram_seconds += time.perf_counter() - block_stamp
        del conjugate, left, block_out
    out = out.reshape(a_count, mode_count, a_count, mode_count).transpose(0, 2, 1, 3)
    if timing_callback is not None:
        details = dict(rows=a_count * mode_count, frequencies=nfreq,
                       a_block=int(a_block))
        timing_callback("V_basis", basis_seconds, details)
        timing_callback("V_gram", gram_seconds, details)
    return out if return_device else _to_host(out, xp)


def _device_free_bytes(xp):
    if _is_numpy(xp):
        return 1 << 62
    free, total = xp.cuda.runtime.memGetInfo()  # pragma: no cover - CUDA only
    # CuPy's pool blocks are reported as used by CUDA but are immediately reusable
    # by this process.  Include them so detector two does not fail a conservative
    # preflight merely because detector one's bank left reusable cached blocks.
    reusable = int(xp.get_default_memory_pool().free_bytes())
    return min(int(total), int(free) + reusable)


def _synchronize(xp):
    if not _is_numpy(xp):  # pragma: no cover - CUDA only
        xp.cuda.Stream.null.synchronize()


def _timed(stage, xp, callback, started, **details):
    if callback is None:
        return time.perf_counter()
    _synchronize(xp)
    now = time.perf_counter()
    callback(stage, now - started, details)
    return now


def _series_arrays(bank_or_dict, xp, mode_order=None):
    """Normalize LAL dictionaries or a DeviceFDModeBank-like object."""
    if hasattr(bank_or_dict, "modes"):
        if not getattr(bank_or_dict, "conditioned", False):
            raise ValueError("waveform provider returned an unconditioned mode bank")
        modes = bank_or_dict.modes
        delta_f = float(bank_or_dict.delta_f)
        delta_t = float(bank_or_dict.delta_t)
        epoch = float(bank_or_dict.epoch)
    else:
        modes = bank_or_dict
        first = next(iter(modes.values()))
        delta_f = float(first.deltaF)
        delta_t = 1.0 / (first.data.length * delta_f)
        epoch = float(first.epoch)
    keys = list(modes)
    if not keys:
        raise ValueError("waveform mode bank is empty")
    if mode_order is not None:
        if set(keys) != set(mode_order):
            raise ValueError("ordinary and conjugate waveform banks have different modes")
        # Legacy dictionary insertion order is not part of the waveform
        # contract. Align by mode labels before upload, not by row position.
        keys = list(mode_order)
    if not hasattr(bank_or_dict, "modes"):
        # The batched FFT uses one common time/frequency grid. Never silently
        # reinterpret a legacy generator's independently shifted mode series.
        expected_length = first.data.length
        epoch_tolerance = max(1e-12, 1e-9 * delta_t)
        for key in keys:
            series = modes[key]
            if series.data.length != expected_length or not np.isclose(
                    float(series.deltaF), delta_f, rtol=5e-13, atol=0.):
                raise ValueError("legacy waveform modes do not share a frequency grid")
            if abs(float(series.epoch) - epoch) > epoch_tolerance:
                raise ValueError("legacy waveform modes do not share a common epoch")
    arrays = []
    for key in keys:
        value = modes[key]
        value = value.data.data if hasattr(value, "data") and hasattr(value.data, "data") else value
        arrays.append(_device_asarray(value, xp, dtype=xp.complex128))
    out = xp.stack(arrays, axis=0)
    if out.shape[1] % 2:
        raise ValueError("waveform mode spectra must use an even two-sided grid")
    return keys, out, delta_f, delta_t, epoch


def _wrap_host_result(detectors, modes, a_list, q_by_det, u_by_det, v_by_det,
                      epoch_by_det, delta_t, skip_interpolation, tgrid_by_det,
                      verbose):
    """Convert packed arrays to the legacy five-return dictionary structure."""
    import lal
    from . import factored_likelihood as FL
    rholms = {}; interpolants = {}; cross = {}; cross_v = {}
    for det in detectors:
        rholms[det] = {}; interpolants[det] = {}; cross[det] = {}; cross_v[det] = {}
        for ai, a in enumerate(a_list):
            rholms[det][a] = {}
            for mi, mode in enumerate(modes):
                ts = lal.CreateCOMPLEX16TimeSeries(
                    "GPU compound Q", lal.LIGOTimeGPS(float(epoch_by_det[det])),
                    0.0, float(delta_t), lal.DimensionlessUnit, q_by_det[det].shape[-1])
                ts.data.data[:] = q_by_det[det][ai, mi]
                rholms[det][a][mode] = ts
            interpolants[det][a] = (None if skip_interpolation else
                                     FL.InterpolateRholms(rholms[det][a],
                                                         tgrid_by_det[det], verbose=verbose))
        for ai, a in enumerate(a_list):
            for aj, ap in enumerate(a_list):
                cross[det][(a, ap)] = {
                    (m1, m2): u_by_det[det][ai, aj, i, j]
                    for i, m1 in enumerate(modes) for j, m2 in enumerate(modes)}
                cross_v[det][(a, ap)] = {
                    (m1, m2): v_by_det[det][ai, aj, i, j]
                    for i, m1 in enumerate(modes) for j, m2 in enumerate(modes)}
    return interpolants, cross, cross_v, rholms


def _physical_device_key(value, name="array"):
    """Return ``(platform, device id)`` for an unsharded array."""
    module = type(value).__module__.split(".")[0]
    if module == "cupy":
        return "gpu", int(value.device.id)
    if module in ("jax", "jaxlib"):
        devices = tuple(value.devices())
        if len(devices) != 1:
            raise ValueError("%s spans %d devices; one physical device is required" %
                             (name, len(devices)))
        return str(devices[0].platform), int(devices[0].id)
    if module == "numpy":
        return "cpu", 0
    raise TypeError("%s is not a recognized NumPy/CuPy/JAX array" % name)


def pack_device_precompute(packed, meta, require_gpu=True):
    """Expose a device result in the five packed objects used by classic ILE.

    This is a view-only operation: Q rows and dense U/V banks remain the exact
    backend arrays returned by ``return_device=True``.  The conventional NoLoop
    compound evaluator already accepts ``rho_by_a`` dictionaries with dense
    ``(A,A,K,K)`` U/V arrays, so no LAL objects or host packing are required.
    """
    required = {"q", "U", "V", "epoch", "delta_t", "modes", "a_list"}
    missing = required.difference(packed)
    if missing:
        raise ValueError("packed device result is missing %s" % sorted(missing))
    if not bool(meta.get("gpu_precompute")) or not bool(meta.get("device_resident")):
        raise ValueError("meta does not describe a device-resident GPU precompute")
    modes = [tuple(map(int, lm)) for lm in packed["modes"]]
    a_list = [tuple(map(int, a)) for a in packed["a_list"]]
    if modes != [tuple(map(int, lm)) for lm in meta.get("modes", ())] \
            or a_list != [tuple(map(int, a)) for a in meta.get("a_list", ())]:
        raise ValueError("packed mode or compound-index order differs from meta")
    detectors = list(packed["q"])
    if not detectors or any(set(packed[name]) != set(detectors)
                            for name in ("U", "V", "epoch")):
        raise ValueError("Q/U/V/epoch detector sets differ")
    A, K = len(a_list), len(modes)
    delta_t = float(packed["delta_t"])
    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("packed delta_t must be finite and positive")
    lookup = {}; rho = {}
    device_key = None
    for det in detectors:
        q = packed["q"][det]; u = packed["U"][det]; v = packed["V"][det]
        if getattr(q, "ndim", None) != 3 or tuple(q.shape[:2]) != (A, K):
            raise ValueError("%s Q must have shape (A,K,N)" % det)
        if tuple(getattr(u, "shape", ())) != (A, A, K, K) \
                or tuple(getattr(v, "shape", ())) != (A, A, K, K):
            raise ValueError("%s U/V must have shape (A,A,K,K)" % det)
        if not np.isfinite(float(packed["epoch"][det])):
            raise ValueError("%s Q epoch is non-finite" % det)
        for label, array in (("Q", q), ("U", u), ("V", v)):
            here = _physical_device_key(array, "%s %s" % (det, label))
            if require_gpu and here[0] != "gpu":
                raise RuntimeError("%s %s is not on a GPU" % (det, label))
            if device_key is None:
                device_key = here
            elif here != device_key:
                raise ValueError("mixed physical devices in packed bank: %r and %r" %
                                 (device_key, here))
        lookup[det] = np.asarray(modes, dtype=int)
        rho[det] = {a: q[i] for i, a in enumerate(a_list)}
    # U and V stay dense.  The maintained evaluator's dense branch consumes
    # exactly this ordering and avoids A^2 Python dictionary entries.
    return lookup, rho, packed["U"], packed["V"], dict(packed["epoch"])


def PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
        event_time_geo, t_window, P, data_dict, psd_dict, Lmax, fMax,
        Qmax=4, L_arm=None, p_max=0, f_sidereal=None,
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.,
        verbose=True, quiet=False, skip_interpolation=False,
        backend=None, context=None, waveform_provider=None, waveform_backend="jax",
        return_device=False,
        fft_batch=4, q_row_batch=4, v_a_block=2,
        frequency_chunk=1 << 18, timing_callback=None, **hlm_kwargs):
    """GPU counterpart of ``PrecomputeLikelihoodTermsRotatingFreqResponse``.

    The default waveform adapter calls RIFT's LAL generator once and uploads its
    conditioned two-sided modes.  A native provider may instead return either
    ``(bank, conjugate_bank)`` or one bank with ``conjugate_modes``.  With
    ``return_device=False`` this returns the exact legacy five-item structure.
    With ``return_device=True`` it returns packed device arrays and metadata, avoiding
    the final narrow host copies for the device-resident ILE/JAX handoff.
    """
    initialization_stamp = time.perf_counter()
    from . import factored_likelihood as FL
    from . import factored_likelihood_rotating_freqresponse as fr
    from . import slowrot_freqresponse as sfr
    from .. import lalsimutils as lsu

    xp = _resolve_backend(backend)
    context = context or default_context(xp)
    if context.backend is not xp:
        raise ValueError("precompute context uses a different array backend")
    context.check_device()
    if set(data_dict) != set(psd_dict) or not data_dict:
        raise ValueError("data and PSD detector sets differ or are empty")
    if analyticPSD_Q:
        raise ValueError("GPU precompute requires a tabulated PSD")
    detectors = list(data_dict)
    if f_sidereal is None:
        f_sidereal = _sidereal_default()
    P.dist = FL.distMpcRef * 1e6 * lsu.lsu_PC
    P.deltaF = data_dict[detectors[0]].deltaF

    stamp = _timed("initialization", xp, timing_callback, initialization_stamp)
    if waveform_provider is None:
        host, host_conj = FL.internal_hlm_generator(
            P, Lmax, verbose=verbose, quiet=quiet, **hlm_kwargs)
        upload_stamp = _timed("waveform_generation", xp, timing_callback, stamp,
                              provider="legacy_host")
        modes, base, delta_f, delta_t, epoch = _series_arrays(host, xp)
        modes_c, base_c, df_c, dt_c, epoch_c = _series_arrays(host_conj, xp, mode_order=modes)
        _timed("waveform_pack_upload", xp, timing_callback, upload_stamp,
               modes=len(modes), frequencies=int(base.shape[-1]))
    else:
        supplied = waveform_provider(P, Lmax, backend=waveform_backend, **hlm_kwargs)
        if isinstance(supplied, tuple) and len(supplied) == 2:
            supplied_main, supplied_conj = supplied
        else:
            supplied_main = supplied
            conj_modes = getattr(supplied, "conjugate_modes", None)
            if conj_modes is None:
                raise ValueError("native waveform provider must supply conjugate_modes")
            supplied_conj = supplied
            supplied_conj = type("ConjugateBankView", (), dict(
                modes=conj_modes, delta_f=supplied.delta_f, delta_t=supplied.delta_t,
                epoch=supplied.epoch, conditioned=supplied.conditioned))()
        modes, base, delta_f, delta_t, epoch = _series_arrays(supplied_main, xp)
        modes_c, base_c, df_c, dt_c, epoch_c = _series_arrays(supplied_conj, xp, mode_order=modes)
    stamp = _timed("waveform", xp, timing_callback, stamp,
                   modes=len(modes), frequencies=int(base.shape[-1]))
    epoch_tol = max(1.0e-12, 1.0e-9 * min(delta_t, dt_c))
    if modes != modes_c or not np.isclose(delta_f, df_c, rtol=5e-13, atol=0.0) \
            or not np.isclose(delta_t, dt_c, rtol=5e-13, atol=0.0) \
            or abs(epoch - epoch_c) > epoch_tol:
        raise ValueError("ordinary and conjugate waveform banks use different grids or modes")
    if not np.isclose(delta_t, float(P.deltaT), rtol=5e-13, atol=0.0):
        raise ValueError("waveform delta_t differs from requested P.deltaT")
    nfreq = base.shape[-1]
    if any(data_dict[d].data.length != nfreq or
           not np.isclose(data_dict[d].deltaF, delta_f, rtol=5e-13, atol=0.0)
           for d in detectors):
        raise ValueError("waveform and detector data grids differ")
    a_list = fr.compound_index_set(int(Qmax), int(p_max))

    q_out = {}; u_out = {}; v_out = {}; q_epoch = {}; tgrid = {}; lengths = {}
    for det in detectors:
        length = fr._arm_for_detector(L_arm, det)
        _, _, _, length = sfr.detector_geometry(det, L_arm=length)
        lengths[det] = float(length)
        f_host = np.asarray(lal_frequency_axis(nfreq, delta_f, xp=np))
        response_host = np.asarray(sfr.finite_size_response_weights(
            f_host, {'L': float(length), 'T': float(length)/sfr.C_SI}, int(Qmax)))
        response = context.array((det, "finite-response"), response_host,
                                 dtype=np.complex128)

        psd = psd_dict[det]
        ip = lsu.ComplexIP(P.fmin, fMax, 1.0/2.0/P.deltaT, P.deltaF, psd,
                           False, inv_spec_trunc_Q, T_spec)
        weight = context.array((det, "weights"), ip.weights2side, dtype=np.float64)
        data = context.array((det, "data"), data_dict[det].data.data,
                             dtype=np.complex128)
        stamp = _timed("input_prep", xp, timing_callback, stamp, detector=det,
                       cache=context.stats())
        basis = build_compound_basis(
            base, response, a_list, delta_f, delta_t, epoch,
            f_sidereal=f_sidereal, backend=xp, fft_batch=fft_batch)
        stamp = _timed("basis", xp, timing_callback, stamp, detector=det,
                       elements=len(a_list), bytes=int(basis.nbytes))

        t_det = FL.ComputeArrivalTimeAtDetector(det, P.phi, P.theta, event_time_geo)
        rho_epoch = float(data_dict[det].epoch) - float(epoch)
        shift = float(t_det) - float(t_window) - rho_epoch
        n_shift = int(shift / P.deltaT + 0.5)
        n_window = int(2.0 * float(t_window) / P.deltaT)
        q_epoch[det] = rho_epoch + n_shift * P.deltaT
        tgrid[det] = np.arange(n_window) * P.deltaT + q_epoch[det]
        q, u, v = compound_precompute_arrays(
            basis, None, data, weight, delta_f, delta_t, n_shift, n_window,
            backend=xp, frequency_chunk=frequency_chunk, row_block=q_row_batch,
            return_device=return_device, timing_callback=timing_callback)
        stamp = _timed("Q_U", xp, timing_callback, stamp, detector=det)
        v = streamed_v_matrix(
            base_c, response, a_list, basis, weight, delta_f, delta_t, epoch,
            f_sidereal=f_sidereal, backend=xp, a_block=v_a_block,
            fft_batch=min(fft_batch, v_a_block), frequency_chunk=frequency_chunk,
            return_device=return_device, timing_callback=timing_callback)
        stamp = _timed("V", xp, timing_callback, stamp, detector=det)
        q_out[det], u_out[det], v_out[det] = q, u, v
        del basis

    meta = dict(feature='rotation_freqresponse', Qmax=int(Qmax), p_max=int(p_max),
                f_sidereal=float(f_sidereal), a_list=a_list, modes=modes,
                event_time_geo=float(event_time_geo), L=lengths, L_arm=L_arm,
                post_phase_required=True, gpu_precompute=True,
                device_resident=bool(return_device), grid_order='lal_descending')
    if return_device:
        _timed("device_export", xp, timing_callback, stamp,
               detectors=len(detectors), cache=context.stats())
        return dict(q=q_out, U=u_out, V=v_out, epoch=q_epoch,
                    delta_t=float(delta_t), modes=modes, a_list=a_list), meta
    interpolants, cross, cross_v, rholms = _wrap_host_result(
        detectors, modes, a_list, q_out, u_out, v_out, q_epoch, delta_t,
        skip_interpolation, tgrid, verbose)
    _timed("host_export", xp, timing_callback, stamp,
           detectors=len(detectors), cache=context.stats())
    return interpolants, cross, cross_v, rholms, meta
