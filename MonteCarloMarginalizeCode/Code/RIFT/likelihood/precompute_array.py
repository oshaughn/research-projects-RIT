"""Full-rate Q/U/V on NumPy or CuPy; waveform generation stays with LAL.

All arithmetic is complex128 on the original two-sided frequency grid. A context
can be reused across intrinsic points. Detector products are cached, with content
fingerprints to detect even in-place data/PSD edits. No global device cache is used.
"""
import hashlib

import lal
import numpy as np

from RIFT import lalsimutils as lsu


class FullRatePrecompute:
    """Explicit array backend and bounded FFT/left-matrix batches.

    The full ordinary mode matrix stays on the device across detectors. Only
    ``batch_size`` conjugated modes, weighted rows and FFTs are needed at once.
    Public results are the usual host dictionaries and LAL time series.
    """

    def __init__(self, backend="numpy", batch_size=4):
        if backend not in ("numpy", "cupy"):
            raise ValueError("precompute backend must be numpy or cupy")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("precompute batch size must be a positive integer")
        self.backend = backend
        self.batch_size = batch_size
        if backend == "cupy":
            import cupy
            cupy.zeros(1)  # Explicit CuPy never silently falls back to CPU.
            self.xp = cupy
        else:
            self.xp = np
        self._signature = None
        self._detectors = {}

    def _device(self):
        return None if self.xp is np else self.xp.cuda.runtime.getDevice()

    @staticmethod
    def _fingerprint(series):
        if callable(series):
            # Hidden callable state cannot be fingerprinted. Analytic PSDs are
            # rebuilt below, even when the callable's identity is unchanged.
            return ("callable",)
        a = np.asarray(series.data.data if hasattr(series, "deltaF") else series)
        return (a.shape, a.dtype.str,
                hashlib.blake2b(memoryview(np.ascontiguousarray(a)).cast("B"),
                                digest_size=16).digest(),
                getattr(series, "deltaF", None), getattr(series, "f0", None),
                str(getattr(series, "epoch", "")))

    def prepare(self, data_dict, psd_dict, fmin, fmax, fnyq,
                analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.):
        """Cache exact production weights and weighted FD data, once per context.

        Fingerprinting is included in warm-call timings. Analytic PSDs are
        rebuilt on every call because their hidden state cannot be fingerprinted.
        """
        if not data_dict or set(data_dict) != set(psd_dict):
            raise ValueError("precompute requires matching nonempty data/PSD detectors")
        signature = (self._device(), fmin, fmax, fnyq, analyticPSD_Q,
                     inv_spec_trunc_Q, T_spec,
                     tuple((det, self._fingerprint(data_dict[det]),
                            self._fingerprint(psd_dict[det])) for det in data_dict))
        if not analyticPSD_Q and signature == self._signature:
            return
        # Retain one context only; changing inputs or CUDA device releases old arrays.
        self._signature = None
        self._detectors.clear()
        xp = self.xp
        for det, data in data_dict.items():
            ip = lsu.ComplexIP(fmin, fmax, fnyq, data.deltaF, psd_dict[det],
                               analyticPSD_Q, inv_spec_trunc_Q, T_spec)
            if data.data.length != ip.len2side:
                raise ValueError("data grid does not match precompute weights")
            w = xp.asarray(ip.weights2side)
            self._detectors[det] = dict(weights=w,
                weighted_data=xp.asarray(data.data.data) * w,
                data_epoch=lal.LIGOTimeGPS(data.epoch), deltaF=data.deltaF, n=ip.len2side)
        self._signature = signature

    def upload_modes(self, hlms):
        """Upload each ordinary mode once; avoid a second full stack allocation."""
        keys = list(hlms)
        if not keys:
            raise ValueError("precompute requires at least one mode")
        n = hlms[keys[0]].data.length
        H = self.xp.empty((len(keys), n), dtype=np.complex128)
        for i, key in enumerate(keys):
            if hlms[key].data.length != n:
                raise ValueError("waveform modes must share a frequency grid")
            H[i] = self.xp.asarray(hlms[key].data.data)
        return keys, H

    def _host(self, a):
        return a if self.xp is np else a.get()

    def compute(self, hlms, hlms_conj, det, N_shift, N_window, modes=None):
        """Return U, V and Q with production normalization, lag roll and epochs."""
        if self._signature is None or self._device() != self._signature[0]:
            raise ValueError("prepare precompute inputs on the current device before computing")
        xp = self.xp
        keys, H = self.upload_modes(hlms) if modes is None else modes
        prep = self._detectors[det]
        n, df = prep["n"], prep["deltaF"]
        if H.shape != (len(keys), n) or not 0 < N_window <= n:
            raise ValueError("waveform/data lengths or overlap window do not match")
        # ComputeModeIPTimeSeries requires exact equality for the first mode,
        # before the per-mode ComplexOverlap checks using TOL_DF.
        if hlms[keys[0]].deltaF != df:
            raise ValueError("first waveform/data frequency grid does not match")
        for key in keys:
            if (abs(hlms[key].deltaF - df) > lsu.TOL_DF or
                    hlms_conj[key].data.length != n or
                    abs(hlms_conj[key].deltaF - df) > lsu.TOL_DF):
                raise ValueError("waveform/data frequency grids do not match")
        U, V, rholms = {}, {}, {}
        # Indexing implements DataRollBins followed by CutCOMPLEX16TimeSeries.
        indices = xp.asarray((N_shift + np.arange(N_window)) % n)
        for start in range(0, len(keys), self.batch_size):
            stop = min(start + self.batch_size, len(keys))
            block = H[start:stop]
            weighted = xp.conj(block) * prep["weights"]
            u = self._host((2. * df) * (weighted @ H.T))
            del weighted
            C = xp.empty(block.shape, dtype=np.complex128)
            for i, key in enumerate(keys[start:stop]):
                C[i] = xp.asarray(hlms_conj[key].data.data)
            # V uses the actual generated conjugated waveform, without imposing
            # nonprecessing mode symmetries or reconstructing negative frequencies.
            xp.conj(C, out=C)
            C *= prep["weights"]
            v = self._host((2. * df) * (C @ H.T))
            del C
            integrand = 2. * xp.conj(block) * prep["weighted_data"]
            # LAL's complex FFTs use centred frequency order and df normalization.
            qfull = xp.fft.ifft(xp.fft.ifftshift(integrand, axes=-1), axis=-1)
            del integrand
            q = self._host(qfull[:, indices] * (n * df))
            del qfull
            for i, key in enumerate(keys[start:stop]):
                for j, other in enumerate(keys):
                    U[key, other] = u[i, j]
                    V[key, other] = v[i, j]
                dt = 1. / (n * df)
                ts = lal.CreateCOMPLEX16TimeSeries("Complex overlap",
                    prep["data_epoch"] - hlms[key].epoch, 0., dt,
                    lsu.lsu_DimensionlessUnit, N_window)
                ts.epoch += N_shift * dt
                ts.data.data = q[i]
                rholms[key] = ts
        return U, V, rholms
