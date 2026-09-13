"""Regression coverage for legacy CPU waveforms in GPU precompute.

The GPU path must default to RIFT's existing ``internal_hlm_generator`` for
every approximant and waveform option, then upload both ordinary and
conjugated conditioned mode banks.  Native waveform providers are opt-in.
"""

import importlib.util
import os
import unittest
from unittest import mock

import numpy as np


def _imports():
    import lal
    import lalsimulation as lalsim
    from RIFT import lalsimutils as lsu
    from RIFT.likelihood import factored_likelihood as fl
    from RIFT.likelihood import factored_likelihood_rotating_freqresponse as fr
    from RIFT.likelihood import gpu_precompute as gpu
    return lal, lalsim, lsu, fl, fr, gpu


def _fd_series(lal, name, values, df, epoch):
    out = lal.CreateCOMPLEX16FrequencySeries(
        name, lal.LIGOTimeGPS(epoch), 0.0, df, lal.DimensionlessUnit, len(values)
    )
    out.data.data[:] = values
    return out


def _short_problem(approximant, precessing=False):
    lal, lalsim, lsu, _, _, _ = _imports()
    event, dt, df = 1.0e9, 1.0 / 512.0, 0.25
    p = lsu.ChooseWaveformParams(
        m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI,
        fmin=40.0, fref=60.0, deltaT=dt, deltaF=df,
        approx=approximant, radec=True, phi=1.2, theta=0.3,
        incl=0.7, psi=0.5, phiref=0.4, tref=event,
        dist=200e6 * lal.PC_SI, detector="H1",
    )
    if precessing:
        p.s1x, p.s1y, p.s1z = 0.2, -0.04, 0.1
        p.s2x, p.s2y, p.s2z = 0.03, 0.1, -0.15
    data = {"H1": lsu.non_herm_hoff(p.manual_copy())}
    n = data["H1"].data.length
    psd = lal.CreateREAL8FrequencySeries(
        "H1 PSD", lal.LIGOTimeGPS(0), 0.0, df, lal.SecondUnit, n // 2 + 1
    )
    frequencies = np.arange(n // 2 + 1) * df
    psd.data.data[:] = [
        lalsim.SimNoisePSDaLIGOZeroDetHighPower(max(10.0, f))
        for f in frequencies
    ]
    return event, p, data, {"H1": psd}


def _packed(fr, result):
    return fr.pack_rotating_freqresponse_arrays(
        result[4], result[3], result[1], result[2]
    )


def _assert_precompute_close(testcase, fr, cpu, candidate, rtol):
    cp, gp = _packed(fr, cpu), _packed(fr, candidate)
    testcase.assertEqual(cpu[4]["modes"], candidate[4]["modes"])
    testcase.assertEqual(cpu[4]["a_list"], candidate[4]["a_list"])
    for det in cp[1]:
        for a in cp[1][det]:
            np.testing.assert_allclose(gp[1][det][a], cp[1][det][a],
                                       rtol=rtol, atol=1e-9)
        np.testing.assert_allclose(gp[2][det], cp[2][det], rtol=rtol, atol=1e-8)
        np.testing.assert_allclose(gp[3][det], cp[3][det], rtol=rtol, atol=1e-8)
        testcase.assertAlmostEqual(gp[4][det], cp[4][det], places=12)


class TestLegacyWaveformContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.lal, cls.lalsim, cls.lsu, cls.fl, cls.fr, cls.gpu = _imports()
        except ImportError as exc:
            raise unittest.SkipTest("RIFT waveform dependencies unavailable: %s" % exc)

    def test_default_forwards_all_options_and_uploads_both_banks(self):
        lal, fl, gpu = self.lal, self.fl, self.gpu
        n, df, dt, epoch = 64, 1.0, 1.0 / 64.0, -0.75
        rng = np.random.default_rng(891)
        ordinary_values = rng.normal(size=n) + 1j * rng.normal(size=n)
        conjugate_values = 3 + rng.normal(size=n) + 1j * rng.normal(size=n)
        ordinary = {(2, 2): _fd_series(lal, "ordinary", ordinary_values, df, epoch)}
        conjugate = {(2, 2): _fd_series(lal, "conjugate", conjugate_values, df, epoch)}

        from types import SimpleNamespace
        p = SimpleNamespace(
            dist=100e6 * lal.PC_SI, deltaF=df, deltaT=dt, fmin=4.0,
            phi=1.1, theta=0.2,
        )
        data_values = rng.normal(size=n) + 1j * rng.normal(size=n)
        data = {"H1": _fd_series(lal, "data", data_values, df, 100.0)}
        psd = lal.CreateREAL8FrequencySeries(
            "PSD", lal.LIGOTimeGPS(0), 0, df, lal.SecondUnit, n // 2 + 1
        )
        psd.data.data[:] = 1.0
        forwarded = dict(
            extra_waveform_kwargs={"fd_standoff_factor": 0.91, "token": "nested"},
            use_gwsignal=True, use_gwsignal_approx="SEOBNRv5PHM",
            use_external_EOB=True, nr_lookup=True,
            NR_group="nr-group", NR_param="nr-param",
            ROM_group="rom-group", ROM_param="rom-param",
            force_22_mode=True, perturbative_extraction=True,
        )
        calls, uploads = [], []
        real_device_asarray = gpu._device_asarray

        def generator(*args, **kwargs):
            calls.append((args, kwargs))
            return ordinary, conjugate

        def record_upload(value, xp, dtype=None):
            uploads.append(np.asarray(value).copy())
            return real_device_asarray(value, xp, dtype=dtype)

        with mock.patch.object(fl, "internal_hlm_generator", side_effect=generator), \
                mock.patch.object(gpu, "_device_asarray", side_effect=record_upload):
            gpu.PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
                100.75, 0.125, p, data, {"H1": psd}, 2, 24.0,
                Qmax=0, p_max=0, backend=np, skip_interpolation=True,
                quiet=True, verbose=False, **forwarded
            )

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertIs(args[0], p)
        self.assertEqual(args[1], 2)
        self.assertFalse(kwargs["verbose"])
        self.assertTrue(kwargs["quiet"])
        for key, value in forwarded.items():
            self.assertEqual(kwargs[key], value)
        # The first two conversions are the distinct ordinary and conjugate
        # waveform arrays.  This guards against regenerating, aliasing, or
        # synthesizing conjugates in the H2D default path.
        np.testing.assert_array_equal(uploads[0], ordinary_values)
        np.testing.assert_array_equal(uploads[1], conjugate_values)

    def test_rejects_legacy_modes_with_different_frequency_grids(self):
        values = np.ones(64, dtype=complex)
        modes = {
            (2, 2): _fd_series(self.lal, "h22", values, 1.0, -0.5),
            (3, 3): _fd_series(self.lal, "h33", values, 0.5, -0.5),
        }
        with self.assertRaisesRegex(ValueError, "do not share a frequency grid"):
            self.gpu._series_arrays(modes, np)

    def test_rejects_legacy_modes_with_different_epochs(self):
        values = np.ones(64, dtype=complex)
        modes = {
            (2, 2): _fd_series(self.lal, "h22", values, 1.0, -0.5),
            (3, 3): _fd_series(self.lal, "h33", values, 1.0, -0.49),
        }
        with self.assertRaisesRegex(ValueError, "do not share a common epoch"):
            self.gpu._series_arrays(modes, np)

    def test_aligns_modes_by_labels_not_dictionary_order(self):
        values = np.arange(64, dtype=complex)
        modes = {
            (2, -2): _fd_series(self.lal,"hm",3*values,1.,-0.5),
            (2, 2): _fd_series(self.lal,"hp",values,1.,-0.5),
        }
        keys, arrays, *_ = self.gpu._series_arrays(
            modes,np,mode_order=[(2,2),(2,-2)])
        self.assertEqual(keys,[(2,2),(2,-2)])
        np.testing.assert_array_equal(arrays[0],values)
        np.testing.assert_array_equal(arrays[1],3*values)


class TestSecondPhysicalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.lal, cls.lalsim, cls.lsu, cls.fl, cls.fr, cls.gpu = _imports()
        except ImportError as exc:
            raise unittest.SkipTest("RIFT waveform dependencies unavailable: %s" % exc)

    def _run(self, xp):
        event, p, data, psd = _short_problem(self.lalsim.TaylorF2)
        common = dict(
            event_time_geo=event, t_window=0.05, P=p, data_dict=data,
            psd_dict=psd, Lmax=2, fMax=200.0, Qmax=0, p_max=0,
            analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.0,
            verbose=False, quiet=True, skip_interpolation=True,
        )
        real_generator = self.fl.internal_hlm_generator

        def reversed_conjugate_order(*args, **kwargs):
            ordinary, conjugate = real_generator(*args, **kwargs)
            return ordinary, dict(reversed(list(conjugate.items())))

        with mock.patch.object(self.fl,"internal_hlm_generator",side_effect=reversed_conjugate_order):
            cpu = self.fr.PrecomputeLikelihoodTermsRotatingFreqResponse(**common)
            candidate = self.gpu.PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
                **common, backend=xp, context=self.gpu.GPUPrecomputeContext(xp)
            )
        _assert_precompute_close(self, self.fr, cpu, candidate,
                                 3e-10 if xp is np else 3e-9)

    def test_taylorf2_numpy_precompute_matches_cpu(self):
        self._run(np)

    def test_taylorf2_cupy_precompute_matches_cpu(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
        except Exception as exc:
            self.skipTest("no usable CuPy GPU: %s" % exc)
        self._run(cp)


class TestGenericMultimodeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.lal, cls.lalsim, cls.lsu, cls.fl, cls.fr, cls.gpu = _imports()
            cls.approximant = cls.lalsim.IMRPhenomXPHM
        except (ImportError, AttributeError) as exc:
            raise unittest.SkipTest("IMRPhenomXPHM unavailable: %s" % exc)

    def _run(self, xp):
        event, p, data, psd = _short_problem(self.approximant, precessing=True)
        # Inspect the real legacy generator result itself: the compatibility
        # claim requires generic labels and a common grid, not a 22-only proxy.
        modes, conjugate = self.fl.internal_hlm_generator(
            p, 4, verbose=False, quiet=True
        )
        labels = list(modes)
        expected_labels = {
            (ell, m) for ell in range(2, 5) for m in range(-ell, ell + 1)
        }
        self.assertEqual(set(labels), expected_labels)
        self.assertEqual(len(labels), 21)
        self.assertEqual(set(labels), set(conjugate))
        first = modes[labels[0]]
        for bank in (modes, conjugate):
            for series in bank.values():
                self.assertEqual(series.data.length, first.data.length)
                self.assertAlmostEqual(series.deltaF, first.deltaF, places=14)
                self.assertAlmostEqual(float(series.epoch), float(first.epoch), places=12)

        common = dict(
            event_time_geo=event, t_window=0.05, P=p, data_dict=data,
            psd_dict=psd, Lmax=4, fMax=200.0, Qmax=0, p_max=0,
            analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.0,
            verbose=False, quiet=True, skip_interpolation=True,
        )
        cpu = self.fr.PrecomputeLikelihoodTermsRotatingFreqResponse(**common)
        candidate = self.gpu.PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
            **common, backend=xp, context=self.gpu.GPUPrecomputeContext(xp)
        )
        _assert_precompute_close(self, self.fr, cpu, candidate,
                                 5e-10 if xp is np else 5e-9)

        cpu_packed, gpu_packed = _packed(self.fr, cpu), _packed(self.fr, candidate)
        pv = p.manual_copy()
        # Row zero is near the injected point; row one is deliberately offset
        # in sky, orientation, phase, and distance.
        pv.phi = np.array([p.phi + 1e-4, p.phi + 0.35])
        pv.theta = np.array([p.theta - 1e-4, p.theta - 0.22])
        pv.incl = np.array([p.incl + 1e-4, 1.15])
        pv.psi = np.array([p.psi + 1e-4, p.psi + 0.31])
        pv.phiref = np.array([p.phiref + 1e-4, p.phiref + 0.47])
        pv.dist = np.array([p.dist * 1.001, p.dist * 1.7])
        pv.tref = event
        pv.deltaT = p.deltaT
        tvals = np.array([-p.deltaT, 0.0, p.deltaT])
        ln_cpu = self.fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
            tvals, pv, cpu[4], *cpu_packed, Lmax=4, array_output=True,
            time_interp="nearest", xpy=np,
        )
        ln_gpu = self.fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
            tvals, pv, candidate[4], *gpu_packed, Lmax=4, array_output=True,
            time_interp="nearest", xpy=np,
        )
        self.assertTrue(np.all(np.isfinite(ln_cpu)))
        self.assertGreater(float(np.max(np.abs(np.asarray(ln_cpu)[0] -
                                                np.asarray(ln_cpu)[1]))), 0.0)
        np.testing.assert_allclose(ln_gpu, ln_cpu,
                                   rtol=5e-10 if xp is np else 5e-9,
                                   atol=2e-8)

    def test_xphm_numpy_precompute_and_likelihood_match_cpu(self):
        self._run(np)

    def test_xphm_cupy_precompute_and_likelihood_match_cpu(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
        except Exception as exc:
            self.skipTest("no usable CuPy GPU: %s" % exc)
        self._run(cp)


class TestGWSignalSEOBNRv5PHM(unittest.TestCase):
    """Short real-model guard for the legacy GWSignal-to-device route."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.lal, cls.lalsim, cls.lsu, cls.fl, cls.fr, cls.gpu = _imports()
        except ImportError as exc:
            raise unittest.SkipTest("RIFT waveform dependencies unavailable: %s" % exc)
        if importlib.util.find_spec("pyseobnr") is None:
            raise unittest.SkipTest("SEOBNRv5PHM needs the optional pyseobnr backend")
        # Once the advertised backend exists, a broken GWSignal import is a
        # compatibility failure rather than an optional-dependency skip.
        if not cls.fl.has_GWS:
            raise RuntimeError("pyseobnr is installed but RIFT could not import GWSignal")

    def _run(self, xp):
        event, p, data, psd = _short_problem(
            self.lalsim.IMRPhenomXPHM, precessing=True
        )
        waveform_kwargs = dict(
            use_gwsignal=True,
            use_gwsignal_approx="SEOBNRv5PHM",
            # This deliberately short 512 Hz test is a transport/parity gate,
            # not a high-mode accuracy study.  Disable the model's per-mode
            # ringdown/Nyquist veto exactly as the maintained diagnostic does.
            extra_waveform_kwargs={"lmax_nyquist": 1},
        )
        common = dict(
            event_time_geo=event, t_window=0.05, P=p, data_dict=data,
            psd_dict=psd, Lmax=4, fMax=200.0, Qmax=0, p_max=0,
            analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.0,
            verbose=False, quiet=True, skip_interpolation=True,
            **waveform_kwargs
        )

        routed_calls = []
        real_gwsignal = self.fl.rgws.std_and_conj_hlmoff

        def record_gwsignal_route(*args, **kwargs):
            banks = real_gwsignal(*args, **kwargs)
            routed_calls.append((args, kwargs, banks))
            return banks

        # P.approx is intentionally an ordinary available LAL approximant:
        # SEOBNRv5PHM is selected only by the explicit GWSignal string.  Thus
        # observing both calls here proves neither precompute silently fell
        # back to the default LAL route.
        with mock.patch.object(
                self.fl.rgws, "std_and_conj_hlmoff",
                side_effect=record_gwsignal_route):
            cpu = self.fr.PrecomputeLikelihoodTermsRotatingFreqResponse(**common)
            candidate = self.gpu.PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
                **common, backend=xp, context=self.gpu.GPUPrecomputeContext(xp)
            )

        self.assertEqual(len(routed_calls), 2)
        for args, kwargs, (ordinary, conjugate) in routed_calls:
            self.assertIsInstance(args[0], self.lsu.ChooseWaveformParams)
            self.assertEqual(args[1], 4)
            self.assertEqual(kwargs["approx_string"], "SEOBNRv5PHM")
            self.assertEqual(kwargs["lmax_nyquist"], 1)
            labels = set(ordinary)
            self.assertEqual(labels, set(conjugate))
            self.assertGreater(len(labels), 2)
            self.assertTrue(any(ell > 2 for ell, _ in labels))
            first = ordinary[next(iter(labels))]
            self.assertEqual(first.data.length, data["H1"].data.length)
            self.assertAlmostEqual(first.deltaF, data["H1"].deltaF, places=14)
            for bank in (ordinary, conjugate):
                for series in bank.values():
                    self.assertEqual(series.data.length, first.data.length)
                    self.assertAlmostEqual(series.deltaF, first.deltaF, places=14)
                    self.assertAlmostEqual(float(series.epoch),
                                           float(first.epoch), places=12)

        self.assertEqual(set(cpu[4]["modes"]), set(candidate[4]["modes"]))
        _assert_precompute_close(self, self.fr, cpu, candidate,
                                 5e-10 if xp is np else 5e-9)

    def test_seobnrv5phm_gwsignal_numpy_precompute_matches_cpu(self):
        self._run(np)

    def test_seobnrv5phm_gwsignal_cupy_precompute_matches_cpu(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
        except Exception as exc:
            if os.environ.get("RIFT_REQUIRE_GPU_PRECOMPUTE") == "1":
                self.fail("mandatory CuPy device gate unavailable: %s" % exc)
            self.skipTest("no usable CuPy GPU: %s" % exc)
        self._run(cp)


if __name__ == "__main__":
    unittest.main()
