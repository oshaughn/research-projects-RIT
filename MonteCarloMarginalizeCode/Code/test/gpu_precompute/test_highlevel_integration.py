"""High-level CPU/GPU wrapper and downstream ILE likelihood parity.

The waveform generator is replaced with a deterministic synthetic LAL mode bank.  Detector
geometry, finite-arm response weights, PSD handling, epochs, packing, and the maintained
rotation/frequency-response likelihood are real production code.
"""

from types import SimpleNamespace

import numpy as np
import pytest


def _fd_series(lal, name, values, delta_f, epoch):
    out = lal.CreateCOMPLEX16FrequencySeries(
        name, lal.LIGOTimeGPS(float(epoch)), 0.0, float(delta_f),
        lal.DimensionlessUnit, len(values))
    out.data.data[:] = values
    return out


def _synthetic_problem():
    lal = pytest.importorskip("lal")
    from RIFT.likelihood import factored_likelihood as fl

    n, df = 256, 0.25
    dt = 1.0 / (n * df)
    event = 1_420_000_000.125
    rng = np.random.default_rng(1509)
    # Full non-Hermitian RIFT spectra, including unequal positive/negative halves.
    h = rng.normal(size=n) + 1j * rng.normal(size=n)
    hc = rng.normal(size=n) + 1j * rng.normal(size=n)
    d = 0.8 * h + rng.normal(size=n) + 1j * rng.normal(size=n)
    h *= np.linspace(0.7, 1.4, n)
    hc *= np.linspace(1.3, 0.6, n)
    data = _fd_series(lal, "H1 synthetic data", d, df, event - 2.0)
    mode = (2, 2)
    modes = {mode: _fd_series(lal, "h22", h, df, -2.0)}
    modes_c = {mode: _fd_series(lal, "hc22", hc, df, -2.0)}
    psd = lal.CreateREAL8FrequencySeries(
        "H1 synthetic PSD", lal.LIGOTimeGPS(0), 0.0, df,
        lal.DimensionlessUnit, n // 2 + 1)
    psd.data.data[:] = 1.0 + 0.01 * np.arange(n // 2 + 1)
    p = SimpleNamespace(
        dist=100.0 * 1e6 * lal.PC_SI, deltaF=df, deltaT=dt,
        fmin=2.0, phi=1.17, theta=-0.31,
    )
    return lal, fl, event, p, {"H1": data}, {"H1": psd}, modes, modes_c


@pytest.mark.parametrize("backend_name", ["numpy", "cupy"])
def test_highlevel_precompute_pack_epoch_and_downstream_likelihood(monkeypatch, request, backend_name):
    lal, fl, event, p, data, psd, modes, modes_c = _synthetic_problem()
    from RIFT.likelihood import factored_likelihood_rotating_freqresponse as fr
    from RIFT.likelihood.gpu_precompute import (
        GPUPrecomputeContext, PrecomputeLikelihoodTermsRotatingFreqResponseGPU)

    if backend_name == "numpy":
        xp = np
    else:
        try:
            import cupy as xp
            if xp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
        except Exception as exc:
            if request.config.getoption("--require-gpu") or pytestconfig_requires_gpu():
                pytest.fail("real CUDA device required: %r" % (exc,))
            pytest.skip("no usable CUDA device: %r" % (exc,))

    monkeypatch.setattr(fl, "internal_hlm_generator",
                        lambda *args, **kwargs: (modes, modes_c))
    common = dict(
        event_time_geo=event, t_window=0.25, P=p, data_dict=data,
        psd_dict=psd, Lmax=2, fMax=24.0, Qmax=1, p_max=1,
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.0,
        verbose=False, quiet=True, skip_interpolation=True,
    )
    # Force the maintained CPU implementation even if the submission environment set the
    # feature variable globally.
    monkeypatch.delenv("RIFT_GPU_PRECOMPUTE", raising=False)
    cpu = fr.PrecomputeLikelihoodTermsRotatingFreqResponse(**common)
    timings = []
    got = PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
        **common, backend=xp, context=GPUPrecomputeContext(xp), return_device=False,
        fft_batch=7, q_row_batch=9, frequency_chunk=37,
        timing_callback=lambda stage, elapsed, details: timings.append((stage, elapsed)))
    assert timings[0][0] == "initialization"
    assert [stage for stage, _ in timings[:4]] == [
        "initialization", "waveform_generation", "waveform_pack_upload", "waveform"]
    assert all(np.isfinite(elapsed) and elapsed >= 0 for _, elapsed in timings)

    cpu_i, cpu_u, cpu_v, cpu_q, cpu_meta = cpu
    got_i, got_u, got_v, got_q, got_meta = got
    assert cpu_meta["a_list"] == got_meta["a_list"]
    assert cpu_meta["modes"] == got_meta["modes"]
    assert got_meta["post_phase_required"] is True
    for a in cpu_meta["a_list"]:
        for mode in cpu_meta["modes"]:
            cts, gts = cpu_q["H1"][a][mode], got_q["H1"][a][mode]
            assert float(cts.epoch) == pytest.approx(float(gts.epoch), abs=1e-12)
            assert cts.deltaT == pytest.approx(gts.deltaT, rel=0, abs=1e-15)
            np.testing.assert_allclose(gts.data.data, cts.data.data,
                                       rtol=2e-10, atol=2e-10)
        for ap in cpu_meta["a_list"]:
            for pair in cpu_u["H1"][(a, ap)]:
                np.testing.assert_allclose(got_u["H1"][(a, ap)][pair],
                                           cpu_u["H1"][(a, ap)][pair],
                                           rtol=2e-10, atol=2e-10)
                np.testing.assert_allclose(got_v["H1"][(a, ap)][pair],
                                           cpu_v["H1"][(a, ap)][pair],
                                           rtol=2e-10, atol=2e-10)

    cpu_packed = fr.pack_rotating_freqresponse_arrays(cpu_meta, cpu_q, cpu_u, cpu_v)
    got_packed = fr.pack_rotating_freqresponse_arrays(got_meta, got_q, got_u, got_v)
    # Evaluate three times around the detector arrival.  This exercises epoch placement,
    # Q slicing, response coefficients, post-phases, U and V through maintained ILE code.
    pvec = SimpleNamespace(
        phi=np.array([1.17, 1.21]), theta=np.array([-0.31, -0.28]),
        incl=np.array([0.7, 1.0]), phiref=np.array([0.2, 1.1]),
        psi=np.array([0.4, 0.9]),
        dist=np.array([110.0, 170.0]) * 1e6 * lal.PC_SI,
        tref=lal.LIGOTimeGPS(event), deltaT=p.deltaT,
    )
    tvals = np.array([-p.deltaT, 0.0, p.deltaT])
    ln_cpu = fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, pvec, cpu_meta, *cpu_packed, Lmax=2, array_output=True,
        time_interp="nearest", xpy=np)
    ln_got = fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, pvec, got_meta, *got_packed, Lmax=2, array_output=True,
        time_interp="nearest", xpy=np)
    np.testing.assert_allclose(ln_got, ln_cpu, rtol=2e-10, atol=2e-8)


def test_context_replaces_old_response_orders_and_cutoffs(monkeypatch):
    from RIFT.likelihood.gpu_precompute import (
        GPUPrecomputeContext, PrecomputeLikelihoodTermsRotatingFreqResponseGPU)
    _, fl, event, p, data, psd, modes, modes_c = _synthetic_problem()
    monkeypatch.setattr(fl, "internal_hlm_generator", lambda *a, **k: (modes, modes_c))
    context = GPUPrecomputeContext(np)
    for order, cutoff, arm in [(0, 24., 4000.), (1, 20., 3000.), (0, 22., 3500.)]:
        common = dict(event_time_geo=event, t_window=.25, P=p,
                      data_dict=data, psd_dict=psd, Lmax=2, fMax=cutoff,
                      Qmax=order, p_max=0, L_arm=arm, backend=np,
                      return_device=True, verbose=False, quiet=True)
        reused, _ = PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
            **common, context=context)
        fresh, _ = PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
            **common, context=GPUPrecomputeContext(np))
        assert context.stats()["retained_arrays"] == 3
        for name in ("q", "U", "V"):
            np.testing.assert_allclose(reused[name]["H1"], fresh[name]["H1"])


def pytestconfig_requires_gpu():
    # pytest's fixture object is deliberately not threaded through the scientific helper;
    # the environment is the stable Condor/container gate used by README.md.
    import os
    return os.environ.get("RIFT_REQUIRE_GPU_PRECOMPUTE") == "1"
