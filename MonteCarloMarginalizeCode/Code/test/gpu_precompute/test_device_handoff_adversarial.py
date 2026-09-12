"""Adversarial residency, ownership, and dispatch tests for GPU-to-JAX handoff."""

import gc
import os
from types import SimpleNamespace

import numpy as np
import pytest


def _require_cupy_jax_gpu():
    try:
        import cupy as cp
        import jax
        import jax.numpy as jnp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy reports no CUDA device")
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("JAX reports no GPU device")
        if not bool(jax.config.x64_enabled):
            raise RuntimeError("JAX x64 is disabled")
        cp.cuda.Stream.null.synchronize()
        return cp, jax, jnp
    except Exception as exc:
        if os.environ.get("RIFT_REQUIRE_GPU_PRECOMPUTE") == "1":
            pytest.fail("real CuPy/JAX GPU handoff required: %r" % (exc,))
        pytest.skip("no usable CuPy/JAX GPU handoff: %r" % (exc,))


def _synthetic_bank(seed=20260912):
    from RIFT.likelihood import factored_likelihood_rotating_freqresponse as fr

    rng = np.random.default_rng(seed)
    modes = [(2, 2), (2, -2)]
    a_list = fr.compound_index_set(0, 0)
    a_count, mode_count, n_time = len(a_list), len(modes), 128
    q = (rng.normal(size=(a_count, mode_count, n_time))
         + 1j * rng.normal(size=(a_count, mode_count, n_time)))
    # A positive Hermitian U and zero V keep the synthetic likelihood finite
    # without borrowing the implementation's contraction to make the fixture.
    u = np.zeros((a_count, a_count, mode_count, mode_count), complex)
    for a in range(a_count):
        u[a, a] = np.eye(mode_count) * (0.2 + 0.01 * a)
    v = np.zeros_like(u)
    dt = 1.0 / 1024.0
    # Center the stored Q buffer on the geocentric event.  H1's Earth-center
    # delay and the short integration window then remain well inside support.
    epoch = {"H1": 1_000_000_000.0 - (n_time // 2) * dt}
    meta = dict(
        feature="rotation_freqresponse", gpu_precompute=True,
        device_resident=True, post_phase_required=True,
        event_time_geo=1_000_000_000.0, modes=modes, a_list=a_list,
        Qmax=0, p_max=0, f_sidereal=1.160576e-5,
    )
    tvals = (np.arange(9) - 4) * dt
    return q, u, v, epoch, dt, meta, tvals


def _conventional_data(q, u, v, epoch, dt, meta, tvals, geometry):
    from RIFT.likelihood.jax_ile.banded import build_rotating_freqresponse_data

    a_list = meta["a_list"]
    lookup = {"H1": np.asarray(meta["modes"], dtype=int)}
    rho = {"H1": {a: q[i] for i, a in enumerate(a_list)}}
    u_dict = {"H1": {(a, ap): u[i, j]
                      for i, a in enumerate(a_list)
                      for j, ap in enumerate(a_list)}}
    v_dict = {"H1": {(a, ap): v[i, j]
                      for i, a in enumerate(a_list)
                      for j, ap in enumerate(a_list)}}
    return build_rotating_freqresponse_data(
        meta, lookup, rho, u_dict, v_dict, epoch, dt, tvals, geometry)


def _likelihood_arguments(jnp):
    return [jnp.asarray(value) for value in (
        [1.2, 1.25], [0.3, 0.26], [0.2, 0.7],
        [0.7, 1.05], [0.1, 0.55], [100.0, 160.0],
    )]


def test_cupy_handoff_has_no_bulk_host_copy_and_survives_source_deletion(monkeypatch):
    cp, jax, jnp = _require_cupy_jax_gpu()
    from RIFT.likelihood import slowrot_freqresponse as sfr
    from RIFT.likelihood.gpu_jax_handoff import (
        build_jax_rotating_freqresponse_data_from_device,
    )
    from RIFT.likelihood.jax_ile.core import fused_log_likelihood

    q, u, v, epoch, dt, meta, tvals = _synthetic_bank()
    geometry = {"H1": sfr.detector_geometry("H1", L_arm=4000.0)}
    conventional = _conventional_data(q, u, v, epoch, dt, meta, tvals, geometry)
    args = _likelihood_arguments(jnp)
    expected = fused_log_likelihood(conventional, *args, interp="nearest")
    zero_q = _conventional_data(np.zeros_like(q), u, v, epoch, dt, meta,
                                tvals, geometry)
    zero_q_likelihood = fused_log_likelihood(zero_q, *args, interp="nearest")
    jax.block_until_ready(expected)

    q_device, u_device, v_device = cp.asarray(q), cp.asarray(u), cp.asarray(v)
    packed = dict(
        q={"H1": q_device}, U={"H1": u_device}, V={"H1": v_device},
        epoch=epoch, delta_t=dt, modes=meta["modes"], a_list=meta["a_list"],
    )

    # Any cp.asnumpy call here is a bulk-copy regression.  Small geometry and
    # index tables originate on the host and do not need this escape hatch.
    def forbidden_asnumpy(*unused_args, **unused_kwargs):
        raise AssertionError("GPU handoff copied a CuPy array to host")

    monkeypatch.setattr(cp, "asnumpy", forbidden_asnumpy)
    direct = build_jax_rotating_freqresponse_data_from_device(
        packed, meta, tvals, geometry, require_gpu=True)
    for key in ("Q_bank", "U_bank", "V_bank"):
        value = direct.detectors["H1"][key]
        assert all(device.platform == "gpu" for device in value.devices())
    assert direct.gpu_handoff["contract_Q_U_V_host_copies"] == 0

    # DLPack ownership must outlive every producer-side reference.  Releasing
    # the CuPy pool and churning same-sized blocks makes a borrowed-buffer bug
    # deterministic enough to catch without allocating a long waveform bank.
    shapes = [q_device.shape, u_device.shape, v_device.shape]
    del packed, q_device, u_device, v_device
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    churn = [cp.full(shape, 17.0 + i, dtype=cp.complex128)
             for i, shape in enumerate(shapes)]
    cp.cuda.Stream.null.synchronize()
    del churn
    gc.collect()

    got = fused_log_likelihood(direct, *args, interp="nearest")
    jax.block_until_ready(got)
    np.testing.assert_allclose(
        np.asarray(direct.detectors["H1"]["Q_bank"]),
        np.transpose(q, (0, 2, 1)), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected),
                               rtol=3e-12, atol=3e-12)
    assert np.max(np.abs(np.asarray(got) - np.asarray(zero_q_likelihood))) > 1e-8


def test_handoff_rejects_numpy_banks_before_jax_evaluation():
    from RIFT.likelihood import slowrot_freqresponse as sfr
    from RIFT.likelihood.gpu_jax_handoff import (
        build_jax_rotating_freqresponse_data_from_device,
    )

    q, u, v, epoch, dt, meta, tvals = _synthetic_bank()
    geometry = {"H1": sfr.detector_geometry("H1", L_arm=4000.0)}
    packed = dict(
        q={"H1": q}, U={"H1": u}, V={"H1": v}, epoch=epoch,
        delta_t=dt, modes=meta["modes"], a_list=meta["a_list"],
    )
    with pytest.raises(TypeError, match="device-resident"):
        build_jax_rotating_freqresponse_data_from_device(
            packed, meta, tvals, geometry, require_gpu=False)


@pytest.mark.parametrize("interp", ["nearest", "cubic"])
def test_highlevel_device_precompute_to_classic_likelihood_has_no_bulk_d2h(monkeypatch, interp):
    cp, unused_jax, unused_jnp = _require_cupy_jax_gpu()
    from test_highlevel_integration import _synthetic_problem
    from RIFT.likelihood import factored_likelihood_rotating_freqresponse as fr
    from RIFT.likelihood.gpu_precompute import (
        GPUPrecomputeContext, PrecomputeLikelihoodTermsRotatingFreqResponseGPU,
        pack_device_precompute,
    )

    lal, fl, event, p, data, psd, modes, modes_c = _synthetic_problem()
    monkeypatch.setattr(fl, "internal_hlm_generator",
                        lambda *args, **kwargs: (modes, modes_c))
    common = dict(
        event_time_geo=event, t_window=0.25, P=p, data_dict=data,
        psd_dict=psd, Lmax=2, fMax=24.0, Qmax=1, p_max=1,
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.0,
        verbose=False, quiet=True, skip_interpolation=True,
    )
    monkeypatch.delenv("RIFT_GPU_PRECOMPUTE", raising=False)
    cpu = fr.PrecomputeLikelihoodTermsRotatingFreqResponse(**common)
    cpu_packed = fr.pack_rotating_freqresponse_arrays(
        cpu[4], cpu[3], cpu[1], cpu[2])

    pvec = SimpleNamespace(
        phi=np.array([1.17, 1.21]), theta=np.array([-0.31, -0.28]),
        incl=np.array([0.7, 1.0]), phiref=np.array([0.2, 1.1]),
        psi=np.array([0.4, 0.9]),
        dist=np.array([110.0, 170.0]) * 1e6 * lal.PC_SI,
        tref=lal.LIGOTimeGPS(event), deltaT=p.deltaT,
    )
    tvals = np.array([-p.deltaT, 0.0, p.deltaT])
    expected = fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, pvec, cpu[4], *cpu_packed, Lmax=2, array_output=True,
        time_interp=interp, xpy=np)
    expected_marginal = fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, pvec, cpu[4], *cpu_packed, Lmax=2, array_output=False,
        time_interp=interp, xpy=np)

    real_asnumpy = cp.asnumpy
    transfers = []

    def scalar_only_asnumpy(value, *args, **kwargs):
        ndim = int(getattr(value, "ndim", -1))
        transfers.append((ndim, int(getattr(value, "nbytes", -1))))
        if ndim != 0:
            raise AssertionError("bulk GPU-to-host transfer during device precompute/packing")
        return real_asnumpy(value, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", scalar_only_asnumpy)
    packed, meta = PrecomputeLikelihoodTermsRotatingFreqResponseGPU(
        **common, backend=cp, context=GPUPrecomputeContext(cp),
        return_device=True, fft_batch=7, q_row_batch=9,
        frequency_chunk=37)
    device_packed = pack_device_precompute(packed, meta)
    lookup, rho, u_bank, v_bank, unused_epoch = device_packed
    assert u_bank["H1"] is packed["U"]["H1"]
    assert v_bank["H1"] is packed["V"]["H1"]
    for index, a in enumerate(meta["a_list"]):
        assert cp.shares_memory(rho["H1"][a], packed["q"]["H1"][index])
    got_device = fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, pvec, meta, *device_packed, Lmax=2, array_output=True,
        time_interp=interp, xpy=cp)
    got_marginal = fr.DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, pvec, meta, *device_packed, Lmax=2, array_output=False,
        time_interp=interp, xpy=cp)
    cp.cuda.Stream.null.synchronize()
    assert all(ndim == 0 for ndim, unused_nbytes in transfers)
    np.testing.assert_allclose(real_asnumpy(got_device), expected,
                               rtol=3e-10, atol=2e-8)
    np.testing.assert_allclose(real_asnumpy(got_marginal), expected_marginal,
                               rtol=3e-10, atol=2e-8)


def test_wrapper_gpu_dispatch_bypasses_legacy_pack_and_preserves_options(monkeypatch):
    from RIFT.likelihood import gpu_jax_handoff
    from RIFT.likelihood import gpu_precompute
    from RIFT.likelihood import factored_likelihood_rotating_freqresponse as fr
    from RIFT.likelihood.jax_ile import banded, wrapper

    sentinel_data = object()
    a_list = [(0, 0, 0)]
    modes = [(2, 2)]
    packed = dict(
        q={"H1": np.zeros((1, 1, 3), complex)},
        U={"H1": np.zeros((1, 1, 1, 1), complex)},
        V={"H1": np.zeros((1, 1, 1, 1), complex)},
        epoch={"H1": 999.8}, delta_t=0.125, modes=modes, a_list=a_list,
    )
    meta = dict(feature="rotation_freqresponse", device_resident=True,
                modes=modes, a_list=a_list)
    calls = {}

    def fake_gpu(*args, **kwargs):
        calls["gpu"] = (args, kwargs)
        return packed, meta

    def fake_handoff(*args, **kwargs):
        calls["handoff"] = (args, kwargs)
        return sentinel_data

    monkeypatch.setenv("RIFT_GPU_PRECOMPUTE", "1")
    monkeypatch.setattr(gpu_precompute,
                        "PrecomputeLikelihoodTermsRotatingFreqResponseGPU", fake_gpu)
    monkeypatch.setattr(gpu_jax_handoff,
                        "build_jax_rotating_freqresponse_data_from_device", fake_handoff)
    monkeypatch.setattr(fr, "pack_rotating_freqresponse_arrays",
                        lambda *a, **k: pytest.fail("legacy pack was called"))
    monkeypatch.setattr(banded, "build_rotating_freqresponse_data",
                        lambda *a, **k: pytest.fail("legacy JAX builder was called"))
    monkeypatch.setattr(wrapper.factored_likelihood, "marginalization_time_grid",
                        lambda half, dt, xpy=None: np.array([-dt, 0.0, dt]))

    p = SimpleNamespace(deltaT=0.125)
    result, extras = wrapper.build_rotating_freqresponse_data_from_precompute(
        p, {"H1": object()}, {"H1": object()}, 1000.0, 0.125,
        2, 256.0, t_window=0.25, Qmax=3, L_arm={"H1": 4000.0},
        p_max=2, analyticPSD_Q=False, inv_spec_trunc_Q=True, T_spec=0.75,
        verbose=True, custom_waveform_option="kept",
    )
    assert result is sentinel_data
    gpu_args, gpu_kwargs = calls["gpu"]
    assert gpu_args[:2] == (1000.0, 0.25)
    assert gpu_kwargs["return_device"] is True
    assert gpu_kwargs["Qmax"] == 3 and gpu_kwargs["p_max"] == 2
    assert gpu_kwargs["inv_spec_trunc_Q"] is True and gpu_kwargs["T_spec"] == 0.75
    assert gpu_kwargs["custom_waveform_option"] == "kept"
    handoff_args, handoff_kwargs = calls["handoff"]
    assert handoff_args[0] is packed and handoff_args[1] is meta
    assert extras["meta"] is meta
    assert extras["U_by_aa"] is packed["U"]
    assert extras["V_by_aa"] is packed["V"]


def test_gpu_order_control_fails_before_allocating_reference_bank(monkeypatch):
    from RIFT.likelihood import gpu_precompute
    from RIFT.likelihood.jax_ile import wrapper
    monkeypatch.setenv("RIFT_GPU_PRECOMPUTE", "1")
    monkeypatch.setattr(gpu_precompute,
                        "PrecomputeLikelihoodTermsRotatingFreqResponseGPU",
                        lambda *a, **k: pytest.fail("unexpected reference precompute"))
    with pytest.raises(NotImplementedError, match="response-order selection"):
        wrapper.build_rotating_freqresponse_data_from_precompute(
            SimpleNamespace(deltaT=0.125), {"H1": object()}, {"H1": object()},
            1000.0, 0.125, 2, 256.0,
            order_control={"choose_p": True, "p_reference": 3})
