"""Multirate per-point preparation on an array backend (TwoRateDevice, ComputeModeCrossTermsPiecesDevice)
against the CPU reference path (Q_two_rate, ComputeModeCrossTermsPieces).  The cupy legs are in
test_multirate_cupy.py (GPU host only); RIFT_CI_REQUIRE_GPU=1 runs the cupy cases here too."""
import numpy as np
import pytest
import lal

import RIFT.lalsimutils as lsu
from RIFT.likelihood import factored_likelihood_multirate as flm
from test_multirate_precompute import _P, _schedule, inputs, T_GEO, LMAX, FMAX   # noqa: F401 (fixture)


def _run(tmp_path, inputs, **kw):
    flm._JOB_CACHE.clear()
    return flm.PrecomputeLikelihoodTermsMultirate(lal.LIGOTimeGPS(T_GEO), 0.05, _P(), inputs["data"], inputs["psd"],
                                                  LMAX, FMAX, verbose=False, quiet=True,
                                                  schedule=_schedule(tmp_path, "two_rate"), skip_interpolation=True, **kw)


def _assert_close(a, b, tol):
    _, Ua, Va, Ra, _, _ = a
    _, Ub, Vb, Rb, _, _ = b
    assert set(Ra) == set(Rb)
    for det in Ra:
        assert set(Ua[det]) == set(Ub[det]) == set(Va[det]) == set(Vb[det])
        assert set(Ra[det]) == set(Rb[det])
        sU = max(abs(v) for v in Ua[det].values())
        for key in Ua[det]:
            assert abs(Ua[det][key] - Ub[det][key]) <= tol * sU, (det, "U", key)
            assert abs(Va[det][key] - Vb[det][key]) <= tol * sU, (det, "V", key)
        sQ = max(np.max(np.abs(v.data.data)) for v in Ra[det].values())
        for k in Ra[det]:
            assert abs(float(Ra[det][k].epoch) - float(Rb[det][k].epoch)) < 1e-9
            assert Ra[det][k].data.length == Rb[det][k].data.length
            assert np.max(np.abs(Ra[det][k].data.data - Rb[det][k].data.data)) <= tol * sQ, (det, "Q", k)
    assert abs(a[4] - b[4]) <= tol * max(abs(a[4]), 1e-30)


def test_ifft_centred_matches_lal():
    rng = np.random.default_rng(1)
    n, df = 4096, 0.25
    hf = lal.CreateCOMPLEX16FrequencySeries("h", lal.LIGOTimeGPS(0), -n * df / 2, df, lsu.lsu_HertzUnit, n)
    hf.data.data = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ref = lsu.DataInverseFourier(hf).data.data
    new = flm.ifft_centred(np, hf.data.data, df)
    assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_xphm_matches_unrestricted_mode_generation():
    """An independent waveform oracle: backend parity alone misses a wrong template."""
    import lalsimulation as lalsim
    p = _P()
    p.approx = lalsim.IMRPhenomXPHM
    p.dist = 100e6 * lal.PC_SI
    actual = flm._per_m_modes(p, 4, 256, 512, 20., 24.)
    for am in (1, 2, 3, 4):
        q = p.manual_copy()
        q.fmin, q.deltaT, q.deltaF = am * 10., 1. / 256, 1. / 512
        reference, _ = lsu.std_and_conj_hlmoff(q, Lmax=4, fd_alignment_postevent_time=24.,
            extra_waveform_args=dict(PhenomXHMThresholdMband=0, PhenomXPHMThresholdMband=0))
        for key in actual:
            if abs(key[1]) == am:
                a, b = actual[key].data.data, reference[key].data.data
                assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(b)), key


def test_backend_numpy_matches_reference(tmp_path, inputs):
    ref = _run(tmp_path, inputs, multirate_reference=True)
    new = _run(tmp_path, inputs, multirate_xpy="numpy")
    _assert_close(ref, new, 1e-10)


# (3, +-2) reaches LAL only to ~1e-6: LAL's own spherical-harmonic ringdown uses finite differences
# (gpu_waveform_xhm's gate, RIFT #465); every other mode agrees to roundoff
XP_WF_TOL = {(3, 2): 1e-5}


def test_xp_waveform_matches_lal_pieces(xp=np):
    """Device templates (gpu_waveform_xhm plus hlmoft's conditioning on xp) against lalsimulation."""
    p = _P()
    p.m1, p.m2, p.s1z, p.s2z, p.phiref = 30 * lal.MSUN_SI, 25 * lal.MSUN_SI, 0.3, -0.2, 0.7
    p.dist = 100e6 * lal.PC_SI
    for fref in (10., 0.):
        p.fref = fref
        ref = flm._per_m_modes(p, 4, 2048, 16, 10., 2.)
        got = flm._per_m_modes_xp(xp, p, 4, 2048, 16, 10., 2.)
        assert sorted(ref) == sorted(got)
        for k in ref:
            a, b = flm._to_host(xp, got[k].dev), ref[k].data.data
            assert abs(got[k].epoch - float(ref[k].epoch)) < 1e-9 and got[k].deltaF == ref[k].deltaF
            tol = XP_WF_TOL.get((k[0], abs(k[1])), 1e-10)
            assert np.max(np.abs(a - b)) <= tol * np.max(np.abs(b)), (fref, k)


def test_xp_waveform_precompute_matches_lal(tmp_path, inputs):
    a = _run(tmp_path, inputs, multirate_xpy="numpy")
    b = _run(tmp_path, inputs, multirate_xpy="numpy", multirate_waveform="xp")
    _assert_close(a, b, 1e-6)


@pytest.mark.parametrize("bad", [dict(multirate_waveform="xp", multirate_reference=True),
                                 dict(multirate_waveform="jax")])
def test_xp_waveform_refusals(tmp_path, inputs, bad):
    with pytest.raises(ValueError):
        _run(tmp_path, inputs, **bad)


@pytest.mark.parametrize("change", ["window", "fmax", "lmax", "psd", "data", "sky", "truncation"])
def test_cache_tracks_inputs(tmp_path, inputs, change):
    """A warm plan must give the same result as a fresh plan after inputs change."""
    data = {
        "data": {k: lal.CutCOMPLEX16FrequencySeries(v, 0, v.data.length) for k, v in inputs["data"].items()},
        "psd": {k: lal.CutREAL8FrequencySeries(v, 0, v.data.length) for k, v in inputs["psd"].items()},
    }
    schedule = _schedule(tmp_path, "two_rate")
    p = _P()
    args = dict(t_window=0.05, Lmax=LMAX, fMax=FMAX)

    def run():
        return flm.PrecomputeLikelihoodTermsMultirate(
            lal.LIGOTimeGPS(T_GEO), P=p, data_dict=data["data"], psd_dict=data["psd"],
            schedule=schedule, skip_interpolation=True, verbose=False, quiet=True,
            multirate_xpy="numpy", **args)

    flm._JOB_CACHE.clear()
    run()
    if change == "window":
        args["t_window"] = 0.075
    elif change == "fmax":
        args["fMax"] = 1200.
    elif change == "lmax":
        args["Lmax"] = 2
    elif change == "psd":
        data["psd"]["H1"].data.data *= 1.7
    elif change == "data":
        data["data"]["H1"].data.data *= 1.7
    elif change == "sky":
        p.phi += 0.5
    else:
        args.update(inv_spec_trunc_Q=True, T_spec=2.)
    warm = run()
    flm._JOB_CACHE.clear()
    cold = run()
    _assert_close(cold, warm, 1e-10)


def test_invalid_backend_refused():
    with pytest.raises(ValueError, match="backend"):
        flm._select_xpy("typo")


@pytest.mark.parametrize("field,value", [("fs_e", 0), ("guard", float("nan")),
                                         ("atten_db", 4), ("mc_range", [1.3, 1.1]),
                                         ("mc_range", [1.1]), ("late_buffer", 64)])
def test_bad_schedule_refused(tmp_path, field, value):
    sch = _schedule(tmp_path, "two_rate")
    sch[field] = value
    with pytest.raises(ValueError):
        flm._validate_schedule(sch)


def test_cache_device_isolation(tmp_path, inputs):
    from types import SimpleNamespace
    device = [0]
    xp = SimpleNamespace(__name__="cupy", cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda: device[0])))
    schedule = _schedule(tmp_path, "two_rate")
    flm._JOB_CACHE.clear()
    flm._cache_context(inputs["data"], inputs["psd"], schedule, (), xp)
    flm._JOB_CACHE["sentinel"] = True
    flm._cache_context(inputs["data"], inputs["psd"], schedule, (), xp)
    assert flm._JOB_CACHE["sentinel"]
    device[0] = 1
    flm._cache_context(inputs["data"], inputs["psd"], schedule, (), xp)
    assert "sentinel" not in flm._JOB_CACHE


@pytest.mark.parametrize("ratio", [1, 2, 8])
def test_complex_correlation_offsets(ratio):
    import os
    xp = flm._select_xpy("cupy" if os.environ.get("RIFT_CI_REQUIRE_GPU") else "numpy")
    rng = np.random.default_rng(17)
    def noise(n):
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)
    fs, n, nl = 512, 2048, 512
    sch = flm.TwoRateSchedule(fs, fs / ratio, 300., 4., 160.,
                             mc_min=1.1, m_max=2, lag_halfwidth=0.15)
    prep = dict(dE=noise(n // ratio), dL=noise(607), i_l0=n - 607)
    he, hl = noise(n // ratio), noise(nl)
    lags = np.arange(-77, 83)
    # Put the synthetic template-window transition inside this short array. The
    # physical schedule's long tau_hc otherwise makes wh identically zero here.
    j_peak = int(round(sch.tau_hc * fs)) + n // 2
    wh = flm.early_template_window(len(he), j_peak, sch, 1. / fs)
    assert wh.min() < 0.5 < wh.max()
    dev = flm.TwoRateDevice(xp, prep, sch, 1. / fs, len(he), nl, n, j_peak, lags, n - nl)
    actual = flm._to_host(xp, dev.Q(xp.asarray(he), xp.asarray(hl)))
    ref = flm.Q_two_rate(prep, he, hl, j_peak, n - nl, sch, 1. / fs, lags)
    np.testing.assert_allclose(actual, ref, rtol=1e-11, atol=1e-11 * np.max(np.abs(ref)))


@pytest.mark.parametrize("case", ["xhm_l2", "xhm_l3", "xphm", "mass_low", "mass_high", "spin", "early512", "network_truncated", "srate2048", "srate8192"])
def test_backend_configurations(tmp_path, inputs, case):
    import lalsimulation as lalsim
    # RIFT_CI_REQUIRE_GPU turns a missing device into a failed test, never a skip.
    import os
    xp = flm._select_xpy("cupy" if os.environ.get("RIFT_CI_REQUIRE_GPU") else None)
    p = _P()
    schedule = _schedule(tmp_path, "two_rate")
    lmax, trunc, window, fmax = LMAX, False, 0.05, FMAX
    data, psds = dict(inputs["data"]), dict(inputs["psd"])
    if case.startswith("xhm_l"):
        lmax = int(case[-1])
    elif case == "xphm":
        p.approx = lalsim.IMRPhenomXPHM
    elif case.startswith("mass_"):
        mc = (p.m1 * p.m2)**0.6 / (p.m1 + p.m2)**0.2 / lal.MSUN_SI
        scale = (1.101 if case == "mass_low" else 1.299) / mc
        p.m1 *= scale
        p.m2 *= scale
    elif case == "spin":
        p.s1z, p.s2z = 0.1, -0.05
    elif case == "early512":
        schedule["fs_e"] = 512
    elif case.startswith("srate"):
        fs = int(case[5:])
        p.deltaT = 1. / fs
        n = 512 * fs
        psd = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0), 0., 1. / 512,
                                             lsu.lsu_HertzUnit, n // 2 + 1)
        lalsim.SimNoisePSDaLIGOAdVO4T1800545(psd, 10.)
        psd.data.data = np.where(psd.data.data > 0, psd.data.data, np.inf)
        td = lal.CreateCOMPLEX16TimeSeries("d", data["H1"].epoch, 0., p.deltaT,
                                          lsu.lsu_DimensionlessUnit, n)
        td.data.data = 1e-22 * np.random.default_rng(31).standard_normal(n) + 0j
        data, psds = {"H1": lsu.DataFourier(td)}, {"H1": psd}
        fmax = min(FMAX, 0.45 * fs)
    else:
        data["L1"] = lal.CutCOMPLEX16FrequencySeries(data["H1"], 0, data["H1"].data.length)
        psds["L1"] = lal.CutREAL8FrequencySeries(psds["H1"], 0, psds["H1"].data.length)
        data["L1"].data.data *= 0.7
        psds["L1"].data.data *= 1.3
        trunc, window = True, 0.075

    def run(**kw):
        return flm.PrecomputeLikelihoodTermsMultirate(
            lal.LIGOTimeGPS(T_GEO), window, p, data, psds, lmax, fmax,
            schedule=schedule, inv_spec_trunc_Q=trunc, T_spec=2. if trunc else 0.,
            skip_interpolation=False, verbose=False, quiet=True, **kw)

    flm._JOB_CACHE.clear()
    ref = run(multirate_reference=True)
    actual = run(multirate_xpy=xp.__name__)
    _assert_close(ref, actual, 1e-10)
    for det in data:
        for mode in ref[0][det]:
            t = float(ref[3][det][mode].epoch) + np.arange(3, 12) * p.deltaT
            np.testing.assert_allclose(ref[0][det][mode](t), actual[0][det][mode](t), rtol=1e-9,
                                       atol=1e-10 * np.max(np.abs(ref[3][det][mode].data.data)))
