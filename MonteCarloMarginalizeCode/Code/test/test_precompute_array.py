"""Full-rate array precompute against the production LAL implementation."""
import os
import numpy as np
import pytest
import lal
import lalsimulation as lalsim

from RIFT import lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood.precompute_array import FullRatePrecompute


def backends():
    return ["numpy", "cupy"] if os.environ.get("RIFT_CI_REQUIRE_GPU") else ["numpy"]


def fixture_inputs(n=4096, fs=1024):
    rng = np.random.default_rng(436)
    df = fs / n
    def frequency(a, epoch):
        s = lal.CreateCOMPLEX16FrequencySeries("f", lal.LIGOTimeGPS(epoch), -fs/2, df,
                                              lal.HertzUnit, n)
        s.data.data = a
        return s
    def noise():
        return rng.normal(size=n) + 1j*rng.normal(size=n)
    keys = [(2,-2), (2,2), (3,1), (4,-3), (4,4)]
    h = {k: frequency(noise(), -3.25) for k in keys}
    # Independent complex arrays catch incorrect V conjugations/symmetries.
    c = {k: frequency(noise(), -3.25) for k in keys}
    data = {k: frequency(noise(), 1000000000) for k in ["H1", "L1"]}
    psd = {}
    for i, det in enumerate(data):
        p = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0), 0., df,
                                          lal.HertzUnit, n//2+1)
        p.data.data = (1+i) * (1+np.arange(n//2+1)**.2)
        psd[det] = p
    return h, c, data, psd


def assert_products(ref, actual, tol=1e-11):
    for a, b in zip(ref[:2], actual[:2]):
        assert set(a) == set(b)
        scale = max(abs(v) for v in a.values())
        assert max(abs(a[k]-b[k]) for k in a) < tol*scale
    assert set(ref[2]) == set(actual[2])
    scale = max(np.max(np.abs(v.data.data)) for v in ref[2].values())
    for k, a in ref[2].items():
        b = actual[2][k]
        assert a.epoch == b.epoch
        assert a.deltaT == b.deltaT
        assert a.data.length == b.data.length
        assert np.max(np.abs(a.data.data-b.data.data)) < tol*scale


@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("batch", [1, 4, 20])
@pytest.mark.parametrize("trunc", [False, True])
@pytest.mark.parametrize("shift", [-107, 0, 201, 4080])
def test_products(backend, batch, trunc, shift):
    h, c, data, psd = fixture_inputs()
    ctx = FullRatePrecompute(backend, batch)
    ctx.prepare(data, psd, 30., 400., 512., False, trunc, .25 if trunc else 0.)
    modes = ctx.upload_modes(h)
    for det in data:
        args = (psd[det], 30., 400., 512., .25, False, trunc, .25 if trunc else 0.)
        ref = (fl.ComputeModeCrossTermIP(h, h, *args, verbose=False),
               fl.ComputeModeCrossTermIP(c, h, *args, verbose=False),
               fl.ComputeModeIPTimeSeries(h, data[det], psd[det], 30., 400., 512.,
                                         shift, 64, False, trunc, .25 if trunc else 0.))
        assert_products(ref, ctx.compute(h, c, det, shift, 64, modes))


@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("change", ["data", "psd", "fmax", "truncation"])
def test_cache_refresh(backend, change):
    h, c, data, psd = fixture_inputs()
    ctx = FullRatePrecompute(backend)
    args = [data, psd, 30., 400., 512., False, False, 0.]
    ctx.prepare(*args)
    old = ctx._detectors["H1"]["weighted_data"]
    ctx.prepare(*args)
    assert ctx._detectors["H1"]["weighted_data"] is old
    if change == "data":
        data["H1"].data.data *= 1.7
    elif change == "psd":
        psd["H1"].data.data *= 1.7
    elif change == "fmax":
        args[3] = 350.
    else:
        args[-2:] = [True, .25]
    ctx.prepare(*args)
    assert ctx._detectors["H1"]["weighted_data"] is not old
    fresh = FullRatePrecompute(backend)
    fresh.prepare(*args)
    assert_products(fresh.compute(h, c, "H1", -10, 64), ctx.compute(h, c, "H1", -10, 64))


@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("precessing", [False, True])
def test_waveform_driver_and_likelihood(backend, precessing):
    _, _, data, psd = fixture_inputs(n=32768, fs=1024)
    p = lsu.ChooseWaveformParams(m1=20*lal.MSUN_SI, m2=12*lal.MSUN_SI,
        deltaT=1./1024, deltaF=1./32, fmin=30., fref=30.,
        approx=lalsim.IMRPhenomXPHM if precessing else lalsim.IMRPhenomXHM,
        s1x=.2 if precessing else 0., s1z=.1, radec=True, theta=.3, phi=.7)
    # PSD/data amplitudes appropriate for the generated signal.
    for det in data:
        data[det].data.data *= 1e-22
        psd[det].data.data *= 1e-44
    args = (lal.LIGOTimeGPS(1000000000+30), .15, p, data, psd, 4, 400.)
    kw = dict(verbose=False, quiet=True, ignore_threshold=None,
              inv_spec_trunc_Q=True, T_spec=.25,
              extra_waveform_kwargs=dict(fd_alignment_postevent_time=2))
    ref = fl.PrecomputeLikelihoodTerms(*args, **kw)
    ctx = FullRatePrecompute(backend, 3)
    actual = fl.PrecomputeLikelihoodTerms(*args, **kw, precompute_backend=backend,
                                          precompute_batch_size=3, precompute_context=ctx)
    for det in data:
        assert_products((ref[1][det], ref[2][det], ref[3][det]),
                        (actual[1][det], actual[2][det], actual[3][det]))
        for mode in ref[0][det]:
            ts = ref[3][det][mode]
            times = float(ts.epoch) + np.arange(3, 10)*ts.deltaT
            np.testing.assert_allclose(ref[0][det][mode](times), actual[0][det][mode](times),
                                       rtol=1e-10, atol=1e-11)
    assert actual[4] == pytest.approx(ref[4], rel=1e-11)
    p.tref = args[0]
    p.dist = 100e6*lal.PC_SI
    p.incl, p.psi, p.phiref = .5, .2, .4
    # Check the physical consumer, including interference between precessing modes.
    a = fl.FactoredLogLikelihood(p, ref[3], ref[0], ref[1], ref[2], 4)
    b = fl.FactoredLogLikelihood(p, actual[3], actual[0], actual[1], actual[2], 4)
    assert b == pytest.approx(a, rel=1e-10, abs=1e-10)
    xp = ctx.xp
    pvec = p.manual_copy()
    for name in ("phi", "theta", "psi", "incl", "phiref", "dist"):
        setattr(pvec, name, xp.asarray([getattr(p, name)]*3))
    pvec.phi += xp.asarray([0., .1, -.1])
    pvec.tref = float(p.tref)
    tvals = xp.arange(-32, 32)*p.deltaT
    def packed_likelihood(products):
        nk, rho, u, v, epochs = {}, {}, {}, {}, {}
        for det in data:
            packed = fl.PackLikelihoodDataStructuresAsArrays(products[3][det].keys(),
                None, products[3][det], products[1][det], products[2][det])
            nk[det], rho[det], u[det], v[det], epochs[det] = [
                xp.asarray(packed[i]) for i in [0, 5, 3, 4, 7]]
        return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(tvals, pvec,
            nk, rho, u, v, epochs, Lmax=4, xpy=xp, time_interp="cubic")
    np.testing.assert_allclose(ctx._host(packed_likelihood(ref)),
                               ctx._host(packed_likelihood(actual)), rtol=1e-10, atol=1e-10)


def test_invalid_backend_and_batch():
    with pytest.raises(ValueError):
        FullRatePrecompute("typo")
    with pytest.raises(ValueError):
        FullRatePrecompute("numpy", 0)


@pytest.mark.parametrize("backend", backends())
def test_replaced_identical_data_does_not_borrow_old_epoch(backend):
    h, c, original, psd = fixture_inputs()
    ctx = FullRatePrecompute(backend)
    ctx.prepare(original, psd, 30., 400., 512.)
    replacement = {k: lal.CutCOMPLEX16FrequencySeries(v, 0, v.data.length)
                   for k, v in original.items()}
    ctx.prepare(replacement, psd, 30., 400., 512.)
    original["H1"].epoch += 10.
    ctx.prepare(replacement, psd, 30., 400., 512.)
    fresh = FullRatePrecompute(backend)
    fresh.prepare(replacement, psd, 30., 400., 512.)
    assert_products(fresh.compute(h, c, "H1", 0, 64), ctx.compute(h, c, "H1", 0, 64))


@pytest.mark.parametrize("backend", backends())
def test_mutable_analytic_psd(backend):
    h, c, data, _ = fixture_inputs()
    class PSD:
        scale = 1.
        def __call__(self, f):
            return self.scale*(1+f*f)
    psd = PSD()
    psds = {det: psd for det in data}
    ctx = FullRatePrecompute(backend)
    ctx.prepare(data, psds, 30., 400., 512., analyticPSD_Q=True)
    psd.scale = 1.7
    ctx.prepare(data, psds, 30., 400., 512., analyticPSD_Q=True)
    fresh = FullRatePrecompute(backend)
    fresh.prepare(data, psds, 30., 400., 512., analyticPSD_Q=True)
    assert_products(fresh.compute(h, c, "H1", 0, 64), ctx.compute(h, c, "H1", 0, 64))


def test_device_switch_requires_prepare(monkeypatch):
    h, c, data, psd = fixture_inputs()
    ctx = FullRatePrecompute("numpy")
    device = [0]
    monkeypatch.setattr(ctx, "_device", lambda: device[0])
    ctx.prepare(data, psd, 30., 400., 512.)
    device[0] = 1
    with pytest.raises(ValueError, match="device"):
        ctx.compute(h, c, "H1", 0, 64)
    ctx.prepare(data, psd, 30., 400., 512.)
    ctx.compute(h, c, "H1", 0, 64)


@pytest.mark.parametrize("backend", backends())
def test_q_rejects_first_mode_grid_mismatch(backend):
    h, c, data, psd = fixture_inputs()
    ctx = FullRatePrecompute(backend)
    ctx.prepare(data, psd, 30., 400., 512.)
    h[next(iter(h))].deltaF += .5*lsu.TOL_DF
    with pytest.raises(AssertionError):
        fl.ComputeModeIPTimeSeries(h, data["H1"], psd["H1"], 30., 400., 512., 0, 64)
    with pytest.raises(ValueError, match="grid"):
        ctx.compute(h, c, "H1", 0, 64)


@pytest.mark.parametrize("backend", backends())
def test_non_power_two_grid(backend):
    h, c, data, psd = fixture_inputs(n=8190, fs=1000)
    df = 1000./8190
    ctx = FullRatePrecompute(backend)
    ctx.prepare(data, psd, 30., 400., 500.)
    ref = (fl.ComputeModeCrossTermIP(h, h, psd["H1"], 30., 400., 500., df, verbose=False),
           fl.ComputeModeCrossTermIP(c, h, psd["H1"], 30., 400., 500., df, verbose=False),
           fl.ComputeModeIPTimeSeries(h, data["H1"], psd["H1"], 30., 400., 500., -17, 64))
    assert_products(ref, ctx.compute(h, c, "H1", -17, 64))
