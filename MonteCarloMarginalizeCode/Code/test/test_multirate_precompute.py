"""PrecomputeLikelihoodTermsMultirate: the two-rate schedule against the one-segment (full-rate,
per-|m|) control on a short O4 BNS with higher modes, and the refusals
(DESIGN_early_time_multirate.md, "Opt-in interface")."""
import json
import numpy as np
import pytest
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood import factored_likelihood_multirate as flm

FS, SEGLEN, FLOW, FMAX, LMAX = 4096, 512, 20., 1700., 4
T_GEO = 1000000000.0
SCHED = dict(fs_e=256, tau_tr=60., guard=4., atten_db=160., late_start=96., late_buffer=128, post=24,
             uv_margin=32., mc_range=[1.1, 1.3])


def _P(s1x=0., mc_scale=1.0):
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 1.40 * mc_scale * lal.MSUN_SI, 1.35 * mc_scale * lal.MSUN_SI
    P.s1x = s1x
    P.fmin, P.fref = FLOW, FLOW
    P.deltaT, P.deltaF = 1. / FS, 1. / SEGLEN
    P.approx = lalsim.IMRPhenomXHM
    P.phi, P.theta, P.tref = 1.0, 0.4, T_GEO
    return P


def _schedule(tmp_path, kind):
    p = tmp_path / ("%s.json" % kind)
    p.write_text(json.dumps(dict(SCHED, kind=kind)))
    return flm.load_schedule(str(p))


@pytest.fixture(scope="module")
def inputs():
    dt, df, N = 1. / FS, 1. / SEGLEN, FS * SEGLEN
    psd = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0), 0., df, lsu.lsu_HertzUnit, N // 2 + 1)
    lalsim.SimNoisePSDaLIGOAdVO4T1800545(psd, 10.)
    v = np.array(psd.data.data); v[v <= 0] = np.inf; psd.data.data = v
    t_det = float(fl.ComputeArrivalTimeAtDetector("H1", 1.0, 0.4, lal.LIGOTimeGPS(T_GEO)))
    rng = np.random.default_rng(3)
    d = lal.CreateCOMPLEX16TimeSeries("d", lal.LIGOTimeGPS(t_det - (SEGLEN - 24.) - 0.0123), 0., dt,
                                      lsu.lsu_DimensionlessUnit, N)
    d.data.data = 1e-22 * rng.standard_normal(N) + 0j
    return dict(data={"H1": lsu.DataFourier(d)}, psd={"H1": psd})


def _run(tmp_path, inputs, kind, P=None):
    return flm.PrecomputeLikelihoodTermsMultirate(lal.LIGOTimeGPS(T_GEO), 0.05, P or _P(), inputs["data"],
                                                  inputs["psd"], LMAX, FMAX, verbose=False, quiet=True,
                                                  schedule=_schedule(tmp_path, kind), skip_interpolation=True)


@pytest.mark.parametrize("approx", [lalsim.IMRPhenomXHM, lalsim.IMRPhenomXPHM])
def test_two_rate_matches_one_segment(tmp_path, inputs, approx):
    p = _P()
    p.approx = approx
    _, U1, V1, rho1, _, _ = _run(tmp_path, inputs, "one_segment", p)
    _, U2, V2, rho2, _, _ = _run(tmp_path, inputs, "two_rate", p)
    sU = max(abs(v) for v in U1["H1"].values())
    for key in U1["H1"]:
        assert abs(U2["H1"][key] - U1["H1"][key]) <= 1e-5 * sU, ("U", key)
        assert abs(V2["H1"][key] - V1["H1"][key]) <= 1e-5 * sU, ("V", key)
    sQ = max(np.max(np.abs(v.data.data)) for v in rho1["H1"].values())
    for k in rho1["H1"]:
        a, b = rho1["H1"][k], rho2["H1"][k]
        assert abs(float(a.epoch) - float(b.epoch)) < 1e-9
        assert a.data.length == b.data.length
        assert np.max(np.abs(a.data.data - b.data.data)) <= 1e-5 * sQ, k


def test_schedule_rejects_short_late_buffer(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(dict(SCHED, kind="two_rate", late_buffer=64)))
    with pytest.raises(ValueError, match="late_buffer"):
        flm.load_schedule(str(p))


@pytest.mark.parametrize("P, msg", [(_P(s1x=0.1), "in-plane"), (_P(mc_scale=1.2), "outside")])
def test_refusals(tmp_path, inputs, P, msg):
    with pytest.raises(ValueError, match=msg):
        _run(tmp_path, inputs, "two_rate", P)


def test_pieces_uv_plan_matches_full_grid_reference(inputs):
    """The per-job plan (band support only, gather indices, matrix-product pair sums) reproduces
    the full-grid selection with explicit pair loops it replaced."""
    P = _P(); df = 1. / SEGLEN; fNyq = FS / 2.
    mc_lo = SCHED["mc_range"][0]
    f22_late = float(flm.f22_newtonian(SCHED["late_start"], mc_lo))
    early = flm._per_m_modes(P, LMAX, SCHED["fs_e"], SEGLEN, FLOW, SCHED["post"])
    late = flm._per_m_modes(P, LMAX, FS, SCHED["late_buffer"], f22_late, SCHED["post"])
    keys = sorted(early)
    sched = flm.multiband_schedule(FLOW, FMAX, SEGLEN, mc_lo, LMAX, extra_s=SCHED["uv_margin"],
                                   top_edge_K=SEGLEN // SCHED["late_buffer"])
    psd = inputs["psd"]["H1"]
    plan = flm.pieces_uv_plan(psd, FLOW, FMAX, fNyq, df, sched, SCHED["fs_e"], f22_late,
                              early[keys[0]].data.length, late[keys[0]].data.length, keys)
    U, V, nk = flm.ComputeModeCrossTermsPieces(early, late, plan)
    # reference: every bin of the full grid, each band, explicit loops
    w = lsu.ComplexIP(FLOW, FMAX, fNyq, df, psd, False, False, 0.).weights2side
    n = len(w); kf_all = np.arange(n) - n // 2
    ne, nl = early[keys[0]].data.length, late[keys[0]].data.length; R = n // nl
    kf = []; g = []
    for j, (lo, hi, K) in enumerate(sched["bands"]):
        sel = np.nonzero(kf_all % K == 0)[0]
        gj = flm._band_window(sched, j, np.abs(kf_all[sel]) * df) * w[sel] * K
        kf.append(kf_all[sel][np.abs(gj) > 0]); g.append(gj[np.abs(gj) > 0])
    kf, g = np.concatenate(kf), np.concatenate(g)

    def val(k, kk):
        ul = (np.abs(kk) * df >= 1.05 * (abs(k[1]) / 2.) * f22_late) & (kk % R == 0)
        return np.where(ul, late[k].data.data[np.clip(kk // R + nl // 2, 0, nl - 1)],
                        early[k].data.data[np.clip(kk + ne // 2, 0, ne - 1)])
    s = max(abs(x) for x in U.values())
    for a in keys:
        for b in keys:
            Ur = 2 * df * np.sum(np.conj(val(a, kf)) * val(b, kf) * g)
            Vr = 2 * df * np.sum(val(a, -kf) * val(b, kf) * g)
            assert abs(U[(a, b)] - Ur) <= 1e-12 * s, ("U", a, b)
            assert abs(V[(a, b)] - Vr) <= 1e-12 * s, ("V", a, b)
