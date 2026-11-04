"""The four-axis policy's peak-local time reserve, as wired.

The rule (one commensurate lattice, fixed shape, coarser check, zero-weight
repeats), the width prediction from the row's tables, the scheme pair table
and its refusals, the angular-kernel seam, agreement of the peak-local reserve
with the whole-window refined reserve and with an analytic fine reference on
two synthetic tables (the policy tests' three-harmonic table, and a
Gaussian-envelope carrier whose peak is 0.16 samples wide), the escalation
report when the width cannot be predicted, fail-closed behaviour when the
escalations are exhausted, and the driver options.  Evidence:
DESIGN_direct_marginalization_policy.md, "Peak-local time reserve".

TEST-TIMING: five policy evaluations on 33-sample tables; the carrier
fixture's whole-window reserve escalates to refine 32.
"""
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                             # noqa: E402

from RIFT.likelihood.jax_ile import anglemarg as AM                 # noqa: E402
from RIFT.likelihood.jax_ile import core as _core                   # noqa: E402
from RIFT.likelihood.jax_ile import direct_marginalization_policy as DP   # noqa: E402
from RIFT.likelihood.jax_ile import peaklocal_time_reserve as PLR   # noqa: E402
from test_direct_marginalization_policy import (                    # noqa: E402
    _guarded_problem, _fake_data, _install_tables, _fine_reference, _grid,
    _GUARD, _N)
from test_angle_marg_exact import INTERP                            # noqa: E402

AMP_SIZING = 40.0
# The carrier drives the phi exponent to x A = 160, so its exact kernel is
# sized for that: an undersized phi grid leaves a ripple of period
# 1/(n_phi f_c) in lnL(t) that a coarser check rule aliases (measured
# 1.85e-3 nat at amp_sizing 40).
AMP_SIZING_CARRIER = 200.0


# ------------------------------------------------------------------ the rule
def _rule(centres, widths, live, sigma_t, n_target=40, n_fine=9, scan_refine=2,
          dt_scale=0.5, mult=1):
    return PLR.peaklocal_time_rule(jnp.asarray(centres, dtype=jnp.float64),
                                   jnp.asarray(widths, dtype=jnp.float64),
                                   jnp.asarray(live, dtype=bool), n_target,
                                   dt_scale, sigma_t_samples=sigma_t,
                                   n_fine=n_fine, scan_refine=scan_refine,
                                   fine_refine_multiplier=mult)


def test_rule_is_one_commensurate_lattice_of_fixed_shape():
    n_target, n_fine, scan = 40, 9, 2
    expected = PLR.peaklocal_rule_size(n_target, 3, n_fine, scan)
    narrow = _rule([10.3, 25.0, 3.0], [0.1, 0.4, 2.0], [True, True, False],
                   sigma_t=0.1, n_target=n_target, n_fine=n_fine, scan_refine=scan)
    wide = _rule([2.0, 30.0, 36.0], [3.0, 0.05, 1.0], [True, False, True],
                 sigma_t=4.0, n_target=n_target, n_fine=n_fine, scan_refine=scan)
    # sigma 0.1 at 3 nodes per sigma against a 0.5 scan needs m = 15; the
    # wide prediction (sigma 4) is capped by its narrowest live maximum
    # (width 1.0), which needs m = 2.
    assert int(narrow["fine_refine"]) == 15 and int(wide["fine_refine"]) == 2
    for r in (narrow, wide):
        assert (int(r["nodes"].size), int(r["check_nodes"].size)) == expected
        nodes, check = np.asarray(r["nodes"]), np.asarray(r["check_nodes"])
        h = float(r["fine_spacing_samples"])
        assert np.allclose(nodes / h, np.round(nodes / h), atol=1e-9)
        assert np.allclose(check / h, np.round(check / h), atol=1e-9)
        assert np.all(np.diff(nodes) >= 0.0) and np.all(np.diff(check) >= 0.0)
        assert nodes[0] == 0.0 and nodes[-1] == n_target - 1
        assert check[0] == nodes[0] and check[-1] == nodes[-1]
        assert np.max(np.diff(check)) > np.max(np.diff(nodes))
        assert np.max(np.diff(nodes)) < 1.0
        w, wc = np.asarray(r["weights"]), np.asarray(r["check_weights"])
        assert np.all(w >= 0.0) and np.all(wc >= 0.0)
        # Trapezoid weights telescope: the rule integrates the whole window
        # whatever the block positions.
        assert np.isclose(w.sum(), (n_target - 1) * 0.5)
        assert np.isclose(wc.sum(), (n_target - 1) * 0.5)
        assert int(r["n_live_blocks"]) == 2
    # The narrow rule's live blocks span (n_fine - 1) h around their centres
    # and no more; the dead block's nodes are scan nodes (zero extra weight).
    nodes = np.asarray(narrow["nodes"])
    h = float(narrow["fine_spacing_samples"])
    assert float(narrow["block_span_samples"]) == pytest.approx((n_fine - 1) * h)
    inside = nodes[(nodes > 10.3 - 0.5) & (nodes < 10.3 + 0.5)]
    assert len(inside) >= n_fine
    assert float(narrow["sigma_t_located_samples"]) == pytest.approx(0.1)
    assert float(narrow["sigma_t_used_samples"]) == pytest.approx(0.1)
    assert float(narrow["first_block_centre_samples"]) == pytest.approx(10.3)
    # The lattice is never coarser than the narrowest located maximum.
    coarse_pred = _rule([10.3, 25.0, 3.0], [0.02, 0.4, 2.0], [True, True, False],
                        sigma_t=0.1, n_target=n_target, n_fine=n_fine, scan_refine=scan)
    assert int(coarse_pred["fine_refine"]) == 75
    # A tier doubles the lattice at fixed span.
    tier = _rule([10.3, 25.0, 3.0], [0.1, 0.4, 2.0], [True, True, False],
                 sigma_t=0.1, n_target=n_target, n_fine=2 * n_fine - 1,
                 scan_refine=scan, mult=2)
    assert int(tier["fine_refine"]) == 30
    assert float(tier["block_span_samples"]) == pytest.approx(
        float(narrow["block_span_samples"]))


def test_rule_refuses_what_its_check_rule_cannot_express():
    with pytest.raises(ValueError):
        PLR.validate_peaklocal_rule_arguments(40, 8, 2)      # even fine count
    with pytest.raises(ValueError):
        PLR.validate_peaklocal_rule_arguments(40, 9, 3)      # odd scan
    with pytest.raises(ValueError):
        _rule([1.0, 2.0], [1.0, 1.0], [True, True], sigma_t=1.0, mult=0)


def test_an_unpredictable_width_degenerates_to_the_scan():
    r = _rule([10.0, 20.0], [np.inf, np.inf], [False, False], sigma_t=float("nan"))
    assert int(r["fine_refine"]) == 1 and not bool(r["prediction_finite"])
    nodes = np.asarray(r["nodes"])
    assert np.allclose(nodes * 2.0, np.round(nodes * 2.0))


# --------------------------------------------------------- the prediction
def test_predict_time_rule_compares_the_two_node_counts():
    narrow = PLR.predict_time_rule(163.0, 0.01557, 614, n_blocks=8, n_fine=49,
                                   scan_refine=2)
    assert narrow["sigma_t_samples"] == pytest.approx(
        1.0 / (2 * np.pi * 163.0 * 0.01557))
    assert narrow["whole_window_nodes_needed"] > 2 * narrow["peaklocal_nodes"]
    assert narrow["prefer_peaklocal"]
    broad = PLR.predict_time_rule(8.0, 0.01557, 614, n_blocks=8, n_fine=49,
                                  scan_refine=2)
    assert not broad["prefer_peaklocal"]
    assert PLR.predict_time_rule(float("nan"), 0.01, 100, n_blocks=2, n_fine=9,
                                 scan_refine=2)["fine_refine"] == 1


# ---------------------------------------------------------- the carrier
# f_c 0.1, not 0.2: the Gaussian envelope (tau 2) must be band-limited on the
# stored grid, and at 0.2 its amplitude 8e-4 past Nyquist aliased the
# reflected primitive 0.018 nat away from the analytic table (measured); at
# 0.1 the leak is 3e-6.
_CARRIER = dict(amp=20.0, B=10.0, tau=2.0, f_c=0.1, t0=16.37)


def _carrier_table(n, guard, t):
    c = _CARRIER
    C_A = np.zeros((3, 3, t.size), dtype=np.complex128)
    C_A[2, 1] = (c["amp"] * np.exp(-0.5 * ((t - c["t0"]) / c["tau"]) ** 2)
                 * np.exp(2j * np.pi * c["f_c"] * (t - c["t0"])))
    C_B = np.zeros((5, 5), dtype=np.complex128)
    C_B[0, 2] = c["B"]
    return C_A, C_B


def _carrier_problem(n, guard):
    """A (2, +-2)-like carrier with a Gaussian envelope: the angular maximum
    is 2 amp g(t), so lnL(t) peaks at 2 amp^2 / B with rho^2 = 4 amp^2 / B
    and exp(lnL) has width tau / rho, here 0.159 samples."""
    support = np.arange(-guard, n + guard, dtype=float)
    return _carrier_table(n, guard, support)


def _carrier_fine_reference(data, x_grid, log_w, refine=32):
    n = int(data.npts)
    t = np.arange((n - 1) * refine + 1, dtype=float) / float(refine)
    C_A, C_B = _carrier_table(n, 0, t)
    lnL_t = AM.coefficient_table_distphipsimarg_exact(
        C_A, C_B, x_grid, log_w, amp_sizing=AMP_SIZING_CARRIER, dense_chunk=8,
        grid_block=32)
    w = jnp.asarray(_core._simpson_weights(t.size, data.deltaT / refine))
    return float(_core._time_marginalize(lnL_t, w)[0])


def _carrier_sigma_t():
    """Width of the phi-marginalized peak (the envelope), and rho."""
    c = _CARRIER
    rho = 2.0 * c["amp"] / np.sqrt(c["B"])
    return c["tau"] / rho, rho


def _carrier_fixed_angle_sigma():
    """Width from the curvature of the FIXED-angle field at its maximum,
    which carries the carrier: ``1 / (rho sqrt(omega^2 + 1/tau^2))``.  This
    is what the locator measures and the narrowest structure the primitive
    has; the phi-marginalized peak is wider by the same ratio the raw and
    central moments differ by."""
    c = _CARRIER
    _, rho = _carrier_sigma_t()
    omega = 2.0 * np.pi * c["f_c"]
    return 1.0 / (rho * np.sqrt(omega ** 2 + 1.0 / c["tau"] ** 2))


def test_the_prediction_reads_rho_and_the_bandwidth_off_the_table():
    C_A, C_B = _carrier_problem(_N, _GUARD)
    sigma_t_true, rho_true = _carrier_sigma_t()
    rho = float(PLR.row_amplitude(C_A, C_B, _GUARD))
    sigma_f = float(PLR.table_bandwidth_cycles(C_A, _GUARD))
    # rho reads the envelope at the stored samples; the peak sits 0.37 of a
    # sample off the lattice, so the triangle bound is 2% low here.
    assert rho == pytest.approx(rho_true, rel=0.05)
    # Raw two-sided rms frequency: the carrier plus the Gaussian envelope's
    # power spread 1 / (2 sqrt2 pi tau), in quadrature.
    expected_f = np.hypot(_CARRIER["f_c"], 1.0 / (2 * np.sqrt(2) * np.pi * _CARRIER["tau"]))
    assert sigma_f == pytest.approx(expected_f, rel=0.05)
    sigma_t = float(PLR.predicted_width_samples(rho, sigma_f))
    # The prediction is the narrowest peak the primitive can make: below the
    # envelope's true width here, and within the factor-four band.
    assert sigma_t_true / 4.0 < sigma_t < sigma_t_true
    # The three-harmonic table's only frequency is its cosine, 1/32 cycles
    # per sample diluted by the constant lanes; the raw moment reads it and
    # predicts a width within the consistency band of the measured 0.75
    # samples (the prediction is the narrowest peak, so it sits below).
    C_A3, C_B3, _ = _guarded_problem(_N, _GUARD)
    sigma_f3 = float(PLR.table_bandwidth_cycles(C_A3, _GUARD))
    assert 0.005 < sigma_f3 < 1.0 / 32.0
    sigma_t3 = float(PLR.predicted_width_samples(
        float(PLR.row_amplitude(C_A3, C_B3, _GUARD)), sigma_f3))
    assert 0.75 / 4.0 < sigma_t3 <= 0.75 * 1.05
    assert not np.isfinite(float(PLR.predicted_width_samples(10.0, 0.0)))


# ------------------------------------------------------- config and the seam
def test_scheme_pairs_and_defaults_leave_the_shipped_reserve_alone():
    assert DP.PolicyConfig().reserve_scheme == "exact"
    assert DP.PolicyConfig().max_time_nodes >= 2
    assert DP.reserve_pair("exact") == ("exact", "window")
    assert DP.reserve_pair("laplace") == ("laplace", "window")
    assert DP.reserve_pair("peaklocal") == ("laplace", "peaklocal")
    assert DP.reserve_pair("peaklocal-exact") == ("exact", "peaklocal")
    assert "peaklocal-exact" not in DP.RESERVE_SCHEME_CHOICES
    with pytest.raises(ValueError) as err:
        DP.reserve_pair("auto")
    assert "predict_reserve_pair" in str(err.value)
    with pytest.raises(ValueError):
        DP.reserve_pair("peak-local")
    with pytest.raises(ValueError):
        DP.validate_policy_config(DP.PolicyConfig(reserve_scheme="cover"))
    with pytest.raises(ValueError):
        DP.validate_policy_config(DP.PolicyConfig(
            reserve_scheme="peaklocal-exact", reserve_peaklocal_fine_nodes=48))
    with pytest.raises(ValueError):
        DP.validate_policy_config(DP.PolicyConfig(
            reserve_scheme="peaklocal-exact",
            reserve_peaklocal_sigma_t_override_samples=0.0))
    with pytest.raises(ValueError):
        DP.validate_policy_config(DP.PolicyConfig(max_time_nodes=1))


def test_laplace_kernel_is_resolved_from_anglemarg_or_refused(monkeypatch):
    x_grid, log_w = _grid(16)
    assert DP.resolve_reserve_angular_kernel(
        "exact", x_grid, log_w, amp_sizing=AMP_SIZING, m_max=2, dense_chunk=8,
        grid_block=32) is None
    monkeypatch.delattr(AM, DP._LAPLACE_TABLE_KERNEL, raising=False)
    with pytest.raises(ValueError) as err:
        DP.resolve_reserve_angular_kernel(
            "laplace", x_grid, log_w, amp_sizing=AMP_SIZING, m_max=2,
            dense_chunk=8, grid_block=32)
    assert DP._LAPLACE_TABLE_KERNEL in str(err.value)
    with pytest.raises(ValueError):
        DP.validate_policy_config(DP.PolicyConfig(reserve_scheme="peaklocal"))
    seen = {}

    def fake(table, norm_table, xg, lw, *, amp_sizing, m_max):
        # The Laplace kernel's signature: no dense_chunk/grid_block (those are
        # the exact kernel's); a fake that accepted them hid the seam until
        # the first real rung-652 evaluation (2026-09-09).
        seen.update(amp_sizing=amp_sizing, m_max=m_max, n=int(xg.shape[0]))
        return jnp.zeros(table.shape[-1:])
    monkeypatch.setattr(AM, DP._LAPLACE_TABLE_KERNEL, fake, raising=False)
    kernel = DP.resolve_reserve_angular_kernel(
        "laplace", x_grid, log_w, amp_sizing=AMP_SIZING, m_max=2,
        dense_chunk=8, grid_block=32)
    out = kernel(jnp.zeros((3, 3, 7), dtype=jnp.complex128),
                 jnp.zeros((5, 5), dtype=jnp.complex128))
    assert out.shape == (7,)
    assert seen == dict(amp_sizing=AMP_SIZING, m_max=2, n=16)


def test_laplace_kernel_runs_through_the_real_anglemarg_function():
    """The resolved Laplace kernel must call anglemarg's REAL function with
    keywords it accepts: shape (npts,), finite, and within the psi-Laplace
    approximation of the exact kernel on the carrier tables."""
    x_grid, log_w = _grid(16)
    n, guard = 8, 4
    C_A, C_B = _carrier_problem(n, guard)
    kernel = DP.resolve_reserve_angular_kernel(
        "laplace", x_grid, log_w, amp_sizing=AMP_SIZING_CARRIER, m_max=2,
        dense_chunk=8, grid_block=32)
    lap = np.asarray(kernel(jnp.asarray(C_A), jnp.asarray(C_B)))
    ex = np.asarray(AM.coefficient_table_distphipsimarg_exact(
        C_A, C_B, x_grid, log_w, amp_sizing=AMP_SIZING_CARRIER, m_max=2,
        dense_chunk=8, grid_block=32))
    # Both kernels return the batched (B, npts) contract, B = 1 here.
    assert lap.shape == ex.shape == (1, C_A.shape[-1])
    assert np.all(np.isfinite(lap))
    lap, ex = lap[0], ex[0]
    peak = int(np.argmax(ex))
    assert abs(float(lap[peak] - ex[peak])) < 0.5 * abs(float(ex[peak]))
    assert abs(int(np.argmax(lap)) - peak) <= 1


def test_locator_search_phi_grid_is_sized_for_the_amplitude():
    """At rho ~ 630 the 64-node phi grid's ripple, about rho^2 (pi/64)^2 =
    975 nat, exceeds the profile's change across one search cell, here
    (rho^2/2)(0.125/tau)^2 = 49 nat with tau = 8, so the search maximum lands
    on the wrong cell and the polish cannot reach the peak (measured on the
    rung-652 production row, 2026-09-09).  With the policy's 4096 nodes the
    ripple is 0.24 nat and the locator lands within the marginal peak's
    width tau / rho of the carrier's centre, at the profile's true maximum
    rho^2 / 2."""
    n, guard = 40, 8
    c = dict(_CARRIER, amp=1000.0, tau=8.0, t0=19.37)
    support = np.arange(-guard, n + guard, dtype=float)
    C_A = np.zeros((3, 3, support.size), dtype=np.complex128)
    C_A[2, 1] = (c["amp"] * np.exp(-0.5 * ((support - c["t0"]) / c["tau"]) ** 2)
                 * np.exp(2j * np.pi * c["f_c"] * (support - c["t0"])))
    C_B = np.zeros((5, 5), dtype=np.complex128)
    C_B[0, 2] = c["B"]
    rho = 2.0 * c["amp"] / np.sqrt(c["B"])
    width = c["tau"] / rho
    x_lo, x_hi = 0.0, 1.0e6
    kw = dict(n_candidates=2, search_refine=8, angular_lattice=8,
              newton_steps=8, newton_step_max=1.0)
    sized = PLR.locate_time_maxima(C_A, C_B, guard, n, x_lo, x_hi, n_phi=4096, **kw)
    k = int(np.argmax(np.where(np.asarray(sized["live"]), np.asarray(sized["values"]), -np.inf)))
    assert abs(float(sized["centres"][k]) - c["t0"]) < width
    assert abs(float(sized["values"][k]) - 0.5 * rho ** 2) < 1.0
    # The 64-node grid's failure is not pinned here: on this one-harmonic
    # carrier the ripple's phase can favour the right cell by chance.  The
    # production row is the evidence (DESIGN, rung 652 row 0).
    cfg = DP.PolicyConfig(reserve_scheme="peaklocal-exact")
    assert int(cfg.reserve_peaklocal_search_phi_nodes) == 4096
    assert rho ** 2 * (np.pi / cfg.reserve_peaklocal_search_phi_nodes) ** 2 < 1.0


# ------------------------------------------------ agreement on the reserve
def _policy(monkeypatch, tables, cfg, amp_sizing=AMP_SIZING):
    monkeypatch.setattr(_core, "_DISTMARG_GH_N", 0)
    _install_tables(monkeypatch, tables, _GUARD)
    data = _fake_data(_N)
    x_grid, log_w = _grid()
    value, ledger = DP.fused_log_likelihood_four_axis_policy(
        data, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), x_grid, log_w,
        interp=INTERP, amp_sizing=amp_sizing, config=cfg, return_ledger=True)
    L = {k: np.asarray(v)[0] for k, v in ledger.items()}
    return float(value[0]), L, data, x_grid, log_w


_WINDOW = DP.PolicyConfig(time_guard=_GUARD, reserve_time_refine=4,
                          max_modes=1, enriched_max_modes=1)
_PEAK = _WINDOW._replace(reserve_scheme="peaklocal-exact")


def _assert_warranted_peaklocal(L, cfg):
    assert not L["accepted_local"]
    assert L["reserve_executed"] and L["reserve_time_rule_peaklocal"]
    assert L["reserve_time_check_rule_internal"]
    assert L["reserve_time_resolution_validated"]
    assert L["reserve_time_guard_validated"]
    assert L["reserve_time_warranted"] and L["usable"] and L["reconciles"]
    assert int(L["reserve_time_refine_used"]) == 0
    assert int(L["reserve_peaklocal_live_blocks"]) >= 1
    expected, _ = PLR.peaklocal_rule_size(
        _N, cfg.reserve_peaklocal_blocks,
        int(L["reserve_peaklocal_fine_nodes_used"]),
        cfg.reserve_peaklocal_scan_refine, n_scan=cfg.reserve_peaklocal_scan_nodes)
    assert int(L["reserve_time_points"]) == expected
    # Support-limited: the scan spans the hull of the live maxima plus the
    # printed margin, not the window, and the omitted mass is bounded and
    # charged through the cropped-cover warrant.
    assert L["reserve_time_cropped_cover_warranted"]
    assert L["reserve_time_tail_ok"]
    assert np.isfinite(float(L["reserve_peaklocal_outside_log_bound"]))
    assert float(L["reserve_peaklocal_scan_lo_samples"]) >= 0.0
    assert float(L["reserve_peaklocal_scan_hi_samples"]) <= _N - 1


def test_carrier_peak_is_predicted_sized_and_matches_the_references(monkeypatch):
    """The narrow carrier: the whole-window rule needs refine 32 to resolve a
    0.16-sample peak, the peak-local rule resolves it at its FIRST tier
    (no escalation), the prediction agrees with the plan's Newton width, and
    both reserves match the analytic fine reference."""
    tables = ([_carrier_problem(_N, _GUARD)[0]], _carrier_problem(_N, _GUARD)[1])
    v_w, L_w, data, x_grid, log_w = _policy(monkeypatch, tables, _WINDOW,
                                            amp_sizing=AMP_SIZING_CARRIER)
    v_p, L_p, _, _, _ = _policy(monkeypatch, tables, _PEAK,
                                amp_sizing=AMP_SIZING_CARRIER)
    fine = _carrier_fine_reference(data, x_grid, log_w)
    sigma_t_true, rho_true = _carrier_sigma_t()
    assert L_w["reserve_time_warranted"] and not L_w["reserve_time_rule_peaklocal"]
    assert int(L_w["reserve_time_refine_used"]) >= 8
    _assert_warranted_peaklocal(L_p, _PEAK)
    assert int(L_p["reserve_escalations"]) == 0
    assert int(L_p["reserve_peaklocal_fine_nodes_used"]) == _PEAK.reserve_peaklocal_fine_nodes
    assert L_p["reserve_peaklocal_prediction_finite"]
    assert L_p["reserve_peaklocal_prediction_consistent"]
    assert float(L_p["reserve_peaklocal_rho_pred"]) == pytest.approx(rho_true, rel=0.02)
    assert float(L_p["reserve_peaklocal_rho_bound"]) >= 0.95 * rho_true
    assert L_p["reserve_time_focus_ok"]
    assert float(L_p["reserve_time_focus_offset_samples"]) < 0.05
    pred = float(L_p["reserve_peaklocal_sigma_t_pred_samples"])
    assert sigma_t_true / 4.0 < pred < sigma_t_true
    # The locator measures the ENVELOPE's curvature (what the marginalized
    # reserve integrates), so its width is the phi-marginalized peak's, wider
    # than the raw-moment prediction by the carrier-to-envelope ratio.
    located = float(L_p["reserve_peaklocal_sigma_t_located_samples"])
    assert located == pytest.approx(sigma_t_true, rel=0.05)
    assert float(L_p["reserve_peaklocal_sigma_t_used_samples"]) == pytest.approx(
        min(pred, located))
    assert not np.isfinite(float(L_p["reserve_peaklocal_sigma_f_q_cycles"]))
    assert int(L_p["reserve_peaklocal_fine_refine"]) >= 8
    assert abs(float(L_p["reserve_peaklocal_first_block_centre_samples"])
               - _CARRIER["t0"]) < 1e-3
    # On this 33-sample toy window the whole-window rule at refine 8 is
    # cheaper than the fixed 357; the advantage is a production-window
    # statement, made through the same predictor the selector reads.
    expected_pl, _ = PLR.peaklocal_rule_size(_N, _PEAK.reserve_peaklocal_blocks,
                                             _PEAK.reserve_peaklocal_fine_nodes,
                                             _PEAK.reserve_peaklocal_scan_refine,
                                             n_scan=_PEAK.reserve_peaklocal_scan_nodes)
    assert int(L_p["reserve_time_points"]) == expected_pl
    production = PLR.predict_time_rule(
        rho_true, float(L_p["reserve_peaklocal_sigma_f_table_cycles"]), 614,
        n_blocks=_PEAK.reserve_peaklocal_blocks,
        n_fine=_PEAK.reserve_peaklocal_fine_nodes,
        scan_refine=_PEAK.reserve_peaklocal_scan_refine,
        n_scan=_PEAK.reserve_peaklocal_scan_nodes)
    assert production["prefer_peaklocal"]
    # The count does not grow with the window: 33 samples or 614.
    assert production["peaklocal_nodes"] == expected_pl
    # The scan hull contains the carrier's maximum and is narrower than the
    # window by the margin rule (16 sigma each side of one maximum).
    assert float(L_p["reserve_peaklocal_scan_lo_samples"]) < _CARRIER["t0"] < float(
        L_p["reserve_peaklocal_scan_hi_samples"])
    assert (float(L_p["reserve_peaklocal_scan_hi_samples"])
            - float(L_p["reserve_peaklocal_scan_lo_samples"])) < _N - 1
    assert abs(v_p - v_w) <= float(_PEAK.total_value_error_budget_nats), (v_p, v_w)
    assert abs(v_p - fine) <= 2.0e-3, (v_p, fine)
    assert abs(v_w - fine) <= 2.0e-3, (v_w, fine)


def test_the_locator_finds_the_carrier_maximum_from_the_primitive():
    """No plan involved: the search grid, the dense phi maximization and the
    parabolic polish put the first block on the analytic maximum with the
    analytic envelope width, and the far candidates are dead."""
    C_A, C_B = _carrier_problem(_N, _GUARD)
    x_grid, _ = _grid(64)
    found = PLR.locate_time_maxima(
        C_A, C_B, _GUARD, _N, float(np.min(x_grid)), float(np.max(x_grid)),
        n_candidates=4, search_refine=8, angular_lattice=8)
    sigma_t_true, rho_true = _carrier_sigma_t()
    live = np.asarray(found["live"])
    centres = np.asarray(found["centres"])
    widths = np.asarray(found["widths"])
    assert live[0]
    assert abs(centres[0] - _CARRIER["t0"]) < 2e-3
    assert widths[0] == pytest.approx(sigma_t_true, rel=0.05)
    # The fixed-angle field's own curvature width is narrower by the carrier
    # ratio; the envelope is what the marginalized reserve integrates.
    assert widths[0] > 1.5 * _carrier_fixed_angle_sigma()
    assert float(np.asarray(found["values"])[0]) == pytest.approx(
        0.5 * rho_true ** 2, rel=0.05)
    assert int(found["n_search_nodes"]) == (_N - 1) * 8 + 1
    # rho from the profile maximum, and the angles it was found at.
    assert float(found["rho_located"]) == pytest.approx(rho_true, rel=0.02)
    assert np.isfinite(float(found["phi"][0])) and np.isfinite(float(found["u"][0]))


def test_a_misplaced_block_fails_the_focus_certificate(monkeypatch):
    """The rule and its check share the block, so a block off the maximum
    agrees with itself (measured 0.22 nat, warranted, on a rung-160 row).
    The kernel's focus certificate compares the evaluated argmax with the
    block centre; here the locator is forced 1.2 samples off."""
    real = PLR.locate_time_maxima

    def shifted(*args, **kwargs):
        found = real(*args, **kwargs)
        found = dict(found)
        found["centres"] = found["centres"] + 1.2
        return found
    monkeypatch.setattr(PLR, "locate_time_maxima", shifted)
    tables = ([_carrier_problem(_N, _GUARD)[0]], _carrier_problem(_N, _GUARD)[1])
    v, L, _, _, _ = _policy(monkeypatch, tables,
                            _PEAK._replace(reserve_peaklocal_escalations=0),
                            amp_sizing=AMP_SIZING_CARRIER)
    assert L["reserve_executed"] and not L["reserve_time_focus_ok"]
    assert float(L["reserve_time_focus_offset_samples"]) > float(
        L["reserve_peaklocal_focus_half_width_samples"])
    assert not L["reserve_time_warranted"] and not L["usable"]
    assert not np.isfinite(v)


@pytest.mark.parametrize("scale", [1.0, 3.0])
def test_three_harmonic_table_is_sized_from_its_cosine_and_needs_no_escalation(
        monkeypatch, scale):
    """The policy tests' table at two amplitudes (peak 0.75 and 0.24 samples
    wide).  The raw moment of its cosine predicts a width inside the
    consistency band, the first tier resolves the peak (no escalation), and
    the value matches the window reserve and the analytic reference."""
    guarded, C_B, constants = _guarded_problem(_N, _GUARD, scale=scale)
    v_w, L_w, data, x_grid, log_w = _policy(monkeypatch, ([guarded], C_B), _WINDOW)
    v_p, L_p, _, _, _ = _policy(monkeypatch, ([guarded], C_B), _PEAK)
    fine = _fine_reference(constants, C_B, data, x_grid, log_w, AMP_SIZING,
                           scale=scale)
    _assert_warranted_peaklocal(L_p, _PEAK)
    assert L_p["reserve_peaklocal_prediction_finite"]
    assert L_p["reserve_peaklocal_prediction_consistent"]
    assert int(L_p["reserve_escalations"]) == 0
    assert int(L_p["reserve_peaklocal_fine_refine"]) >= (2 if scale > 1 else 1)
    assert abs(v_p - v_w) <= float(_PEAK.total_value_error_budget_nats), (v_p, v_w)
    assert abs(v_p - fine) <= 2.0e-3, (v_p, fine)


def test_exhausted_escalations_fail_closed_and_keep_the_diagnostic(monkeypatch):
    """The carrier with the width overridden 20x too wide: the first tier
    cannot resolve the peak and with no escalation allowed the row is
    unusable, its finite diagnostic kept in the ledger and never selected."""
    tables = ([_carrier_problem(_N, _GUARD)[0]], _carrier_problem(_N, _GUARD)[1])
    sigma_t_true, _ = _carrier_sigma_t()
    # Override the prediction 40x too wide AND blind the locator (one
    # candidate on a native-sample search grid, no Newton polish would still
    # find it, so its width is also overridden by the coarse lattice): the
    # first tier is then the scan alone.
    cfg = _PEAK._replace(reserve_peaklocal_escalations=0,
                         reserve_peaklocal_sigma_t_override_samples=40.0 * sigma_t_true,
                         reserve_peaklocal_blocks=1, reserve_peaklocal_fine_nodes=3)
    v, L, _, _, _ = _policy(monkeypatch, tables, cfg, amp_sizing=AMP_SIZING_CARRIER)
    assert L["reserve_executed"] and not L["reserve_time_warranted"]
    assert not L["usable"] and not np.isfinite(v)
    assert int(L["reserve_escalations"]) == 0
    assert not L["reserve_peaklocal_prediction_consistent"]
    assert np.isfinite(float(L["selected_value"]))


# ------------------------------------------------------------------ driver
def test_driver_offers_the_reserve_scheme_and_the_time_node_knobs():
    import importlib.machinery
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).parents[2] / "bin" / "integrate_likelihood_extrinsic_jax"
    loader = importlib.machinery.SourceFileLoader("_plr_driver", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    drv = importlib.util.module_from_spec(spec)
    loader.exec_module(drv)
    optp = drv.build_parser()
    opts, _ = optp.parse_args(["--direct-marginalization-reserve-scheme", "peaklocal",
                               "--direct-marginalization-max-time-nodes", "256",
                               "--direct-marginalization-peaklocal-escalations", "1"])
    assert opts.direct_marginalization_reserve_scheme == "peaklocal"
    assert opts.direct_marginalization_max_time_nodes == 256
    assert opts.direct_marginalization_peaklocal_escalations == 1
    opts, _ = optp.parse_args([])
    assert opts.direct_marginalization_reserve_scheme == "exact"
    # One definition: the flag's default is the config's, whatever it is.
    assert opts.direct_marginalization_max_time_nodes == DP.PolicyConfig().max_time_nodes
    assert (opts.direct_marginalization_peaklocal_escalations
            == DP.PolicyConfig().reserve_peaklocal_escalations)
    with pytest.raises(SystemExit):
        optp.parse_args(["--direct-marginalization-reserve-scheme", "peak-local"])
    with pytest.raises(SystemExit):
        optp.parse_args(["--direct-marginalization-reserve-scheme", "peaklocal-exact"])
    # Each flag is added once (a merge once left two add_option calls and
    # optparse silently took the last).
    src = path.read_text()
    for flag in ("--direct-marginalization-reserve-scheme",
                 "--direct-marginalization-max-time-nodes",
                 "--direct-marginalization-peaklocal-escalations"):
        assert src.count('add_option("%s"' % flag) == 1, flag
