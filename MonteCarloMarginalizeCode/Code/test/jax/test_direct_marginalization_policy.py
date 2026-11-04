"""The opt-in cross-axis policy, as WIRED into the phi/psi-marginalized likelihood.

These test the wiring: that the policy reaches the wrapper and the CLI, refuses
what it cannot honour, converts the local measure to the production convention,
executes a warranted band-limited reserve on decline, keeps every sample, and
records its ledger.  The controller's own numerics are tested in
test_all_axis_peaklocal.py.  Nothing here validates derivatives.
"""
import os
import subprocess
import sys
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile import core as _core
from RIFT.likelihood.jax_ile import direct_marginalization_policy as DP
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood
from test_all_axis_peaklocal import _problem
from test_angle_marg_exact import make_synth, RA, DEC, INCL, INTERP


# ------------------------------------------------------------------ fixtures

def _guarded_problem(n, guard, scale=1.0):
    """The analytic three-harmonic table of test_all_axis_peaklocal with
    ``guard`` primitive-only support samples at each end (the cosine is
    continued analytically, so the guarded reconstruction has a true target)."""
    C_A, C_B, constants = _problem(n)
    support = np.arange(-guard, n + guard, dtype=float)
    guarded = np.zeros(C_A.shape[:-1] + (support.size,), dtype=np.complex128)
    guarded[0, 1] = (constants["k0"]
                     - constants["kt"] * np.cos(2.0 * np.pi * support
                                                / constants["span"]))
    guarded[2, 1] = 0.5 * constants["kp"]
    guarded[0, 0] = 0.5 * constants["ku"]
    guarded[0, 2] = 0.5 * constants["ku"]
    np.testing.assert_allclose(guarded[..., guard:-guard], C_A, atol=1e-14)
    return scale * guarded, scale * scale * C_B, constants


def _fake_data(n, deltaT=1.0 / 4096.0):
    return types.SimpleNamespace(
        npts=n, deltaT=deltaT, distMpcRef=_core.DIST_MPC_REF,
        w_t=jnp.asarray(_core._simpson_weights(n, deltaT)),
        lms=np.asarray([[2, 2], [2, -2]]))


def _install_tables(monkeypatch, tables, guard):
    """Route angle_coefficient_tables to fixed analytic tables, one per row."""
    C_A_rows, C_B = tables
    seen = []

    def fake(data, ra, dec, incl, interp=None, sample_chunk=None, guard=0):
        seen.append(int(guard))
        S = int(np.asarray(ra).shape[0])
        assert S == len(C_A_rows)
        C_A = jnp.asarray(np.stack(C_A_rows, axis=2))          # (KP,KS,S,Nt)
        C_Bb = jnp.broadcast_to(
            jnp.asarray(C_B)[:, :, None, None],
            C_B.shape + (S, C_A.shape[-1]))
        return C_A, C_Bb, dict(m_max=2, nphi_s=9, npsi_s=5, guard=guard,
                               ntime=C_A.shape[-1])

    monkeypatch.setattr(AM, "angle_coefficient_tables", fake)
    return seen


def _production_reference(C_A_target, C_B, data, x_grid, log_w, amp_sizing):
    """What the exact scheme returns for these tables: exact angles on the
    native window, production distance weights, production Simpson time."""
    lnL_t = AM.coefficient_table_distphipsimarg_exact(
        C_A_target, C_B, x_grid, log_w, amp_sizing=amp_sizing,
        dense_chunk=8, grid_block=32)
    return float(_core._time_marginalize(lnL_t, data.w_t)[0])


def _fine_reference(constants, C_B, data, x_grid, log_w, amp_sizing,
                    refine=16, scale=1.0):
    """Independent time reference: the ANALYTIC table evaluated on a
    ``refine``-times finer grid (no reflected primitive involved), exact
    angles, production distance weights, Simpson in seconds."""
    n = int(data.npts)
    t = np.arange((n - 1) * refine + 1, dtype=float) / float(refine)
    C_A = np.zeros((3, 3, t.size), dtype=np.complex128)
    C_A[0, 1] = (constants["k0"]
                 - constants["kt"] * np.cos(2.0 * np.pi * t / constants["span"]))
    C_A[2, 1] = 0.5 * constants["kp"]
    C_A[0, 0] = 0.5 * constants["ku"]
    C_A[0, 2] = 0.5 * constants["ku"]
    lnL_t = AM.coefficient_table_distphipsimarg_exact(
        scale * C_A, C_B, x_grid, log_w, amp_sizing=amp_sizing,
        dense_chunk=8, grid_block=32)
    w = jnp.asarray(_core._simpson_weights(t.size, data.deltaT / refine))
    return float(_core._time_marginalize(lnL_t, w)[0])


_GUARD = 8
_N = 33
_X_RANGE = (0.2, 7.0)


def _grid(n=1024):
    d_min = _core.DIST_MPC_REF / _X_RANGE[1]
    d_max = _core.DIST_MPC_REF / _X_RANGE[0]
    return _core.make_distance_grid(d_min, d_max, n, distMpcRef=_core.DIST_MPC_REF)


# ------------------------------------------------------- choices and refusal

def test_policy_is_opt_in_and_reaches_the_wrapper():
    assert DP.POLICY_DEFAULT == "off"
    assert "auto" in DP.POLICY_CHOICES
    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                       interp=INTERP, angle_marg="exact")
    assert like.direct_marginalization_policy == "off"
    assert like.policy_info is None
    assert like._batched_ledger is None
    with pytest.raises(ValueError, match="direct_marginalization_policy"):
        JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                    interp=INTERP, angle_marg="exact",
                                    direct_marginalization_policy="autp")


@pytest.mark.parametrize("kw, needle", [
    (dict(angle_marg_scheme="laplace", time_quadrature="simpson",
          d_prior="euclidean", dist_grid="uniform"), "exact-angle reserve"),
    (dict(angle_marg_scheme="peak-local", time_quadrature="simpson",
          d_prior="euclidean", dist_grid="uniform"), "exact-angle reserve"),
    (dict(angle_marg_scheme="exact", time_quadrature="bandlimited",
          d_prior="euclidean", dist_grid="uniform"), "time integral"),
    (dict(angle_marg_scheme="exact", time_quadrature="simpson",
          d_prior="uniform", dist_grid="uniform"), "volumetric"),
    (dict(angle_marg_scheme="exact", time_quadrature="simpson",
          d_prior="euclidean", dist_grid="loguniform"), "distance-grid-scheme"),
])
def test_policy_refuses_what_it_cannot_compose(kw, needle):
    with pytest.raises(ValueError, match=needle):
        DP.validate_policy_request("auto", **kw)
    DP.validate_policy_request("off", **kw)        # off composes with anything


def test_wrapper_refuses_laplace_reserve_and_lnLt():
    data = make_synth(scale=2.0)
    with pytest.raises(ValueError, match="exact-angle reserve"):
        JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                    interp=INTERP, angle_marg="laplace",
                                    direct_marginalization_policy="auto")
    with pytest.raises(ValueError, match="volumetric"):
        JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                    interp=INTERP, angle_marg="exact",
                                    d_prior="uniform",
                                    direct_marginalization_policy="auto")


# ------------------------------------------------ measure conversion (accept)

@pytest.mark.parametrize("gh_nodes", [0, 64])
def test_accepted_local_value_lands_in_the_production_convention(
        monkeypatch, gh_nodes):
    """The whole point of the normalization constant: an ACCEPTED local row
    must equal what the exact scheme returns for the same tables, in the
    reserve's units (angles averaged, distance prior normalized, time in
    seconds), on both distance paths."""
    monkeypatch.setattr(_core, "_DISTMARG_GH_N", gh_nodes)
    guarded, C_B, constants = _guarded_problem(_N, _GUARD)
    rows = [guarded, 1.01 * guarded]
    seen = _install_tables(monkeypatch, (rows, C_B), _GUARD)
    data = _fake_data(_N)
    x_grid, log_w = _grid()
    cfg = DP.PolicyConfig(time_guard=_GUARD, reserve_time_refine=4)
    lln, info = DP.policy_log_normalization(data, x_grid, log_w)
    assert info["distance_mode"] == (
        "gh-volumetric" if gh_nodes else "fixed-grid-volumetric")
    assert info["time_weight_scale"] == pytest.approx(1.0)

    value, ledger = DP.fused_log_likelihood_four_axis_policy(
        data, jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), x_grid, log_w,
        interp=INTERP, amp_sizing=40.0, config=cfg, return_ledger=True)
    assert seen == [_GUARD]                       # the guard reached the tables
    assert np.all(np.asarray(ledger["accepted_local"])), {
        k: np.asarray(v) for k, v in ledger.items() if k.startswith("decline")}
    assert np.all(np.asarray(ledger["usable"]))
    assert not np.any(np.asarray(ledger["reserve_executed"]))
    assert np.all(np.asarray(ledger["norm_time_invariant"]))
    assert np.all(np.asarray(ledger["reconciles"]))
    for i, (table, row_scale) in enumerate(zip(rows, (1.0, 1.01))):
        fine = _fine_reference(constants, C_B, data, x_grid, log_w, 40.0,
                               scale=row_scale)
        native = _production_reference(
            table[..., _GUARD:-_GUARD], C_B, data, x_grid, log_w, 40.0)
        # 1e-3 nat is the controller's own budget; the rest is the fixed
        # distance grid's quadrature error, which the 1024-point grid keeps
        # below 1e-4 on this broad fixture.  The native Simpson rule is NOT
        # the reference: this fixture's time peak is ~0.7 samples wide and
        # the production rule misses it by ~0.03 nat, which is the error the
        # composite exists to remove.
        assert abs(float(value[i]) - fine) < 2.0e-3, (
            gh_nodes, i, float(value[i]), fine, native)
        assert abs(native - fine) > 5.0 * abs(float(value[i]) - fine), (
            "the fixture no longer distinguishes the composite from the "
            "native rule", native, fine, float(value[i]))


def test_normalization_constant_is_derived_not_guessed(monkeypatch):
    data = _fake_data(_N)
    x_grid, log_w = _grid()
    monkeypatch.setattr(_core, "_DISTMARG_GH_N", 0)
    total, info = DP.policy_log_normalization(data, x_grid, log_w)
    d = _core.DIST_MPC_REF / np.asarray(x_grid)
    norm = np.sum(d ** 2) * abs(d[1] - d[0])
    expected = (-2.0 * np.log(2.0 * np.pi) + np.log(data.deltaT)
                + 3.0 * np.log(_core.DIST_MPC_REF) - np.log(norm))
    assert total == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="volumetric"):
        DP.policy_log_normalization(data, x_grid, log_w, d_prior="uniform")
    # a non-uniform-in-d grid cannot supply the constant and must say so
    x_lu = jnp.asarray(np.geomspace(0.2, 7.0, 64))
    with pytest.raises(ValueError, match="uniform-in-d"):
        DP.policy_log_normalization(data, x_lu, jnp.zeros(64))


# ---------------------------------------------- decline -> warranted reserve

def test_declined_row_executes_warranted_bandlimited_reserve_and_keeps_sample(
        monkeypatch):
    """Capacity of one for a two-mode table forces the decline; the reserve
    then runs on the refined time rule, its coarser check rule is the native
    production rule, and the selected value is the production value to within
    the resolution tolerance -- so no sample is lost and none is silently
    substituted."""
    monkeypatch.setattr(_core, "_DISTMARG_GH_N", 0)
    guarded, C_B, constants = _guarded_problem(_N, _GUARD)
    _install_tables(monkeypatch, ([guarded], C_B), _GUARD)
    data = _fake_data(_N)
    x_grid, log_w = _grid()
    cfg = DP.PolicyConfig(time_guard=_GUARD, reserve_time_refine=4,
                          max_modes=1, enriched_max_modes=1)
    value, ledger = DP.fused_log_likelihood_four_axis_policy(
        data, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), x_grid, log_w,
        interp=INTERP, amp_sizing=40.0, config=cfg, return_ledger=True)
    L = {k: np.asarray(v)[0] for k, v in ledger.items()}
    assert not L["accepted_local"]
    assert L["reserve_executed"]
    assert L["reserve_uses_bandlimited_time"]
    assert L["reserve_time_check_rule_internal"]
    assert L["reserve_time_resolution_warranted"]
    assert L["reserve_time_resolution_validated"]
    assert L["reserve_time_guard_validated"]
    assert L["reserve_time_warranted"]
    assert L["selected_value_is_warranted_reserve"]
    assert L["usable"] and L["reconciles"] and L["disposition_reconciles"]
    assert int(L["reserve_escalations"]) == 0
    assert int(L["reserve_time_refine_used"]) == cfg.reserve_time_refine
    assert np.isfinite(float(value[0]))
    assert not L["decline_is_waveform_failure"]
    fine = _fine_reference(constants, C_B, data, x_grid, log_w, 40.0)
    assert abs(float(value[0]) - fine) <= 2.0e-3, (float(value[0]), fine)
    assert abs(float(L["reserve_time_check_value"]) - float(value[0])) <= float(
        cfg.total_value_error_budget_nats) + 1e-9
    assert float(L["reserve_time_resolution_error_nats"]) <= float(
        cfg.total_value_error_budget_nats)
    summary = DP.summarize_policy_ledger(ledger)
    assert summary["rows"] == 1 and summary["reserve_executed"] == 1
    assert summary["reserve_warranted"] == 1 and summary["unusable"] == 0
    assert "decline_capacity" in summary["declines"]


def test_ledger_carries_every_named_acceptance_diagnostic(monkeypatch):
    monkeypatch.setattr(_core, "_DISTMARG_GH_N", 0)
    guarded, C_B, _ = _guarded_problem(_N, _GUARD)
    _install_tables(monkeypatch, ([guarded], C_B), _GUARD)
    data = _fake_data(_N)
    x_grid, log_w = _grid(256)
    _, ledger = DP.fused_log_likelihood_four_axis_policy(
        data, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), x_grid, log_w,
        interp=INTERP, amp_sizing=40.0,
        config=DP.PolicyConfig(time_guard=_GUARD), return_ledger=True)
    names = DP.policy_acceptance_diagnostics()
    for group in ("local", "reserve", "declines"):
        for key in names[group]:
            assert key in ledger, (group, key)
            assert np.asarray(ledger[key]).shape == (1,), key
    for key in ("lnL", "selected_value", "reserve_escalations",
                "reserve_time_refine_used"):
        assert key in ledger and np.asarray(ledger[key]).shape == (1,), key
    L = {k: np.asarray(v)[0] for k, v in ledger.items()}
    assert (np.isfinite(L["lnL"]) == bool(L["usable"]))
    with pytest.raises(ValueError, match="time_guard must be >= 2"):
        DP.fused_log_likelihood_four_axis_policy(
            data, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), x_grid, log_w,
            interp=INTERP, amp_sizing=40.0,
            config=DP.PolicyConfig(time_guard=1))
    with pytest.raises(ValueError, match="reserve_time_refine"):
        DP.policy_time_rules(data, 1)
    with pytest.raises(ValueError, match="even"):
        DP.policy_time_rules(data, 3)
    nodes, weights, check_nodes, check_weights = DP.policy_time_rules(data, 2)
    np.testing.assert_allclose(np.asarray(check_weights), np.asarray(data.w_t))
    assert np.max(np.diff(np.asarray(nodes))) == pytest.approx(0.5)
    nodes, weights, check_nodes, check_weights = DP.policy_time_rules(data, 4)
    assert np.max(np.diff(np.asarray(check_nodes))) == pytest.approx(0.5)
    assert float(np.sum(weights)) == pytest.approx(float(np.sum(data.w_t)))


# ---------------------------------------------------- end to end, real tables

@pytest.mark.parametrize("force_decline", [False, True])
def test_wrapper_policy_on_real_synthetic_tables_fails_closed_and_labels(
        force_decline):
    """Real coefficient tables from the accumulate path with a guard, the
    wrapper's own distance grid and amplitude sizing, on the 32-sample
    synthetic window.  The exact lnL(t) on this window peaks at the FIRST
    sample and falls monotonically, so the mass sits on the time boundary;
    the planner's boundary starts are non-stationary and the interior modes
    it keeps integrate to 22.86 nat against an exact 45.5.  Under the
    production operating point the row must therefore decline on
    ``decline_boundary_maximum`` (before this flag existed it ACCEPTED that
    value with every diagnostic passing).  With capacity forced to one mode
    it declines on capacity first.  Either way the window is too short for
    the reflected primitive (the reserve at guard 16 and 8 disagree by
    ~0.06 nat, refine 4 and 2 by ~0.2 nat), so the composite must (a) fail
    the reserve warrant even after escalating to the ceiling, (b) return nan
    while keeping the finite diagnostic in the ledger, (c) say why, and (d)
    count the row as unusable for the run label."""
    from RIFT.likelihood.jax_ile.time_first_peaklocal import (
        _evaluate_time_spectrum, _time_primitive_spectrum)
    data = make_synth(scale=2.0, kappa_boost=10.0)
    kw = dict(nphi=32, npsi=8, interp=INTERP)
    exact = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0,
                                        angle_marg="exact", **kw)
    guard = 16
    cfg = DP.PolicyConfig(time_guard=guard, reserve_time_refine=4,
                          reserve_time_refine_max=8)
    if force_decline:
        cfg = cfg._replace(max_modes=1, enriched_max_modes=1)
    pol = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, angle_marg="exact",
                                      direct_marginalization_policy="auto",
                                      policy_config=cfg, **kw)
    assert pol.direct_marginalization_policy == "auto"
    assert pol.angle_marg_scheme == "exact"
    assert pol.angle_marg_info["direct_marginalization_policy"] == "auto"
    assert pol.policy_info["time_guard"] == guard
    assert "local_log_normalization" in pol.policy_info

    a = np.asarray(exact._batched(jnp.asarray(RA), jnp.asarray(DEC),
                                  jnp.asarray(INCL)))
    b, ledger = pol._batched_ledger(jnp.asarray(RA), jnp.asarray(DEC),
                                    jnp.asarray(INCL))
    b = np.asarray(b)
    c = np.asarray(pol._batched(jnp.asarray(RA), jnp.asarray(DEC),
                                jnp.asarray(INCL)))
    L = {k: np.asarray(v) for k, v in ledger.items()}
    summary = DP.summarize_policy_ledger(ledger)
    np.testing.assert_allclose(b, c, rtol=0.0, atol=1e-12, equal_nan=True)
    assert np.all(L["reconciles"]) and np.all(L["disposition_reconciles"])
    assert np.all(L["norm_time_invariant"])
    assert not np.any(L["decline_is_waveform_failure"])
    tol = float(cfg.total_value_error_budget_nats)
    unusable = ~L["usable"]
    assert summary["unusable"] == int(np.sum(unusable))
    assert summary["nan_rows"] == int(np.sum(unusable))
    # (a)-(c): an unwarranted reserve was escalated to the maximum rule, keeps
    # its finite diagnostic in the ledger, returns nan, and names the failed
    # check; on this window that is the time reconstruction.
    for i in np.flatnonzero(unusable):
        assert L["reserve_executed"][i] and L["reserve_time_failed"][i]
        assert np.isnan(b[i])
        assert np.isfinite(L["selected_value"][i])
        assert np.isfinite(L["reserve_value"][i])
        assert int(L["reserve_time_refine_used"][i]) == cfg.reserve_time_refine_max
        assert int(L["reserve_escalations"][i]) == 1
        assert (float(L["reserve_time_guard_error"][i]) > tol
                or float(L["reserve_time_resolution_error_nats"][i]) > tol), (
            {k: L[k][i] for k in L if k.startswith("reserve_time_")})
    assert np.all(np.isfinite(b[L["usable"]]))
    assert np.any(unusable), ("the 32-sample window became warrantable; "
                              "move this test's fail-closed claim", summary)
    if force_decline:
        assert np.all(L["decline_capacity"]), summary
    else:
        assert np.all(L["decline_boundary_maximum"]), summary
        assert not np.any(L["accepted_local"]), summary
        # the interior-only local diagnostic is the 22.7 nat miss
        assert np.all(L["enriched_value"] < L["reserve_value"] - 10.0), (
            L["enriched_value"], L["reserve_value"])
    # (d) a warranted row, if any, against the 8x refined exact reference.
    usable = L["usable"]
    if np.any(usable):
        C_A, C_B, meta = AM.angle_coefficient_tables(
            data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL), INTERP,
            guard=guard)
        refine = 8
        t = np.arange((data.npts - 1) * refine + 1, dtype=float) / refine
        coeff, freq, off = _time_primitive_spectrum(
            jnp.asarray(C_A).reshape((-1, C_A.shape[-1])), guard)
        fine = _evaluate_time_spectrum(coeff, freq, jnp.asarray(t), off).reshape(
            C_A.shape[:-1] + (t.size,))
        C_Bf = jnp.broadcast_to(jnp.asarray(C_B)[..., :1],
                                tuple(C_B.shape[:-1]) + (t.size,))
        lnL_t = AM.coefficient_table_distphipsimarg_exact(
            fine, C_Bf, pol.x_grid, pol.log_w_grid,
            amp_sizing=pol.angle_marg_info["amp_sizing"], m_max=meta["m_max"])
        w = jnp.asarray(_core._simpson_weights(t.size, data.deltaT / refine))
        ref = np.asarray(_core._time_marginalize(lnL_t, w))
        assert np.max(np.abs(b - ref)[usable]) < 3.0e-3, (summary, b, ref, a)

    theta = jnp.asarray([RA[0], DEC[0], INCL[0]])
    v, g = pol._value_and_grad(theta)
    if np.isfinite(b[0]):
        assert np.isfinite(float(v)) and np.all(np.isfinite(np.asarray(g)))
        assert float(v) == pytest.approx(float(b[0]), abs=1e-9)
    else:
        # nan is the fail-closed value on the AD path too: a MALA step on it
        # is rejected rather than accepted on a number nobody stands behind.
        assert np.isnan(float(v))


def test_wrapper_policy_has_no_lnLt_path():
    """A consumer asking for lnL(t) must be refused, not handed the
    time-marginalized value under that name.  The exact scheme still serves
    it, so the refusal is the policy's, not the wrapper's."""
    data = make_synth(scale=2.0)
    kw = dict(nphi=32, npsi=8, interp=INTERP, angle_marg="exact")
    exact = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, **kw)
    lnLt = exact._fused(data, jnp.asarray(RA[:1]), jnp.asarray(DEC[:1]),
                        jnp.asarray(INCL[:1]), return_lnLt=True)
    assert np.asarray(lnLt).shape == (1, data.npts)
    pol = JAXDistPhiPsiMargLikelihood(
        data, 30.0, 3000.0, direct_marginalization_policy="auto",
        policy_config=DP.PolicyConfig(time_guard=4), **kw)
    with pytest.raises(ValueError, match="no lnL"):
        pol._fused(data, jnp.asarray(RA[:1]), jnp.asarray(DEC[:1]),
                   jnp.asarray(INCL[:1]), return_lnLt=True)


# ------------------------------------------------------- guard vs stored buffer

def test_guard_past_the_stored_buffer_is_refused_at_construction(monkeypatch):
    """A guard the data buffer cannot supply yields a nonfinite table with no
    error from the gather; the wrapper must refuse it with the remedy, not let
    it read as a method decline.  Per row, the same condition is a distinct
    input flag."""
    data = make_synth(scale=2.0)
    real = AM.angle_coefficient_tables

    def poisoned(d, ra, dec, incl, interp=None, sample_chunk=None, guard=0):
        C_A, C_B, meta = real(d, ra, dec, incl, interp, sample_chunk=sample_chunk,
                              guard=guard)
        if guard >= 8:
            C_A = C_A.at[..., 0].set(jnp.nan)
        return C_A, C_B, meta

    monkeypatch.setattr(AM, "angle_coefficient_tables", poisoned)
    kw = dict(nphi=32, npsi=8, interp=INTERP, angle_marg="exact")
    with pytest.raises(ValueError, match="not finite at time_guard=8"):
        JAXDistPhiPsiMargLikelihood(
            data, 30.0, 3000.0, direct_marginalization_policy="auto",
            policy_config=DP.PolicyConfig(time_guard=8), **kw)
    # guard 4 passes the probe; the row-level flag is true
    pol = JAXDistPhiPsiMargLikelihood(
        data, 30.0, 3000.0, direct_marginalization_policy="auto",
        policy_config=DP.PolicyConfig(time_guard=4), **kw)
    lnL, ledger = pol._batched_ledger(jnp.asarray(RA[:1]), jnp.asarray(DEC[:1]),
                                      jnp.asarray(INCL[:1]))
    assert bool(np.asarray(ledger["tables_finite"])[0])
    assert not bool(np.asarray(ledger["input_nonfinite"])[0])
    # a nonfinite row at evaluation time is nan with the input flag set (the
    # policy function is called directly so the poisoned table reaches it)
    def poison_all(d, ra, dec, incl, interp=None, sample_chunk=None, guard=0):
        C_A, C_B, meta = real(d, ra, dec, incl, interp, sample_chunk=sample_chunk,
                              guard=guard)
        return C_A.at[..., 0].set(jnp.nan), C_B, meta

    monkeypatch.setattr(AM, "angle_coefficient_tables", poison_all)
    lnL2, ledger2 = DP.fused_log_likelihood_four_axis_policy(
        data, jnp.asarray(RA[:1]), jnp.asarray(DEC[:1]), jnp.asarray(INCL[:1]),
        pol.x_grid, pol.log_w_grid, interp=INTERP,
        amp_sizing=pol.angle_marg_info["amp_sizing"],
        config=pol.policy_config, return_ledger=True)
    assert bool(np.asarray(ledger2["input_nonfinite"])[0])
    assert not bool(np.asarray(ledger2["usable"])[0])
    assert np.isnan(float(lnL2[0]))


def test_defaults_are_the_production_measured_operating_point():
    """The defaults follow the ladder record that accepted on production
    tables, not PR #268's test fixture values."""
    cfg = DP.PolicyConfig()
    assert (cfg.base_oversample, cfg.enriched_oversample) == (2, 4)
    assert (cfg.max_modes, cfg.enriched_max_modes) == (16, 16)
    assert cfg.local_radius == 6.0
    assert cfg.time_guard == 128
    assert cfg.reserve_time_refine_max >= cfg.reserve_time_refine


# ------------------------------------------------------------------- the CLI

def test_the_driver_CLI_offers_the_policy_and_rejects_a_typo():
    """Deliberately a SUBPROCESS, like the peak-local wiring test: optparse
    builds its choices from POLICY_CHOICES at import time."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    driver = os.path.join(root, "bin", "integrate_likelihood_extrinsic_jax")
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_PLATFORMS"] = "cpu"

    def run(*args):
        p = subprocess.run([sys.executable, driver] + list(args), env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=600)
        return p.returncode, p.stdout.decode("utf-8", "replace")

    rc, out = run("--direct-marginalization-policy", "autp")
    assert rc != 0 and "invalid choice" in out, out[-1500:]
    assert "auto" in out, out[-1500:]
    # Scope (external review P1): the policy outside its one mode, and its
    # knobs without the policy, are refused at parse time, not ignored.
    rc, out = run("--mode", "laplace-is", "--direct-marginalization-policy",
                  "auto")
    assert rc != 0 and "flowmc-phipsimarg" in out, out[-1500:]
    rc, out = run("--mode", "flowmc-phipsimarg",
                  "--direct-marginalization-time-guard", "8")
    assert rc != 0 and "inert" in out, out[-1500:]
    rc, out = run("--mode", "flowmc-phipsimarg",
                  "--direct-marginalization-policy", "auto",
                  "--direct-marginalization-reserve-time-refine", "3")
    assert rc != 0 and "even" in out, out[-1500:]
    rc, out = run("--mode", "flowmc-phipsimarg",
                  "--direct-marginalization-policy", "auto",
                  "--direct-marginalization-reserve-time-refine-max", "2")
    assert rc != 0 and "refine-max" in out, out[-1500:]
    rc, out = run("--help")
    assert "--direct-marginalization-policy" in out
    assert "--direct-marginalization-time-guard" in out
    assert "--direct-marginalization-reserve-time-refine" in out
    assert "--direct-marginalization-reserve-time-refine-max" in out
    assert "--direct-marginalization-error-budget-nats" in out


# ------------------------------------------- row batching is a COST knob only

def _policy_like(batch_rows, data, **kw):
    cfg = DP.PolicyConfig(time_guard=16, reserve_time_refine=4,
                          reserve_time_refine_max=8,
                          reserve_batch_rows=batch_rows)
    return JAXDistPhiPsiMargLikelihood(
        data, 30.0, 3000.0, angle_marg="exact",
        direct_marginalization_policy="auto", policy_config=cfg, **kw)


def test_row_batch_size_changes_cost_not_values_decisions_or_gradients():
    """``reserve_batch_rows`` may only move VALUES, never cost.

    Cost it does move, and against the change: see the design note. This test
    is about the values, decisions and gradients only.

    Executing B rows together turns the controller's tier-escalation
    ``lax.cond`` into a ``select`` under ``vmap``, so EVERY reserve tier runs
    for EVERY row in the batch instead of only for the rows that failed their
    warrant.  That is a real change to the executed graph, and it must leave
    the selected value, every branch decision in the ledger, the summary
    counts, and the reverse-mode gradient exactly where the row-at-a-time path
    put them.  The gradient is checked separately from the value because a
    ``select`` propagates the untaken branch's nan into the cotangent even
    when it discards the untaken branch's value -- the classic ``where``
    nan-gradient trap, which a value-only comparison cannot see.
    """
    S = 6
    ra = np.linspace(0.55, 1.35, S)
    dec = np.linspace(0.05, 0.75, S)
    incl = np.linspace(0.35, 2.45, S)
    data = make_synth(scale=2.0, kappa_boost=10.0)
    kw = dict(nphi=32, npsi=8, interp=INTERP)
    args = (jnp.asarray(ra), jnp.asarray(dec), jnp.asarray(incl))

    ref_like = _policy_like(1, data, **kw)
    ref_lnL, ref_led = ref_like._batched_ledger(*args)
    ref_lnL = np.asarray(ref_lnL)
    ref_led = {k: np.asarray(v) for k, v in ref_led.items()}
    ref_sum = DP.summarize_policy_ledger(ref_led)
    assert ref_sum["reserve_batch_execution_sequential"] is True
    assert ref_sum["reserve_batch_rows"] == 1

    # The comparison is only meaningful if the escalation predicate is
    # actually mixed across the batch: with a uniform predicate the select
    # would agree with the cond for trivial reasons.  Assert the fixture
    # still produces both, so a future retune cannot silently blind this.
    esc = ref_led["reserve_escalations"]
    assert esc.min() != esc.max(), (
        "fixture no longer mixes escalating and non-escalating rows "
        "(escalations=%r); the batched/sequential comparison would be blind"
        % (esc,))

    mid = np.array([ra[S // 2], dec[S // 2], incl[S // 2]])
    ref_v, ref_g = ref_like.value_and_grad(mid)
    assert np.all(np.isfinite(ref_g)), ref_g

    for B in (2, 4, 0):          # 0 == one full batch of all S rows
        like = _policy_like(B, data, **kw)
        lnL, led = like._batched_ledger(*args)
        lnL = np.asarray(lnL)
        led = {k: np.asarray(v) for k, v in led.items()}

        np.testing.assert_array_equal(np.isnan(lnL), np.isnan(ref_lnL))
        np.testing.assert_allclose(lnL, ref_lnL, rtol=0.0, atol=1e-13,
                                   equal_nan=True)

        assert set(led) == set(ref_led)
        for key, want in sorted(ref_led.items()):
            if key in ("reserve_batch_execution_sequential",
                       "reserve_batch_rows_requested",
                       "reserve_batch_rows_executed"):
                continue
            got = led[key]
            if want.dtype == bool or np.issubdtype(want.dtype, np.integer):
                np.testing.assert_array_equal(
                    got, want, err_msg="batch_rows=%d moved ledger key %r"
                    % (B, key))
            else:
                np.testing.assert_array_equal(
                    np.isfinite(got), np.isfinite(want),
                    err_msg="batch_rows=%d moved finiteness of %r" % (B, key))
                np.testing.assert_allclose(
                    got, want, rtol=0.0, atol=1e-13, equal_nan=True,
                    err_msg="batch_rows=%d moved ledger key %r" % (B, key))

        summ = DP.summarize_policy_ledger(led)
        for key in ref_sum:
            # The three batch keys are the ones that MUST differ; they are
            # asserted explicitly below.
            if key in ("reserve_batch_rows",
                       "reserve_batch_rows_requested",
                       "reserve_batch_execution_sequential"):
                continue
            want, got = ref_sum[key], summ[key]
            # Counts must be exact.  max_local_error_score_nats is not a count:
            # it is a max over a per-row float diagnostic that is itself a
            # reduction, so the select reassociates it.  Measured 3.553e-15 on
            # 2 of 6 rows (2.4e-13 relative) while lnL stayed bitwise equal and
            # every branch decision held, so it is compared as a float.
            if isinstance(want, float):
                assert np.isclose(got, want, rtol=1e-11, atol=0.0,
                                  equal_nan=True), (B, key, got, want)
            else:
                assert got == want, (B, key, got, want)

        # The ledger key must state what actually happened.  It read True
        # unconditionally before the batch size was a knob.
        # The ledger reports what EXECUTED, not what was requested: a request
        # above the row count runs a vmap of S, not of B.
        assert summ["reserve_batch_execution_sequential"] is False
        assert summ["reserve_batch_rows"] == (S if (B == 0 or B >= S) else B)
        assert summ["reserve_batch_rows_requested"] == B
        assert not np.any(led["reserve_batch_execution_sequential"])

        # The gradient is compared at EVERY batch size.  Dropping it to one
        # size, and halving nphi, were both tried and neither moved the test's
        # cost (640.8 s against 641.6 s on the ldas-grid CPU runner), so the
        # cheaper variants bought nothing and this keeps the coverage.
        # A single row has nothing to batch, and value_and_grad evaluates
        # exactly one.  If a batch request reached it, both the accept/reserve
        # cond and every escalation tier would become selects and the gradient
        # would cost more than at batch 1 for no amortization.  Pin that the
        # scalar path stays sequential whatever was asked for.
        _, sled = like._batched_ledger(jnp.asarray(mid[0:1]),
                                       jnp.asarray(mid[1:2]),
                                       jnp.asarray(mid[2:3]))
        ssum = DP.summarize_policy_ledger({k: np.asarray(v_)
                                           for k, v_ in sled.items()})
        assert ssum["reserve_batch_rows"] == 1, (B, ssum["reserve_batch_rows"])
        assert ssum["reserve_batch_execution_sequential"] is True

        v, g = like.value_and_grad(mid)
        assert np.all(np.isfinite(g)), (B, g)
        assert abs(v - ref_v) <= 1e-13, (B, v, ref_v)   # measured exactly 0
        # Relative, not absolute: the select re-associates the cotangent sum,
        # so the gradient is equal to ~1 ulp rather than bitwise (measured
        # 2.8e-14 on a Blackwell GPU and 5.6e-14 on CPU against |dlnL/dincl|
        # ~ 141).  A branch decision that actually moved would be O(1) nats.
        np.testing.assert_allclose(g, ref_g, rtol=1e-12, atol=1e-12,
                                   err_msg="batch_rows=%d moved the gradient" % B)


@pytest.mark.parametrize("bad", [-1, -8, 2.5, "4", None])
def test_row_batch_size_refuses_what_it_cannot_execute(bad):
    with pytest.raises(ValueError):
        DP.validate_batch_rows(bad)


# ------------------------------------------- preset local-plan capacities

def test_local_plan_capacities_reach_the_planner(monkeypatch):
    """Both capacities are PRESET, and -- the point of the test -- reachable.

    The regression guarded here is not a wrong number but an unreachable one:
    max_time_nodes defaulted to 64 inside rank_joint_starts_from_uvq_device and
    the policy never passed it, so the value that gated the local path could
    not be moved from any config field or flag.  Asserting the default alone
    would have passed against that bug, so spy on the call and read the kwargs
    the planner is actually handed."""
    seen = []

    class _Stop(Exception):
        pass

    def spy(*a, **kw):
        seen.append(kw)
        raise _Stop

    monkeypatch.setattr(DP._aap, "rank_joint_starts_from_uvq_device", spy)
    guarded, C_B, _ = _guarded_problem(_N, _GUARD)
    _install_tables(monkeypatch, ([guarded], C_B), _GUARD)
    data = _fake_data(_N)
    x_grid, log_w = _grid(256)
    with pytest.raises(_Stop):
        DP.fused_log_likelihood_four_axis_policy(
            data, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), x_grid, log_w,
            interp=INTERP, amp_sizing=40.0,
            config=DP.PolicyConfig(time_guard=_GUARD, max_time_nodes=333,
                                   base_max_starts=77))
    assert seen, "the planner was never called"
    assert seen[0].get("max_time_nodes") == 333, seen[0]
    assert seen[0].get("max_starts") == 77, seen[0]


def test_capacity_and_budget_defaults_are_the_approved_operating_point():
    """Pins the operating point RO approved on 2026-09-08 (evening).

    Shipped was (64, 32) with a 1e-3 budget.  The capacities moved because
    acceptance at rho 652 goes (64, 32) 28%, (256, 32) 31%, (256, 128) 75%;
    they are pinned as a PAIR because of 64 rows, 36 declines fail on time
    nodes and 37 on starts and only 7 on starts alone, so a later change that
    moves one alone is a mistake this test should catch.  The budget moved
    because 1e-3 -> 1e-2 took reserve escalations from 2 to 0 at rho 41, a
    2.19x speedup, while moving lnL by 4.8e-12 nats.

    None of these is value-neutral, which is why they are pinned rather than
    left to drift: base_max_starts 32 -> 128 alone shifts already-accepted
    values by up to 3.5e-3 nats at rho 163."""
    cfg = DP.PolicyConfig()
    assert (cfg.max_time_nodes, cfg.base_max_starts) == (256, 128)
    assert cfg.total_value_error_budget_nats == 1.0e-2


@pytest.mark.parametrize("kw", [{"max_time_nodes": 1}, {"max_time_nodes": 0},
                                {"base_max_starts": 0},
                                {"base_max_starts": -4}])
def test_capacities_refuse_what_cannot_plan(kw):
    with pytest.raises(ValueError):
        DP.validate_policy_config(DP.PolicyConfig(**kw))


def test_driver_offers_the_capacity_knobs_and_scopes_them():
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    driver = os.path.join(root, "bin", "integrate_likelihood_extrinsic_jax")
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_PLATFORMS"] = "cpu"

    def run(*args):
        p = subprocess.run([sys.executable, driver] + list(args), env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=600)
        return p.returncode, p.stdout.decode("utf-8", "replace")

    rc, out = run("--help")
    assert "--direct-marginalization-max-time-nodes" in out
    assert "--direct-marginalization-max-starts" in out
    # --help exits before validation, so it proves only that the flags parse.
    # Both must also be scoped: inert without the policy is an error, not a
    # silently ignored flag.
    for flag in ("--direct-marginalization-max-time-nodes",
                 "--direct-marginalization-max-starts"):
        rc, out = run("--mode", "flowmc-phipsimarg", flag, "64")
        assert rc != 0 and "inert" in out, (flag, out[-1500:])


def test_driver_offers_the_batch_rows_knob_and_scopes_it():
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    driver = os.path.join(root, "bin", "integrate_likelihood_extrinsic_jax")
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_PLATFORMS"] = "cpu"

    def run(*args):
        p = subprocess.run([sys.executable, driver] + list(args), env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=600)
        return p.returncode, p.stdout.decode("utf-8", "replace")

    rc, out = run("--help")
    assert "--direct-marginalization-batch-rows" in out
    # Inert without the policy, and refused rather than clamped when negative.
    rc, out = run("--mode", "flowmc-phipsimarg",
                  "--direct-marginalization-batch-rows", "8")
    assert rc != 0 and "inert" in out, out[-1500:]
    rc, out = run("--mode", "flowmc-phipsimarg",
                  "--direct-marginalization-policy", "auto",
                  "--direct-marginalization-batch-rows", "-1")
    assert rc != 0 and "batch-rows" in out, out[-1500:]
