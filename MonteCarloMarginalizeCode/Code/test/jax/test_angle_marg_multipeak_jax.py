"""Static-cost, JIT/AD-compatible multipeak wiring and contract."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile import direct_marginalization_policy as DP
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood
from test_angle_marg_exact import make_synth, RA, DEC, INCL, INTERP


def test_multipeak_jax_is_explicit_only_and_records_bounded_contract():
    assert "multipeak-jax" in AM.ANGLE_MARG_CHOICES
    for amp in (1.0, 500.0, 5.0e6):
        assert AM.choose_angle_marg_scheme(amp)[0] != "multipeak-jax"

    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(
        data, 30.0, 3000.0, nphi=32, npsi=8, interp=INTERP,
        angle_marg="multipeak-jax")
    assert "amp_sizing" not in like.angle_marg_info
    assert "sample_grid" not in like.angle_marg_info
    assert like.angle_marg_info["config"] == DP.BoundedMultipeakConfig()._asdict()
    assert like.angle_marg_info["bounded_cost"] is True
    assert like.angle_marg_info["dense_reserve"] is False
    assert like.angle_marg_info["fixed_plan_autodiff_only"] is True
    assert like.angle_marg_info["derivative_warrant_certified"] is False


def test_multipeak_jax_wrapper_uses_jit_and_ad(monkeypatch):
    def differentiable_stub(data, ra, dec, incl, *args, **kwargs):
        return ra + 2.0 * dec + 3.0 * incl

    monkeypatch.setattr(
        DP, "fused_log_likelihood_four_axis_bounded", differentiable_stub)
    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(
        data, 30.0, 3000.0, nphi=32, npsi=8, interp=INTERP,
        angle_marg="multipeak-jax")
    value = np.asarray(like._batched(
        jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL)))
    assert value.shape == np.shape(RA)
    val, grad = like._value_and_grad(
        jnp.asarray([RA[0], DEC[0], INCL[0]]))
    assert np.isfinite(np.asarray(val))
    np.testing.assert_allclose(np.asarray(grad), [1.0, 2.0, 3.0])


def test_multipeak_jax_refuses_time_series_and_amp_grid_outputs(monkeypatch):
    monkeypatch.setattr(
        DP, "fused_log_likelihood_four_axis_bounded",
        lambda data, ra, dec, incl, *args, **kwargs: ra * 0.0)
    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(
        data, 30.0, 3000.0, nphi=32, npsi=8, interp=INTERP,
        angle_marg="multipeak-jax")
    args = (data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL))
    with pytest.raises(ValueError, match="no lnL"):
        like._fused(*args, return_lnLt=True)
    with pytest.raises(ValueError, match="static cost envelope"):
        like._fused(*args, return_amp=True)


def test_bounded_config_refuses_nonstatic_or_invalid_envelopes():
    with pytest.raises(ValueError, match="time_guard"):
        DP.validate_bounded_multipeak_config(
            DP.BoundedMultipeakConfig(time_guard=1))
    with pytest.raises(ValueError, match="enriched_oversample"):
        DP.validate_bounded_multipeak_config(
            DP.BoundedMultipeakConfig(base_oversample=2,
                                      enriched_oversample=2))
    with pytest.raises(ValueError, match="mode caps"):
        DP.validate_bounded_multipeak_config(
            DP.BoundedMultipeakConfig(base_max_starts=1,
                                      enriched_max_modes=3))


@pytest.mark.parametrize("invalid", [None, "norm", "table"])
def test_device_function_is_jittable_differentiable_and_forwards_caps(
        monkeypatch, invalid):
    seen = []

    def fake_tables(data, ra, dec, incl, interp, guard):
        del data, interp
        ntime = 2 * int(guard) + 3
        signal = ra + 2.0 * dec + 3.0 * incl
        ca = jnp.broadcast_to(signal[None, None, :, None],
                              (1, 1, signal.size, ntime)).astype(jnp.complex128)
        cb = jnp.ones_like(ca)
        if invalid == "norm":
            cb = cb.at[..., -1].set(2.0)
        if invalid == "table":
            ca = ca.at[..., 0].set(jnp.nan)
        return ca, cb, {"m_max": 0}

    def fake_rank(table, norm, x_min, x_max, **kwargs):
        del table, norm, x_min, x_max
        seen.append((kwargs["max_starts"], kwargs["max_time_nodes"],
                     kwargs["angular_oversample"]))
        return jnp.asarray(0.0)

    def fake_plan(table, norm, base, extra, x_min, x_max, **kwargs):
        del norm, base, extra, x_min, x_max
        token = {"token": jnp.real(jnp.sum(table))}
        one = jnp.asarray(1, dtype=jnp.int32)
        ok = jnp.asarray(True)
        plan = dict(n_selected_modes=one, n_optimizer_starts=one,
                    n_lattice_evaluations=one, n_candidates_before_cap=one,
                    start_capacity_ok=ok, time_capacity_ok=ok)
        shared = {"n_optimizer_starts_executed": one}
        assert kwargs["max_modes"] == 1
        assert kwargs["enriched_max_modes"] == 2
        return token, token, plan, plan, shared

    def fake_integral(table, norm, base_plan, enriched_plan,
                      x_min, x_max, **kwargs):
        del norm, base_plan, enriched_plan, x_min, x_max, kwargs
        # Return a finite candidate even for the bad A table, isolating the
        # outer tables_finite guard from both norm invariance and inner gates.
        value = jnp.real(jnp.sum(jnp.nan_to_num(table)))
        return value, jnp.asarray(True), {"accepted_local": jnp.asarray(True)}

    monkeypatch.setattr(DP._anglemarg, "angle_coefficient_tables", fake_tables)
    monkeypatch.setattr(DP._anglemarg, "_runtime_amp_failsafe",
                        lambda *args, **kwargs: jnp.asarray(0.0))
    monkeypatch.setattr(DP._aap, "rank_joint_starts_from_uvq_device", fake_rank)
    monkeypatch.setattr(DP._aap, "make_all_axis_mode_plan_pair_device", fake_plan)
    monkeypatch.setattr(DP._aap, "empirical_enrichment_marginalize", fake_integral)

    cfg = DP.BoundedMultipeakConfig(
        time_guard=2, base_max_starts=2, max_time_nodes=3,
        base_oversample=1, enriched_oversample=2,
        max_modes=1, enriched_max_modes=2, refine_iterations=1,
        base_order=2, base_check_order=3,
        enriched_order=3, enriched_check_order=4)
    xg = jnp.asarray([0.5, 1.0])
    lwg = jnp.asarray([-np.log(2.0), -np.log(2.0)])

    def scalar(theta):
        value, ledger = DP.fused_log_likelihood_four_axis_bounded(
            object(), theta[0:1], theta[1:2], theta[2:3], xg, lwg,
            interp=None, amp_sizing=10.0, config=cfg,
            local_log_normalization=0.0, x_bounds=(0.5, 1.0),
            return_ledger=True)
        return value[0], ledger

    (value, ledger), grad = jax.jit(jax.value_and_grad(
        scalar, has_aux=True))(
            jnp.asarray([0.1, 0.2, 0.3]))
    if invalid is None:
        assert np.isfinite(np.asarray(value))
        np.testing.assert_allclose(np.asarray(grad), 7.0 * np.asarray([1., 2., 3.]))
    else:
        assert np.isnan(np.asarray(value))
        assert not bool(ledger["usable"][0])
        assert not bool(ledger["norm_time_invariant" if invalid == "norm" else "tables_finite"][0])
    assert bool(np.asarray(ledger["bounded_cost"][0]))
    assert not bool(np.asarray(ledger["derivative_warrant_certified"][0]))
    assert (2, 3, 1) in seen and (2, 3, 2) in seen


@pytest.mark.parametrize("changes", [
    {"base_max_starts": 0}, {"max_time_nodes": 1},
    {"base_oversample": 0}, {"max_modes": 0},
    {"enriched_max_modes": 7}, {"enriched_max_modes": 257},
    {"base_order": 1}, {"base_check_order": 11},
    {"enriched_order": 12}, {"enriched_check_order": 13},
    {"local_radius": 0}, {"refine_iterations": 0},
    {"convergence_tol_nats": 0}, {"time_guard_tol_nats": 0},
    {"total_value_error_budget_nats": 0},
    {"time_outside_tol_nats": 0}, {"time_outside_tol_nats": -np.inf},
    {"norm_invariance_rtol": -1}, {"norm_invariance_rtol": np.nan},
    {"batch_rows": -1}, {"batch_rows": 1.5},
])
def test_invalid_static_envelope_is_refused(changes):
    with pytest.raises(ValueError):
        DP.validate_bounded_multipeak_config(DP.BoundedMultipeakConfig(**changes))


def test_bounded_config_type_and_shared_defaults():
    with pytest.raises(TypeError, match="BoundedMultipeakConfig"):
        DP.validate_bounded_multipeak_config(DP.PolicyConfig())
    cfg = DP.BoundedMultipeakConfig()
    separate = {"time_guard", "base_max_starts", "max_time_nodes",
                "max_modes", "enriched_max_modes",
                "batch_rows", "base_order", "base_check_order",
                "enriched_order", "enriched_check_order",
                "convergence_tol_nats", "total_value_error_budget_nats"}
    for field in cfg._fields:
        if field not in separate:
            assert getattr(cfg, field) == getattr(DP.PolicyConfig(), field)


@pytest.mark.parametrize("bounds", [(0., 1.), (-1., 1.), (1., 1.), (2., 1.)])
def test_invalid_distance_support_refused_before_tables(bounds):
    with pytest.raises(ValueError, match="x_bounds"):
        DP.fused_log_likelihood_four_axis_bounded(
            None, None, None, None, None, None, interp=None, amp_sizing=1.,
            local_log_normalization=0., x_bounds=bounds)


@pytest.mark.parametrize("kwargs,match", [
    ({"time_quadrature": "bandlimited"},
     "time_quadrature='bandlimited' is not valid for distance/phase/polarization marginalization"),
    ({"dist_grid": "loguniform"}, "uniform-in-distance"),
])
def test_wrapper_refuses_incompatible_measures(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JAXDistPhiPsiMargLikelihood(
            make_synth(scale=2.), 30., 3000., interp=INTERP,
            angle_marg="multipeak-jax", **kwargs)


def test_real_kernel_default_envelope_accepts_and_matches_reference(monkeypatch):
    from test_direct_marginalization_policy import (
        _guarded_problem, _fake_data, _grid, _fine_reference, _N)
    cfg = DP.BoundedMultipeakConfig()
    table, norm, constants = _guarded_problem(_N, cfg.time_guard)
    data = _fake_data(_N)
    xg, lw = _grid()
    monkeypatch.setattr(DP._core, "_DISTMARG_GH_N", 0)

    def tables(data, ra, dec, incl, interp, guard):
        assert guard == cfg.time_guard
        ca = jnp.asarray(table)[:, :, None, :] * ra[None, None, :, None]
        cb = jnp.broadcast_to(jnp.asarray(norm)[:, :, None, None],
                              norm.shape + (ra.size, ca.shape[-1]))
        return ca, cb, {"m_max": 2}

    # Only inject analytic coefficient tables: discovery, refinement, all
    # quadratures and acceptance guards below are the actual production code.
    monkeypatch.setattr(AM, "angle_coefficient_tables", tables)
    lln, _ = DP.policy_log_normalization(data, xg, lw)
    bounds = (float(xg.min()), float(xg.max()))

    def evaluate(scales, config=cfg):
        return DP.fused_log_likelihood_four_axis_bounded(
            data, scales, jnp.zeros_like(scales), jnp.zeros_like(scales), xg, lw,
            interp=INTERP, amp_sizing=40., config=config,
            local_log_normalization=lln, x_bounds=bounds, return_ledger=True)

    value, ledger = jax.jit(evaluate)(jnp.array([1., 1.01]))
    assert np.all(ledger["usable"]), {
        k: np.asarray(v) for k, v in ledger.items() if k.startswith("decline")}
    for i, scale in enumerate((1., 1.01)):
        fine = _fine_reference(constants, norm, data, xg, lw, 40., scale=scale)
        assert abs(float(value[i]) - fine) < 1.e-3
    # The old orders decline the same problem, even with the corrected caps.
    old_orders = cfg._replace(base_order=7, base_check_order=9,
                              enriched_order=9, enriched_check_order=11,
                              convergence_tol_nats=1.e-3,
                              total_value_error_budget_nats=1.e-2)
    declined, why = jax.jit(lambda x: evaluate(x, old_orders))(jnp.array([1.]))
    assert np.isnan(declined[0]) and bool(why["decline_quadrature"][0])
    # AD is exercised through the real accepted fixed-plan quadrature.
    scalar = lambda x: evaluate(x[None])[0][0]
    grad = jax.jit(jax.grad(scalar))(jnp.array(1.))
    h = 1.e-5
    fd = (float(jax.jit(scalar)(1. + h)) - float(jax.jit(scalar)(1. - h))) / (2*h)
    np.testing.assert_allclose(grad, fd, rtol=2.e-5, atol=1.e-5)


def test_cli_envelope_reaches_wrapper_and_provenance(monkeypatch):
    from test_direct_marginalization_policy_cli import _load_driver
    mod = _load_driver()
    parser = mod.build_parser()
    opts, _ = parser.parse_args([])
    assert mod.bounded_multipeak_config_from_options(opts) == DP.BoundedMultipeakConfig()
    opts, _ = parser.parse_args([
        "--multipeak-jax-time-guard", "64",
        "--multipeak-jax-base-max-starts", "256",
        "--multipeak-jax-max-time-nodes", "512",
        "--multipeak-jax-base-order", "12"])
    cfg = mod.bounded_multipeak_config_from_options(opts)
    seen = []
    def kernel(data, ra, dec, incl, *args, config, **kwargs):
        seen.append(config)
        values = ra * 0.
        return (values, {"selected_value": values}) if kwargs.get("return_ledger") else values
    monkeypatch.setattr(DP, "fused_log_likelihood_four_axis_bounded", kernel)
    like = JAXDistPhiPsiMargLikelihood(
        make_synth(scale=2.), 30., 3000., interp=INTERP,
        angle_marg="multipeak-jax", bounded_multipeak_config=cfg)
    like.log_likelihood(RA, DEC, INCL).block_until_ready()
    assert seen == [cfg]
    assert cfg.time_guard == 64 and cfg.base_max_starts == 256
    assert cfg.max_time_nodes == 512 and cfg.base_order == 12
    assert like.angle_marg_info["config"] == cfg._asdict()
    assert "dense_reserve=false" in mod.angle_grid_suspect_note("multipeak-jax")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_host_decline_is_latched_and_cannot_publish_survivors(monkeypatch, bad):
    from test_direct_marginalization_policy_cli import _load_driver
    mod = _load_driver()
    def kernel(data, ra, *args, **kwargs):
        values = jnp.where(ra > 1., bad, 0.)
        return (values, {"selected_value": values}) if kwargs.get("return_ledger") else values
    monkeypatch.setattr(DP, "fused_log_likelihood_four_axis_bounded", kernel)
    like = JAXDistPhiPsiMargLikelihood(
        make_synth(scale=2.), 30., 3000., interp=INTERP, angle_marg="multipeak-jax",
        bounded_multipeak_decline_action="refuse")
    with pytest.raises(RuntimeError, match="refusing"):
        like.log_likelihood(jnp.array([0., 2.]), jnp.zeros(2), jnp.zeros(2))
    # Even a caller swallowing the exception and keeping finite rows cannot
    # evade the final publication guard.
    with pytest.raises(RuntimeError, match="refusing to publish"):
        mod.require_bounded_multipeak_rows(like, np.array([0., 1.]))
    fresh = type("Like", (), {"angle_marg_scheme": "multipeak-jax"})()
    with pytest.raises(RuntimeError, match="refusing to publish"):
        mod.require_bounded_multipeak_rows(fresh, np.array([0., bad]))
    mod.require_bounded_multipeak_rows(fresh, np.array([0., 1.]))


@pytest.mark.parametrize("diagnostic", [-100., 10., np.nan])
def test_drop_preserves_proposal_count_and_records_material_declines(monkeypatch, diagnostic):
    from RIFT.likelihood.jax_ile.samplers import evidence_from_logweights
    def kernel(data, ra, *args, **kwargs):
        bad = ra > 1.
        values = jnp.where(bad, jnp.nan, 0.)
        ledger = {"selected_value": jnp.where(bad, diagnostic, 0.),
                  "decline_quadrature": bad}
        return (values, ledger) if kwargs.get("return_ledger") else values
    monkeypatch.setattr(DP, "fused_log_likelihood_four_axis_bounded", kernel)
    like = JAXDistPhiPsiMargLikelihood(
        make_synth(scale=2.), 30., 3000., interp=INTERP, angle_marg="multipeak-jax")
    values = like.log_likelihood(jnp.array([0., 2.]), jnp.zeros(2), jnp.zeros(2))
    np.testing.assert_array_equal(values, [0., -1.e30])
    logZ, _, _ = evidence_from_logweights(values)
    assert logZ == pytest.approx(-np.log(2.))  # not logZ=0 from discarding the denominator
    audit = like.bounded_multipeak_audit
    assert audit["evaluated"] == 2 and audit["declined"] == 1
    assert audit["reasons"] == {"decline_quadrature": 1}
    assert audit["max_accepted"] == 0.
    if np.isfinite(diagnostic):
        assert audit["max_declined_diagnostic"] == diagnostic
    else:
        assert audit["diagnostic_unknown"] == 1
    # Jitted sampler path uses the same target, without host callbacks.
    assert float(jax.jit(like._scalar)(jnp.array([2., 0., 0.]))) == -1.e30


@pytest.mark.parametrize("action", ["drop", "refuse"])
def test_driver_mixed_cloud_publication_and_cli_wiring(monkeypatch, tmp_path, action):
    import types
    from test_direct_marginalization_policy_cli import _load_driver
    from RIFT.likelihood.jax_ile import samplers, wrapper
    mod = _load_driver()
    parser = mod.build_parser()
    opts, _ = parser.parse_args([
        "--mode", "flowmc-phipsimarg", "--distance-marginalization",
        "--angle-marg-scheme", "multipeak-jax",
        "--multipeak-jax-decline-action", action,
        "--multipeak-jax-time-guard", "64",
        "--multipeak-jax-max-time-nodes", "512",
        "--output-file", str(tmp_path / "event"), "--save-samples"])
    seen = []
    def constructor(data, *args, **kwargs):
        cfg = kwargs["bounded_multipeak_config"]
        assert cfg.time_guard == 64 and cfg.max_time_nodes == 512
        assert kwargs["bounded_multipeak_decline_action"] == action
        return types.SimpleNamespace(
            angle_marg_scheme="multipeak-jax", direct_marginalization_policy="off",
            policy_info=None,
            angle_marg_info={"config": cfg._asdict()}, time_quadrature="simpson",
            bounded_multipeak_decline_action=action, bounded_multipeak_audit={})
    monkeypatch.setattr(wrapper, "JAXDistPhiPsiMargLikelihood", constructor)
    data = types.SimpleNamespace(lms=np.array([[2, 2]]), q_time_pregrid_factor=1)
    monkeypatch.setattr(mod, "build_data_from_precompute",
                        lambda *a, **k: (data, {"guess_snr": 10.}))
    result = dict(theta=np.zeros((3, 3)), lnL=np.array([0., -1.e30, 1.]),
                  logZ=.1, sigma_over_Z=.1, neff=2., post_weight=np.ones(3)/3)
    monkeypatch.setattr(samplers, "flowmc_sample_phimarg", lambda *a, **k: result)
    def samples(opts, idx, theta, lnL, with_distance, **kwargs):
        seen.append(("samples", np.asarray(lnL), kwargs))
    def dat(*args, **kwargs):
        seen.append(("dat", kwargs))
    monkeypatch.setattr(mod, "write_samples", samples)
    monkeypatch.setattr(mod, "write_dat", dat)
    P = types.SimpleNamespace(copy=lambda: None)
    if action == "refuse":
        with pytest.raises(RuntimeError, match="refusing to publish"):
            mod.analyze_one(opts, P, {}, {}, False, 0., np.random.default_rng(1), 0, 1)
        assert not seen
    else:
        mod.analyze_one(opts, P, {}, {}, False, 0., np.random.default_rng(1), 0, 1)
        assert len(seen) == 2
        np.testing.assert_array_equal(seen[0][1], [0., 1.])
        assert len(seen[0][2]["logw"]) == 2
        note = seen[0][2]["angle_note"]
        assert "output_rows_dropped=1" in note and "omitted_mass=unbounded" in note
        assert "audit_scope=log_likelihood-batches" in note
        assert "audit_excludes=scalar-MAP-Fisher-MALA" in note
        assert "refuse_latch_scope=log_likelihood-batches" in note
        assert "scalar_refuse_decline=nan-without-raise-or-latch" in note
        assert seen[1][1]["angle_note"] == note


@pytest.mark.parametrize("args,expected", [
    (["--multipeak-jax-base-order", "15"], "requires --angle-marg-scheme"),
    (["--angle-marg-scheme", "multipeak-jax", "--multipeak-jax-base-order", "1"],
     "base_order"),
    (["--mode", "prior-mc", "--angle-marg-scheme", "multipeak-jax"],
     "requires --mode flowmc-phipsimarg"),
])
def test_cli_refuses_inert_or_invalid_envelope(args, expected):
    from test_direct_marginalization_policy_cli import _run
    rc, out = _run("--mode", "flowmc-phipsimarg", *args)
    assert rc != 0 and expected in out, out[-2000:]


def test_drop_refuses_an_entirely_declined_output_cloud():
    from types import SimpleNamespace
    from test_direct_marginalization_policy_cli import _load_driver
    mod = _load_driver()
    like = SimpleNamespace(angle_marg_scheme="multipeak-jax",
                           bounded_multipeak_decline_action="drop")
    with pytest.raises(RuntimeError, match="no accepted output rows"):
        mod.require_bounded_multipeak_rows(like, np.array([-1.e30, np.nan]))


@pytest.mark.parametrize("kwargs,match", [
    ({"bounded_multipeak_decline_action": "invalid"},
     "bounded_multipeak_decline_action must be drop or refuse"),
    ({"multipeak_guard": 32,
      "bounded_multipeak_config": DP.BoundedMultipeakConfig(time_guard=16)},
     "multipeak_guard conflicts with bounded_multipeak_config"),
])
def test_wrapper_rejects_invalid_python_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JAXDistPhiPsiMargLikelihood(
            make_synth(scale=2.), 30., 3000., interp=INTERP,
            angle_marg="multipeak-jax", **kwargs)


@pytest.mark.parametrize("action", ["drop", "refuse"])
def test_scalar_declines_are_explicitly_outside_host_audit(monkeypatch, action):
    def kernel(data, ra, *args, **kwargs):
        return jnp.full_like(ra, jnp.nan)
    monkeypatch.setattr(DP, "fused_log_likelihood_four_axis_bounded", kernel)
    like = JAXDistPhiPsiMargLikelihood(
        make_synth(scale=2.), 30., 3000., interp=INTERP,
        angle_marg="multipeak-jax", bounded_multipeak_decline_action=action)
    theta = jnp.zeros(3)
    value, grad = like.value_and_grad(theta)
    like.fisher(theta)
    scalar = jax.jit(like._scalar)(theta)
    if action == "refuse":
        assert np.isnan(value) and np.isnan(scalar)
    else:
        assert value == -1.e30 and scalar == -1.e30
    assert like.bounded_multipeak_audit["evaluated"] == 0
    assert not getattr(like, "bounded_multipeak_declined", False)
