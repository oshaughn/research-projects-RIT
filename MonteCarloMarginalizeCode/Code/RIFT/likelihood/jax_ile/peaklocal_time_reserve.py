"""Peak-local time rule for the four-axis policy's reserve.

The policy's reserve integrates time on a rule that refines the WHOLE window
(``policy_time_rules``: ``refine`` nodes per native sample, escalating by
doubling to ``reserve_time_refine_max``).  The peak of ``exp(lnL(t))`` has
width ``sigma_t = 1 / (2 pi rho sigma_f)``, so that rule's node count grows
with rho and the angular kernel is paid at every node: 18.6 / 72.6 / 290 /
1278 s per evaluation at rho 41 / 82 / 163 / 326 with the exact kernel on
2453 nodes (ladder controller, 2026-09-08), and at rho 41 the refine-4 rule
already misses its own 1e-3 warrant on two of three rows.

This rule is sized from the PREDICTED width and its node count does not
depend on the row:

* the width is predicted per row, ``sigma_t = 1 / (2 pi rho sigma_f)``, with
  ``rho`` from the row's own coefficient table (the angular triangle bound of
  the field over the U,V norm) and ``sigma_f`` the two-sided rms frequency of
  the stored Q (the policy passes it; the table's own spectrum is the per-row
  fallback and cross-check).  Both are bounds in the safe direction: the
  narrowest peak the primitive can make at the row's amplitude;
* ONE fine lattice per row, commensurate with the coarse scan: fine spacing
  ``h_f = h_scan / m`` with the integer ``m`` chosen so that
  ``h_f <= sigma_t / nodes_per_sigma``.  Every node of the rule sits on that
  lattice, so overlapping blocks and scan nodes inside a block are EXACT
  duplicates (zero trapezoid weight), and the only spacing changes are at
  block ends, where the mass is negligible.  A dead slot collapses onto the
  first scan node.  Mixing two incommensurate
  lattices was measured to cost 0.02 nat on a 0.75-sample peak: the
  trapezoid rule is spectrally accurate only on uniform spacing;
* a coarse scan of the whole window at ``scan_refine`` nodes per native
  sample (the broad, low-lying part of the field and any peak the locator
  missed), and one block of ``n_fine`` consecutive fine-lattice nodes
  centred on each of the row's time maxima.  The maxima come from the
  PRIMITIVE, not from the local branch's plan: :func:`locate_time_maxima`
  maximizes the field over a dense phi grid, a u lattice and the distance at
  every node of a fixed search grid (the envelope the marginalized reserve
  integrates), keeps the ``n_candidates`` highest local maxima, and polishes
  each on that envelope with parabolic steps.  The plan's Newton centres and widths are reported beside
  them as a cross-check and never decide the rule, so the reserve does not
  inherit a decline of the local branch.  A dead slot sits on the scan
  lattice and adds nothing.

The check rule is the same construction at half the scan refinement and every
other fine node (spacing ``2 h_f``), so it is everywhere coarser with the same
endpoints and the kernel's structural resolution warrant applies unchanged.
The two-guard comparison and the tail bound are the kernel's.  Nothing here
is a bound: a peak the plans missed and the scan cannot resolve fails the
resolution warrant and the row escalates (``m`` and ``n_fine`` double, same
span) or returns ``nan``.  An escalation is a report on the prediction, not
a loop.

Evidence: DESIGN_direct_marginalization_policy.md, "Peak-local time reserve".
"""

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["NODES_PER_SIGMA", "peaklocal_time_rule", "peaklocal_rule_size",
           "validate_peaklocal_rule_arguments", "row_amplitude",
           "table_bandwidth_cycles", "predicted_width_samples",
           "predict_time_rule", "locate_time_maxima"]

# Fine nodes per predicted sigma_t.  Three matches the policy's whole-window
# node accounting (_TIME_NODES_PER_SIGMA); the trapezoid rule on a Gaussian at
# spacing sigma/3 is converged to ~1e-8, and its half-refined check at
# 2 sigma/3 to ~1e-4, so the warrant's 1e-3 has margin on both sides.
NODES_PER_SIGMA = 3.0


def validate_peaklocal_rule_arguments(n_target, n_fine, scan_refine):
    n_target = int(n_target)
    n_fine = int(n_fine)
    scan_refine = int(scan_refine)
    if n_target < 2:
        raise ValueError("n_target must be >= 2")
    if n_fine < 3 or n_fine % 2 == 0:
        raise ValueError("n_fine must be an odd integer >= 3 so the check "
                         "rule keeps the block endpoints")
    if scan_refine < 2 or scan_refine % 2:
        raise ValueError("scan_refine must be an even integer >= 2 so the "
                         "check scan is the half-refined scan")
    return n_target, n_fine, scan_refine


def peaklocal_rule_size(n_target, n_blocks, n_fine, scan_refine, n_scan=None):
    """Node counts ``(reserve, check)`` of the rule: a shape statement, so a
    test can pin that the count is independent of the row.  ``n_scan`` is
    the cropped scan's node count; ``None`` is the whole-window scan."""
    n_target, n_fine, scan_refine = validate_peaklocal_rule_arguments(
        n_target, n_fine, scan_refine)
    n_blocks = int(n_blocks)
    if n_scan is None:
        n_s = (n_target - 1) * scan_refine + 1
        n_sc = (n_target - 1) * (scan_refine // 2) + 1
    else:
        n_s = int(n_scan)
        if n_s < 3 or n_s % 2 == 0:
            raise ValueError("n_scan must be an odd integer >= 3")
        n_sc = (n_s - 1) // 2 + 1
    reserve = n_s + n_blocks * n_fine
    check = n_sc + n_blocks * ((n_fine - 1) // 2 + 1)
    return reserve, check


# ------------------------------------------------------------ prediction
def _kp_weights(kp):
    return jnp.where(jnp.arange(kp) == 0, 1.0, 2.0)


def row_amplitude(C_A_t, C_B, guard):
    """Predicted rho of one row from its tables: ``rho^2 = A_max^2 / B_min``.

    ``A_max`` is the angular triangle bound of the field over the target
    window (``sum_kp,ks w |C_A|``, maximized over time) and ``B_min`` the
    triangle lower bound of the U,V norm polynomial, the same bounds
    ``all_axis_peaklocal._time_cell_cover_device`` uses.  An upper bound on
    rho, so a lower bound on the width: the lattice it sizes is never
    coarser than the peak needs.
    """
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    guard = int(guard)
    target = C_A_t[..., guard:-guard] if guard else C_A_t
    wa = _kp_weights(C_A_t.shape[0])
    a_upper = jnp.max(jnp.sum(wa[:, None, None] * jnp.abs(target), axis=(0, 1)))
    ks0 = (C_B.shape[1] - 1) // 2
    wb = _kp_weights(C_B.shape[0])[:, None]
    b_centre = C_B[0, ks0].real
    b_lower = b_centre - (jnp.sum(wb * jnp.abs(C_B)) - jnp.abs(C_B[0, ks0]))
    rho_sq = jnp.where(b_lower > 0.0, jnp.square(a_upper) / b_lower, jnp.inf)
    return jnp.sqrt(rho_sq)


def table_bandwidth_cycles(C_A_t, guard):
    """Effective bandwidth of the row's own primitive, cycles per sample.

    ``sqrt(<f^2>)`` over ALL bins of the reflected series' spectrum (the
    spectrum ``_time_primitive_spectrum`` refines with), summed over lanes
    with the ``kp`` weights: the same two-sided convention as
    ``q_effective_bandwidth_hz`` and the ladder record (0.01557 cycles per
    sample on the ladder-2 tables, identical at rho 41, 163 and 652).  The
    RAW moment, not the spread about a carrier: a primitive with content on
    both sides of zero frequency (the (2, -2) mode, a single polarization)
    keeps carrier-scale structure after the angle marginalization, and
    ``<f^2>`` bounds the curvature of anything the band-limited primitive can
    make, so ``1 / (2 pi rho sqrt<f^2>)`` is the NARROWEST peak the row can
    produce at its amplitude.  A face-on envelope is wider (by the ratio of
    the raw to the central moment), which the plan's Newton width reports.
    """
    from .time_first_peaklocal import _time_primitive_spectrum
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    flat = C_A_t.reshape((-1, C_A_t.shape[-1]))
    coeff, frequency, _ = _time_primitive_spectrum(flat, int(guard))
    wa = jnp.broadcast_to(_kp_weights(C_A_t.shape[0])[:, None],
                          C_A_t.shape[:-1]).reshape(-1)
    power = jnp.sum(wa[:, None] * jnp.square(jnp.abs(coeff)), axis=0)
    total = jnp.sum(power)
    msq = jnp.sum(jnp.square(frequency) * power) / total
    return jnp.where(total > 0.0, jnp.sqrt(jnp.maximum(msq, 0.0)), jnp.nan)


def predicted_width_samples(rho, sigma_f_cycles):
    """``sigma_t = 1 / (2 pi rho sigma_f)`` in native samples."""
    rho = jnp.asarray(rho, dtype=jnp.float64)
    sigma_f = jnp.asarray(sigma_f_cycles, dtype=jnp.float64)
    ok = jnp.isfinite(rho) & jnp.isfinite(sigma_f) & (rho > 0.0) & (sigma_f > 0.0)
    return jnp.where(ok, 1.0 / (2.0 * jnp.pi * jnp.where(ok, rho, 1.0)
                                * jnp.where(ok, sigma_f, 1.0)), jnp.nan)


def predict_time_rule(rho, sigma_f_cycles, n_target, *, n_blocks, n_fine,
                      scan_refine, n_scan=None, nodes_per_sigma=NODES_PER_SIGMA):
    """Host-side prediction for the pair selector: what each rule needs.

    Pure numpy on scalars the selector already has.  ``prefer_peaklocal`` is
    true when a whole-window rule at ``nodes_per_sigma`` per predicted sigma
    would need more than twice the peak-local rule's fixed node count.
    ``whole_window_nodes_needed`` is a linear density: a COST comparison
    between two rules, not a convergence model (the trapezoid converges
    spectrally, so the warrant, not this count, certifies a value).
    """
    rho = float(rho)
    sigma_f = float(sigma_f_cycles)
    n_target, n_fine, scan_refine = validate_peaklocal_rule_arguments(
        n_target, n_fine, scan_refine)
    if rho > 0.0 and sigma_f > 0.0 and np.isfinite(rho) and np.isfinite(sigma_f):
        sigma_t = 1.0 / (2.0 * np.pi * rho * sigma_f)
    else:
        sigma_t = float("nan")
    if np.isfinite(sigma_t) and sigma_t > 0.0:
        window_needed = (n_target - 1) * nodes_per_sigma / sigma_t + 1.0
        fine_refine = max(1, int(np.ceil(nodes_per_sigma / (sigma_t * scan_refine))))
    else:
        window_needed = float("inf")
        fine_refine = 1
    peaklocal_nodes, _ = peaklocal_rule_size(n_target, n_blocks, n_fine, scan_refine,
                                             n_scan=n_scan)
    h_f = 1.0 / (scan_refine * fine_refine)
    return dict(rho=rho, sigma_f_cycles=sigma_f, sigma_t_samples=sigma_t,
                fine_refine=fine_refine, fine_spacing_samples=h_f,
                block_span_samples=(n_fine - 1) * h_f,
                whole_window_nodes_needed=window_needed,
                peaklocal_nodes=int(peaklocal_nodes),
                prefer_peaklocal=bool(window_needed > 2.0 * peaklocal_nodes))



# --------------------------------------------------------------- locator
def _angular_grid_field(table, C_B, n_phi, n_u):
    """``A[t, phi, u]`` and ``B[phi, u]`` on a dense phi grid and a u lattice
    from the coefficient tables (``table`` is ``(KP,KS,Nt)``)."""
    phi = jnp.arange(int(n_phi), dtype=jnp.float64) * (2.0 * jnp.pi / int(n_phi))
    u = jnp.arange(int(n_u), dtype=jnp.float64) * (2.0 * jnp.pi / int(n_u))
    kp_a = jnp.arange(table.shape[0], dtype=jnp.float64)
    ks_a = jnp.arange(-(table.shape[1] - 1) // 2, (table.shape[1] - 1) // 2 + 1,
                      dtype=jnp.float64)
    EA = (jnp.exp(1j * (phi[:, None, None, None] * kp_a[None, None, :, None]
                        + u[None, :, None, None] * ks_a[None, None, None, :]))
          * _kp_weights(table.shape[0])[None, None, :, None])
    A = jnp.einsum("pukq,kqt->tpu", EA, table).real
    kp_b = jnp.arange(C_B.shape[0], dtype=jnp.float64)
    ks_b = jnp.arange(-(C_B.shape[1] - 1) // 2, (C_B.shape[1] - 1) // 2 + 1,
                      dtype=jnp.float64)
    EB = (jnp.exp(1j * (phi[:, None, None, None] * kp_b[None, None, :, None]
                        + u[None, :, None, None] * ks_b[None, None, None, :]))
          * _kp_weights(C_B.shape[0])[None, None, :, None])
    B = jnp.einsum("pukq,kq->pu", EB, C_B).real
    return A, B


def _profile(A, B, x_min, x_max):
    """``max_x (x A - x^2 B / 2)`` on ``[x_min, x_max]``; ``-inf`` where the
    norm is not positive."""
    safe = jnp.where(B > 0.0, B, 1.0)
    x = jnp.clip(A / safe, float(x_min), float(x_max))
    val = x * A - 0.5 * jnp.square(x) * B
    return jnp.where(B > 0.0, val, -jnp.inf)


def locate_time_maxima(C_A_t, C_B, guard, n_target, x_min, x_max, *,
                       n_candidates, search_refine=8, angular_lattice=8,
                       n_phi=64, polish_iterations=4, keep_nats=30.0,
                       newton_steps=3, newton_step_max=0.1):
    """Time maxima of the angle- and distance-maximized field of one row.

    Fixed shape.  The ENVELOPE profile ``P(t) = max_{phi,u,x} (x A - x^2
    B / 2)`` is what the angle-marginalized reserve integrates in time, so
    the maxima are found on it: phi on a dense ``n_phi`` grid (the maximum
    over phi of the carrier's phase is what removes the carrier), u on an
    ``angular_lattice`` grid, x in closed form, on a search grid of
    ``search_refine`` nodes per native sample.  The ``n_candidates`` highest
    local maxima are each polished by ``polish_iterations`` parabolic steps
    on ``P(t)`` at the candidate's ``u`` (phi and x re-maximized at every
    evaluation), the step shrinking by four each time.  A fixed-angle Newton
    polish was measured to land up to ``1 / (16 f_c)`` samples (four samples
    on a 64 Hz carrier) from the envelope maximum, because the fixed-angle
    field peaks where its carrier phase does, not where the envelope does.

    Returns a dict of ``(n_candidates,)`` arrays: ``centres`` (native samples
    of the unguarded window), ``widths`` (``1 / sqrt(-P'')``, the peak's
    sigma from the envelope's curvature at the maximum), ``values`` (``P``
    there), ``live`` (finite, interior, curved, within ``keep_nats`` of the
    best), plus ``n_search_nodes``.
    """
    from .time_first_peaklocal import _time_primitive_spectrum, _evaluate_time_spectrum
    C_A_t = jnp.asarray(C_A_t, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    guard = int(guard)
    n_target = int(n_target)
    K = int(n_candidates)
    R = int(search_refine)
    L = int(angular_lattice)
    if K < 1 or R < 1 or L < 2 or int(n_phi) < 4 or int(polish_iterations) < 1:
        raise ValueError("need n_candidates >= 1, search_refine >= 1, "
                         "angular_lattice >= 2, n_phi >= 4, polish_iterations >= 1")
    flat = C_A_t.reshape((-1, C_A_t.shape[-1]))
    coeff, frequency, offset = _time_primitive_spectrum(flat, guard)
    shape = C_A_t.shape[:-1]
    t_s = jnp.arange((n_target - 1) * R + 1, dtype=jnp.float64) / float(R)
    table = _evaluate_time_spectrum(coeff, frequency, t_s, offset).reshape(
        shape + (t_s.size,))
    A, B = _angular_grid_field(table, C_B, n_phi, L)     # (t,phi,u), (phi,u)
    prof = _profile(A, B[None], x_min, x_max)
    P_u = jnp.max(prof, axis=1)                          # (t, u): max over phi
    P = jnp.max(P_u, axis=1)                             # (t,)
    u_idx = jnp.argmax(P_u, axis=1)                      # (t,)
    left = jnp.concatenate((jnp.array([-jnp.inf]), P[:-1]))
    right = jnp.concatenate((P[1:], jnp.array([-jnp.inf])))
    is_max = (P >= left) & (P >= right) & jnp.isfinite(P)
    scores, idx = jax.lax.top_k(jnp.where(is_max, P, -jnp.inf), K)
    t0 = t_s[idx]
    u_star = u_idx[idx]                                  # (K,)

    kp_a = jnp.arange(C_A_t.shape[0], dtype=jnp.float64)
    ks_a = jnp.arange(-(C_A_t.shape[1] - 1) // 2, (C_A_t.shape[1] - 1) // 2 + 1,
                      dtype=jnp.float64)
    kp_b = jnp.arange(C_B.shape[0], dtype=jnp.float64)
    ks_b = jnp.arange(-(C_B.shape[1] - 1) // 2, (C_B.shape[1] - 1) // 2 + 1,
                      dtype=jnp.float64)
    wa = _kp_weights(C_A_t.shape[0])
    wb = _kp_weights(C_B.shape[0])
    ang = jnp.arange(L, dtype=jnp.float64) * (2.0 * jnp.pi / L)
    u_val = ang[u_star]                                  # (K,)
    phi_grid = jnp.arange(int(n_phi), dtype=jnp.float64) * (2.0 * jnp.pi / int(n_phi))

    def _profile_at(phi, u, a_lanes):
        """Scalar profile at one (phi, u) from one time node's table."""
        EA = jnp.exp(1j * (phi * kp_a[:, None] + u * ks_a[None, :])) * wa[:, None]
        A = jnp.sum(EA * a_lanes).real
        EB = jnp.exp(1j * (phi * kp_b[:, None] + u * ks_b[None, :])) * wb[:, None]
        Bv = jnp.sum(EB * C_B).real
        return _profile(A, Bv, x_min, x_max)

    def _profile_ang(ang, a_lanes):
        return _profile_at(ang[0], ang[1], a_lanes)

    grad_ang = jax.grad(_profile_ang)
    hess_ang = jax.hessian(_profile_ang)

    def _envelope(t, return_angles=False):
        """``P(t)`` at the candidates' own (phi, u), both re-maximized: the
        best of a dense phi grid at the lattice u, then two Newton steps in
        (phi, u) jointly on the trigonometric polynomial.  A grid maximum
        alone leaves a ripple of period ``1 / (n_phi f_c)`` in t that a
        parabola reads as curvature, and a lattice u moves the t-maximum
        (measured 0.17 samples, 2.7 sigma, on a rung-160 production row)."""
        tab = _evaluate_time_spectrum(coeff, frequency, t, offset).reshape(
            shape + (t.size,))                            # (KP,KS,K)
        lanes = jnp.moveaxis(tab, -1, 0)                  # (K,KP,KS)
        grid = jax.vmap(lambda a, u: jax.vmap(
            lambda p: _profile_at(p, u, a))(phi_grid))(lanes, u_val)   # (K,n_phi)
        ang = jnp.stack((phi_grid[jnp.argmax(grid, axis=1)], u_val), axis=1)  # (K,2)

        def _newton(a, _):
            g = jax.vmap(grad_ang)(a, lanes)                           # (K,2)
            h = jax.vmap(hess_ang)(a, lanes)                           # (K,2,2)
            # Newton on a maximum.  A direction with no curvature (a table
            # with no u content has h_uu = 0 exactly) must not stall the
            # other: ridge the Hessian and zero the step in any direction
            # whose own curvature is not negative.
            ridge = 1.0e-8 * (1.0 + jnp.abs(h[:, 0, 0]) + jnp.abs(h[:, 1, 1]))
            h_reg = h - ridge[:, None, None] * jnp.eye(2)[None]
            step = -jnp.linalg.solve(h_reg, g[..., None])[..., 0]
            concave = jnp.stack((h[:, 0, 0] < 0.0, h[:, 1, 1] < 0.0), axis=1)
            step = jnp.where(concave, jnp.clip(step, -float(newton_step_max),
                                               float(newton_step_max)), 0.0)
            return a + step, None

        ang, _ = jax.lax.scan(_newton, ang, None, length=int(newton_steps))
        vals = jax.vmap(_profile_ang)(ang, lanes)
        return (vals, ang) if return_angles else vals

    def _step(carry, _):
        t, delta = carry
        p0 = _envelope(t)
        pp = _envelope(jnp.clip(t + delta, 0.0, float(n_target - 1)))
        pm = _envelope(jnp.clip(t - delta, 0.0, float(n_target - 1)))
        curv = (pp + pm - 2.0 * p0) / jnp.square(delta)
        move = jnp.where(curv < 0.0, -0.5 * delta * (pp - pm)
                         / jnp.where(curv < 0.0, pp + pm - 2.0 * p0, -1.0), 0.0)
        move = jnp.clip(move, -delta, delta)
        t_new = jnp.clip(t + move, 0.0, float(n_target - 1))
        return (t_new, delta * 0.25), curv

    delta0 = jnp.full((K,), 1.0 / float(R))
    (t_ref, _), curvs = jax.lax.scan(_step, (t0, delta0), None,
                                     length=int(polish_iterations))
    curv = curvs[-1]
    values, angles = _envelope(t_ref, return_angles=True)
    curved = curv < 0.0
    widths = jnp.where(curved, 1.0 / jnp.sqrt(jnp.where(curved, -curv, 1.0)), jnp.inf)
    best = jnp.max(values)
    # The profile maximum IS the row's lnL maximum, so rho^2 = 2 P_max: the
    # amplitude the prediction should use.  The angular triangle bound is an
    # upper bound that ran 2.8x over on a rung-160 production row (453
    # against 157), which narrowed the predicted width and the block span by
    # the same factor.
    rho_located = jnp.where(jnp.isfinite(best) & (best > 0.0),
                            jnp.sqrt(2.0 * jnp.maximum(best, 0.0)), jnp.nan)
    live = (jnp.isfinite(values) & curved & jnp.isfinite(scores)
            & (values >= best - float(keep_nats))
            & (t_ref > 0.0) & (t_ref < float(n_target - 1)))
    return dict(centres=t_ref, widths=widths, values=values, live=live,
                phi=angles[:, 0], u=angles[:, 1], rho_located=rho_located,
                search_centres=t0, n_search_nodes=jnp.asarray(t_s.size),
                search_positions=t_s, search_profile=P)


# ------------------------------------------------------------------ rule
def _trapezoid_weights(sorted_nodes):
    """Trapezoid weights on a sorted, possibly repeated, rule (sample units)."""
    left = jnp.concatenate((sorted_nodes[:1], sorted_nodes[:-1]))
    right = jnp.concatenate((sorted_nodes[1:], sorted_nodes[-1:]))
    # Clamped: a repeated position differs by a rounding unit under XLA's
    # fused subtraction, and a weight of -1e-16 would fail the kernel's
    # non-negativity check.
    return jnp.maximum(0.5 * (right - left), 0.0)


def peaklocal_time_rule(centres, widths, live, n_target, dt_scale, *,
                        sigma_t_samples, n_fine, scan_refine,
                        fine_refine_multiplier=1,
                        nodes_per_sigma=NODES_PER_SIGMA,
                        n_scan=None, margin_samples=None,
                        search_positions=None, search_profile=None,
                        outside_slack_nats=5.0):
    """Per-row reserve rule and check rule on one commensurate lattice.

    ``centres``, ``widths`` (sigma at each maximum, samples) and ``live`` are
    the row's time maxima (:func:`locate_time_maxima`).  ``sigma_t_samples``
    is the predicted width (traced scalar); the fine spacing is the smaller
    of the prediction and the narrowest live maximum, divided by
    ``nodes_per_sigma``, so the lattice is never coarser than either says.
    ``dt_scale`` is the seconds-per-sample constant of the production rule
    (``sum(w_t) / (npts - 1)``) so the weights land in the units of
    ``policy_time_rules``.  ``fine_refine_multiplier`` is the escalation
    tier (1, 2, 4, ...): the fine lattice is refined by it and ``n_fine``
    must have been widened to match (``n -> 2n - 1``) so the block span is
    unchanged.

    With ``n_scan`` the scan is SUPPORT-LIMITED: ``n_scan`` nodes on the
    scan lattice across the hull of the live maxima widened by
    ``margin_samples`` each side (never spanning the window), and the mass
    outside the hull is bounded from the locator's ``search_profile`` (the
    angle- and distance-maximized exponent, an upper bound on the marginal at
    each search node) summed over the outside search nodes, plus
    ``outside_slack_nats`` for what lies between nodes.  That bound goes to
    the kernel's cropped-cover warrant, which charges it to the error budget.
    A node count that grows with the window is the wrong design (RO,
    2026-09-09); the count here is ``n_scan + n_blocks n_fine`` whatever the
    window.  Returns a dict with the fixed-shape ``nodes``, ``weights``,
    ``check_nodes``, ``check_weights`` and the numbers behind them.
    """
    n_target, n_fine, scan_refine = validate_peaklocal_rule_arguments(
        n_target, n_fine, scan_refine)
    mult = int(fine_refine_multiplier)
    if mult < 1:
        raise ValueError("fine_refine_multiplier must be >= 1")
    centres = jnp.asarray(centres, dtype=jnp.float64)
    widths = jnp.asarray(widths, dtype=jnp.float64)
    live = jnp.asarray(live).astype(bool) & jnp.isfinite(centres)
    sigma_pred = jnp.asarray(sigma_t_samples, dtype=jnp.float64)
    located = jnp.min(jnp.where(live & (widths > 0.0), widths, jnp.inf))
    sigma_t = jnp.minimum(jnp.where(jnp.isfinite(sigma_pred) & (sigma_pred > 0.0),
                                    sigma_pred, jnp.inf), located)
    finite = jnp.isfinite(sigma_t) & (sigma_t > 0.0)
    need = float(nodes_per_sigma) / (jnp.where(finite, sigma_t, 1.0) * scan_refine)
    m_pred = jnp.where(finite, jnp.maximum(1.0, jnp.ceil(need)), 1.0)
    # Guard the lattice against an absurd prediction: the fine index space
    # is (n_target - 1) * scan_refine * m long and must stay exact in
    # float64 and cheap to build.
    m_pred = jnp.minimum(m_pred, 2.0 ** 20).astype(jnp.int64)
    m = m_pred * mult
    h_f = 1.0 / (scan_refine * m).astype(jnp.float64)
    n_index = (n_target - 1) * scan_refine * m        # last fine index

    half_span = 0.5 * (n_fine - 1)                     # in fine indices
    start = jnp.round(centres / h_f - half_span)
    start = jnp.clip(start, 0.0, jnp.maximum(0.0, n_index - (n_fine - 1)))
    start = jnp.where(live, start, 0.0).astype(jnp.int64)
    k = jnp.arange(n_fine)
    fine_live_idx = jnp.minimum(start[:, None] + k[None, :], n_index)   # (K, n_fine)
    cropped = n_scan is not None
    if not cropped:
        scan_idx = jnp.arange((n_target - 1) * scan_refine + 1) * m
        scan_idx_c = jnp.arange((n_target - 1) * (scan_refine // 2) + 1) * (2 * m)
        hull_lo = jnp.asarray(0.0)
        hull_hi = jnp.asarray(float(n_target - 1))
        outside_log_bound = jnp.asarray(-jnp.inf)
        n_outside = jnp.asarray(0, dtype=jnp.int32)
    else:
        n_s = int(n_scan)
        if n_s < 3 or n_s % 2 == 0:
            raise ValueError("n_scan must be an odd integer >= 3")
        if margin_samples is None or search_positions is None or search_profile is None:
            raise ValueError("a cropped scan needs margin_samples, "
                             "search_positions and search_profile")
        margin = jnp.asarray(margin_samples, dtype=jnp.float64)
        any_live = jnp.any(live)
        c_lo = jnp.min(jnp.where(live, centres, jnp.inf))
        c_hi = jnp.max(jnp.where(live, centres, -jnp.inf))
        hull_lo = jnp.where(any_live, jnp.clip(c_lo - margin, 0.0, float(n_target - 1)), 0.0)
        hull_hi = jnp.where(any_live, jnp.clip(c_hi + margin, 0.0, float(n_target - 1)),
                            float(n_target - 1))
        # Scan lattice indices (multiples of m) across the hull: a step of
        # whole scan cells, at least one, so the scan stays on the lattice.
        lo_idx = jnp.floor(hull_lo / h_f / m).astype(jnp.int64) * m
        hi_idx = jnp.minimum(jnp.ceil(hull_hi / h_f / m).astype(jnp.int64) * m, n_index)
        step = jnp.maximum(1, jnp.ceil((hi_idx - lo_idx) / (m * (n_s - 1))).astype(jnp.int64)) * m
        scan_idx = jnp.minimum(lo_idx + jnp.arange(n_s) * step, n_index)
        scan_idx_c = jnp.minimum(lo_idx + jnp.arange((n_s - 1) // 2 + 1) * (2 * step), n_index)
        hull_lo = lo_idx.astype(jnp.float64) * h_f
        hull_hi = jnp.minimum(lo_idx + (n_s - 1) * step, n_index).astype(jnp.float64) * h_f
        t_s = jnp.asarray(search_positions, dtype=jnp.float64)
        P_s = jnp.asarray(search_profile, dtype=jnp.float64)
        outside = (t_s < hull_lo) | (t_s > hull_hi)
        ds = (t_s[1] - t_s[0]) * float(dt_scale)
        outside_log_bound = (jax.scipy.special.logsumexp(jnp.where(outside, P_s, -jnp.inf))
                             + jnp.log(ds) + float(outside_slack_nats))
        n_outside = jnp.count_nonzero(outside).astype(jnp.int32)
    # A dead slot collapses onto the scan's first node: exact duplicates,
    # zero weight, and no gap that the check rule would share.
    fine_idx = jnp.where(live[:, None], fine_live_idx, scan_idx[0])
    idx = jnp.sort(jnp.concatenate((scan_idx, fine_idx.ravel())))
    nodes = idx.astype(jnp.float64) * h_f
    # Check rule: the half-refined scan and every other fine node, on the
    # same lattice, same endpoints, everywhere coarser.
    fine_idx_c = fine_idx[:, ::2]
    idx_c = jnp.sort(jnp.concatenate((scan_idx_c, fine_idx_c.ravel())))
    check_nodes = idx_c.astype(jnp.float64) * h_f
    weights = _trapezoid_weights(nodes) * float(dt_scale)
    check_weights = _trapezoid_weights(check_nodes) * float(dt_scale)
    n_live = jnp.count_nonzero(live).astype(jnp.int32)
    first_centre = jnp.where(n_live > 0, centres[jnp.argmax(live)], jnp.nan)
    return dict(nodes=nodes, weights=weights, check_nodes=check_nodes,
                check_weights=check_weights,
                scan_lo_samples=hull_lo, scan_hi_samples=hull_hi,
                outside_log_bound=outside_log_bound,
                n_outside_search_nodes=n_outside,
                sigma_t_pred_samples=sigma_pred,
                sigma_t_located_samples=located,
                sigma_t_used_samples=sigma_t,
                fine_refine=m.astype(jnp.int32),
                fine_spacing_samples=h_f,
                block_span_samples=(n_fine - 1) * h_f,
                n_live_blocks=n_live,
                first_block_centre_samples=first_centre,
                prediction_finite=finite)
