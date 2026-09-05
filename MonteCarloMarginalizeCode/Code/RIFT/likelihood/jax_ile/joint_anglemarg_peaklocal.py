"""Joint (phi, psi) peak-local angle marginalization, JAX kernel.

The numpy reference is ``RIFT.likelihood.joint_angle_peak_local``; this is the jittable
form of the same rule.  It is NOT a transcription -- the reference builds 2-D regions
and merges overlapping ones, which is data-dependent control flow and does not jit.  The
formulation here removes the need to merge at all.

THE PARTITION THAT REPLACES MERGING.  At fixed ``phi`` the exponent is
``a + Re(c1 e^{iu}) + Re(c2 e^{2iu})``, whose u-stationary points are the roots of a
quartic.  On the circle maxima and minima ALTERNATE, so the sorted stationary points
already tile the domain: the cell of a maximum is the arc between its two neighbouring
minima.  Those cells are disjoint by construction and cover the circle, so there is
nothing to merge and nothing to double-count -- the failure the reference spends
``_merge_boxes`` on cannot arise.  Everything is then static at trace time: 4 roots,
4 candidate cells, and an amplitude-derived quadrature count streamed in fixed blocks.

WHY THE ROOTS ARE TAKEN WITHOUT A ``|z| = 1`` FILTER.  At exact multiplicity the
computed roots smear off the unit circle by ``eps^(1/m)`` -- measured 4.6e-6 for a
triple root -- so a fixed tolerance drops real modes in precisely the degenerate regime
that is normal here.  Every root contributes its angle; a spurious one produces a
zero-length or redundant cell, which is harmless, whereas a dropped one loses mass.

WHAT SCALES WITH AMPLITUDE AND WHAT DOES NOT.  The stationary points of ``g`` do not
move when the data amplitude grows -- ``g -> lambda g`` leaves them fixed -- so the
CELLS are amplitude-independent, while the peak inside each cell narrows as
``A^-1/2``.  A local window therefore needs a fixed count, but a rejected Newton centre
falls back to a whole cell and needs ``~sqrt(A)`` nodes.  Production uses that conservative
count for every cell because fallback is data-dependent; streaming preserves the memory
economy even though the arithmetic cost is no longer claimed constant.

SCOPE OF THIS KERNEL.  The u axis is localized; the phi axis is a dense grid, scanned
in chunks.  That is deliberately the same cost shape as the shipped ``laplace`` scheme
(``~sqrt(A)`` on phi) and a strict improvement on its u treatment, which uses a blended
O(1/A) width model rather than the exact stationary points.  Localizing phi as well --
the (phi localized, psi localized) cell of the family -- needs the profile ``F(phi)``
and its envelope derivative, and is not attempted here.

MEMORY.  Bounded by ``phi_chunk`` and ``U_NODE_STREAM_CHUNK`` through rolled loops, never
by the full phi or u grids: the largest u transient is
``(phi_chunk, n_x, 4, U_NODE_STREAM_CHUNK)``.  These are cost knobs and cannot change the
result beyond floating-point reassociation.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "required_n_phi",
    "required_u_nodes",
    "u_nodes_in_use",
    "U_WINDOW_SIGMA",
    "U_NODES_PER_CELL",
    "U_NODE_STREAM_CHUNK",
    "PHI_CHUNK_DEFAULT",
    "u_stationary_roots",
    "log_inner_u_integral",
    "joint_lnL_phi_dense",
    "u_profile",
    "eval_g2",
    "phi_local_lnI",
    "PHI_SEEDS",
    "PHI_WINDOW_SIGMA",
    "PHI_NODES_PER_REGION",
    "PHI_BOUND_GRID",
    "OUTSIDE_TOL_NATS",
    "phi_derivative_bound",
    "profile_derivative_bounds",
]

#: Local u-window half-width in units of the local sigma, CLIPPED to the cell.  The cell
#: boundaries are minima of the exponent, so clipping loses nothing that the cell itself
#: does not already exclude; this only decides where the window stops being the binding
#: constraint.
U_WINDOW_SIGMA = 12.0

#: Trapezoid nodes per cell.  Fixed, because the window is sized from the LOCAL
#: curvature: at 48 nodes over +-12 sigma the spacing is sigma/2, and the trapezoid's
#: Poisson-summation error on a Gaussian is 2 exp(-2 pi^2 * 4) = 5e-35.  Derived from
#: that bound, not tuned -- the same argument as UPSAMPLE_SAFETY in the band-limited
#: time quadrature.  This is the u axis's entire cost: 4 cells x 48 nodes = 192 points
#: per phi, INDEPENDENT of amplitude, against the shipped dense rule's ~6.2 sqrt(A)
#: (896 at amplitude 1.25e4).
#:
#: THAT AMPLITUDE-INDEPENDENCE HOLDS FOR A WINDOWED CELL AND NOT FOR A FALLBACK ONE.
#: A cell whose Newton centre is rejected (stalled on a boundary, large stationary
#: residual) is integrated WHOLE, and 48 nodes then span the entire cell rather than
#: +-12 sigma.  The numpy twin measured 1.7e-03 nats of inner-u error that way, so the
#: honest statement is: this default resolves WINDOWED cells at any amplitude.  The
#: production caller may hit a fallback at any phi/distance point, so it uses the
#: amplitude-derived :func:`u_nodes_in_use` policy instead of relying on this floor.
U_NODES_PER_CELL = 48

#: Maximum number of u nodes materialized at once.  The production count grows as
#: sqrt(amplitude), but the quadrature is accumulated through a rolled scan so that its
#: live node axis -- and therefore the batch-memory model -- stays bounded.
U_NODE_STREAM_CHUNK = 8

#: phi points per scan step.
PHI_CHUNK_DEFAULT = 16


def u_nodes_in_use(amp_sizing=None):
    """The u-node count the peak-local kernel WILL ACTUALLY REQUEST at this amplitude.

    SINGLE SOURCE OF TRUTH, and it exists because the batch-memory guard in
    :mod:`~RIFT.likelihood.jax_ile.samplers` has to model the same number the kernel
    requests, and the two are in different files.  External review found the trap before
    it fired: the guard hard-coded ``U_NODES_PER_CELL``, so anyone wiring
    :func:`required_u_nodes` into the kernel would silently invalidate it -- at the
    production floor ``amp_sizing = 450`` that is 896 nodes against a modeled 48, and the
    documented live slab goes from 3.6 GiB to 67 GiB at chunk one.  An automated agent
    then did exactly that wiring, and left the guard untouched, which is the trap firing.

    Both the kernel (:func:`joint_lnL_phi_dense`, whose ``n_nodes`` defaults to ``None``
    and resolves here) and the guard call this, and the fused caller passes the same
    ``amp_sizing`` to both.  An earlier version of this docstring claimed that while only
    the guard called it and the kernel still defaulted straight to ``U_NODES_PER_CELL`` --
    a single source of truth that only one side read, which is no single source of truth
    at all and is exactly the divergence this helper exists to prevent.  Caught in review.

    A direct low-level call without an amplitude retains the validated 48-node windowed
    floor.  Production always supplies ``amp_sizing`` and therefore gets the derived,
    uncapped whole-cell requirement.  The quadrature streams that count in
    ``U_NODE_STREAM_CHUNK``-sized blocks, so accuracy grows with amplitude without making
    the live node dimension grow with it.
    """
    if amp_sizing is None:
        return U_NODES_PER_CELL
    return required_u_nodes(amp_sizing)


def required_u_nodes(amplitude, pts_per_sigma=3.0, cap=None):
    """u nodes per cell adequate for a FALLBACK (whole-cell) integration at ``amplitude``.

    Derived, not tuned.  The u-spectrum has two terms, so ``|d2g/du2| <= M2u`` exactly,
    and at exponent amplitude ``A`` the coefficients scale with ``A`` giving
    ``M2u ~ 5 A``: nothing on this axis is narrower than ``sigma_min = 1/sqrt(M2u)``, and
    a spacing of ``sigma_min / pts_per_sigma`` resolves the sharpest feature the
    coefficients admit.  A fallback cell can span most of the circle, so the requirement
    is ``2 pi * sqrt(M2u) * pts_per_sigma``.

    JAX NEEDS THIS STATICALLY, which is why it is a caller-side helper rather than an
    adaptation inside the kernel: shapes cannot depend on traced values.  The numpy twin
    derives the same quantity per call because it can.

    ``cap`` is available only for explicit diagnostic callers.  It is deliberately
    ``None`` in production: truncating the requested count recreates the inside-cover
    accuracy failure this policy exists to prevent.  Memory is bounded independently by
    streaming the node axis rather than by silently reducing the quadrature.
    """
    a = max(float(amplitude), 1.0)
    need = int(np.ceil(2.0 * np.pi * np.sqrt(5.0 * a) * float(pts_per_sigma))) + 1
    need = max(need, U_NODES_PER_CELL)
    return int(need if cap is None else min(need, int(cap)))


def required_n_phi(amplitude, m_max=2):
    """phi-grid size for a given exponent amplitude.

    THE PHI AXIS IS NOT LOCALIZED IN THIS KERNEL, so it inherits the dense scaling and
    must be sized, not guessed: ``exp(g)`` has phi-width ``~A^-1/2`` and its harmonic
    content reaches ``~6.2 sqrt(A)`` (measured), scaled by mode content.  Hard-coding a
    value instead cost 191 nats at amplitude 1.25e4 during development -- recorded
    because a fixed grid looks harmless right up to the point where it is not.

    Mirrors the phi half of ``anglemarg._dense_grid_sizes`` so the two rules are sized
    by one argument rather than two that must agree.
    """
    from . import anglemarg as _am
    return int(_am._dense_grid_sizes(float(amplitude), m_max=int(m_max))[0])


def _a_c1_c2(C, phi):
    """The u-independent term and the ``e^{iu}``, ``e^{2iu}`` coefficients at ``phi``.

    The u-independent term is NOT the single ``(k=0, q=0)`` coefficient: the whole
    ``q = 0`` column is u-independent and every one of its ``k`` harmonics depends on
    phi.  Taking only ``C[0, KS]`` drops that phi structure entirely and biases the
    result low by an amount that grows with amplitude -- measured -2.0e-3, -2.6 and
    -191 nats at exponent amplitude 6.6, 624 and 1.25e4 before this was fixed.
    """
    KP = C.shape[0]
    KS = (C.shape[1] - 1) // 2
    k = jnp.arange(KP)
    w = jnp.where(k > 0, 2.0, 1.0)
    ph = jnp.exp(1j * phi[..., None] * k) * w
    D = lambda q: (ph * C[:, KS + q]).sum(-1)
    # BOTH SIGNS OF q ARE STORED, so the e^{iu} coefficient is not the +1 column alone:
    # Re[D_{-q} e^{-iqu}] = Re[conj(D_{-q}) e^{+iqu}], hence c_q = D_{+q} + conj(D_{-q}).
    # Using only the +q column drops half the u-dependence.  It is survivable where the
    # roots are mere SEEDS -- the numpy reference refines them with 2-D Newton and was
    # unaffected -- but here the roots define the cell PARTITION, which is load-bearing,
    # and the error reached 17 nats at a single phi.
    a = D(0).real
    return a, D(1) + jnp.conj(D(-1)), D(2) + jnp.conj(D(-2))


def u_stationary_roots(c1, c2):
    """The four u-stationary angles, as a companion eigenproblem.  Static shape (4,).

    ``P(z) = c2 z^4 + (c1/2) z^3 - (conj(c1)/2) z - conj(c2)``; the roots' arguments are
    the stationary points.  No ``|z| = 1`` filtering -- see the module docstring.
    """
    a4 = c2
    lead = jnp.where(jnp.abs(a4) > 0, a4, 1.0 + 0j)
    co = jnp.stack([c1 / 2.0, jnp.zeros_like(c1), -jnp.conj(c1) / 2.0,
                    -jnp.conj(c2)]) / lead
    comp = jnp.zeros((4, 4), dtype=jnp.complex128)
    comp = comp.at[0, :].set(-co)
    comp = comp.at[1:, :-1].set(jnp.eye(3, dtype=jnp.complex128))
    # the stop_gradient goes on the INPUT: placing it on the output still leaves JAX
    # needing eigvals' JVP rule to build the trace, and that is the rule that does not
    # exist.  Cutting the tangent before the eigensolve means it is never asked for.
    z = jnp.linalg.eigvals(jax.lax.stop_gradient(comp))
    # a vanishing quartic leading coefficient degenerates to a cubic; the extra root is
    # spurious but produces only a redundant cell, never a lost one.
    #
    # STOP_GRADIENT, and it is a correctness statement rather than a convenience.
    # (i) It is REQUIRED: jnp.linalg.eigvals has no second derivative in JAX ("the
    #     derivatives of eigenvectors are not implemented"), so without it any Hessian
    #     through this kernel raises -- and the caller that matters, _fisher_whitening,
    #     swallows that in an `except Exception` and silently returns None, so
    #     --fisher-precondition would degrade to raw coordinates with the flag still
    #     recorded as supplied.  It also removes a NaN: as c2 -> 0 the companion matrix
    #     acquires ~1/c2 entries and the eig JVP degenerates (measured grad 0.567 at
    #     c2=1, -1.2e14 at 1e-20, nan at 1e-30).
    # (ii) It is CORRECT: these angles are cell BOUNDARIES of an exact partition of the
    #     circle, so a boundary shift adds to one cell exactly what it removes from its
    #     neighbour and the contribution cancels identically.  Where a window stops short
    #     of its cell edge the integrand there is ~exp(-W^2/2) of the peak, so that
    #     residue is far below the truncation already accepted.  The same argument, and
    #     the same device, is used for the distance nodes in core._distmarg_gh_logL.
    return jnp.mod(jnp.angle(z), 2.0 * jnp.pi)


def _g_u(a, c1, c2, u, order=0):
    """``d^order/du^order`` of ``a + Re(c1 e^{iu}) + Re(c2 e^{2iu})``."""
    t1 = ((1j) ** order) * c1 * jnp.exp(1j * u)
    t2 = ((2j) ** order) * c2 * jnp.exp(2j * u)
    base = (t1 + t2).real
    return base + (a if order == 0 else 0.0)


def log_inner_u_integral(a, c1, c2, n_nodes=U_NODES_PER_CELL,
                         window_sigma=U_WINDOW_SIGMA):
    """``log int_0^{2pi} du exp(a + Re(c1 e^{iu}) + Re(c2 e^{2iu}))``.

    Exact partition, static shapes.  The sorted stationary points alternate max/min, so
    the cell of the maximum at ``u_(i)`` is ``[u_(i-1), u_(i+1)]`` and the maxima's cells
    tile the circle.  Non-maxima contribute a masked ``-inf`` and drop out of the
    log-sum-exp, so no filtering or compaction is needed.
    """
    u = jnp.sort(u_stationary_roots(c1, c2))                     # (4,)

    # MIDPOINT cells, not root-bounded cells.  Bounding a maximum's cell by its
    # neighbouring roots is only a partition if maxima and minima strictly ALTERNATE,
    # and they do not: when two roots leave the unit circle as a conjugate-reciprocal
    # pair (measured: |z| = 1.268 and 0.789, product 1.0003) their shared angle is not a
    # stationary point at all, and using it as a boundary leaves an arc belonging to no
    # cell.  That silently dropped 0.23 nats on a low-amplitude draw.  Midpoints of the
    # sorted angles tile the circle for ANY four angles, so no arc can be orphaned
    # however the roots behave -- which is what lets the roots stay unfiltered.
    mid = 0.5 * (u + jnp.roll(u, -1) + jnp.where(jnp.arange(4) == 3, 2 * jnp.pi, 0.0))
    lo_c = jnp.roll(mid, 1) - jnp.where(jnp.arange(4) == 0, 2 * jnp.pi, 0.0)
    hi_c = mid

    # locate the maximum INSIDE each cell: the cell's own root is only a seed, and for a
    # spurious root it is not even a stationary point.
    def _newton(uc, _):
        g1 = _g_u(a, c1, c2, uc, 1)
        g2 = _g_u(a, c1, c2, uc, 2)
        step = jnp.where(jnp.abs(g2) > 0, -g1 / jnp.where(jnp.abs(g2) > 0, g2, 1.0), 0.0)
        step = jnp.clip(step, -0.5, 0.5)
        return jnp.clip(uc + step, lo_c, hi_c), None

    ustar, _ = lax.scan(_newton, u, None, length=8)

    # A CLIPPED NEWTON POINT IS NOT A PEAK, however negative the curvature.  The
    # iteration is clamped to [lo_c, mid], so it can come to rest ON a boundary with a
    # large stationary residual; curvature alone then centres a +-W sigma window on a
    # non-stationary point and sizes sigma from the wrong curvature.  Measured in the
    # numpy twin: 18% of cells that g'' < 0 accepted fail this gate, the worst at
    # |g_u|/M_1 = 0.33.  A cell failing it is integrated WHOLE -- which ADDS NO NODES, it
    # spreads the same n_nodes over the whole cell, so the fallback is COARSER than the
    # window it replaces.  (An earlier comment here claimed "can only add nodes"; that was
    # wrong, and the numpy twin measured the inner-u error recorded on
    # U_NODES_PER_CELL from it.)  JAX
    # cannot adapt n_nodes -- shapes may not depend on traced values -- so the sizing is
    # exposed to the caller as required_u_nodes() rather than fixed here; see its docstring
    # for why raising it by default is the wrong trade.
    g1s = _g_u(a, c1, c2, ustar, 1)
    g2s = _g_u(a, c1, c2, ustar, 2)
    m1u = jnp.abs(c1) + 2.0 * jnp.abs(c2)          # exact bound on |d g / du|
    edge = 1e-9 * jnp.max(mid - lo_c)
    peaked = ((g2s < 0.0)
              & (jnp.abs(g1s) <= 1e-8 * jnp.maximum(m1u, 1e-300))
              & (ustar > lo_c + edge) & (ustar < mid - edge))
    sigma = jnp.where(peaked, 1.0 / jnp.sqrt(jnp.where(peaked, -g2s, 1.0)), jnp.inf)
    # a cell with no interior maximum is integrated whole; a peaked one is integrated on
    # +-window_sigma, which is self-limiting -- when the integrand is flat sigma is large
    # and the window IS the cell.
    lo = jnp.where(peaked, jnp.maximum(ustar - window_sigma * sigma, lo_c), lo_c)
    hi = jnp.where(peaked, jnp.minimum(ustar + window_sigma * sigma, hi_c), hi_c)
    width = jnp.maximum(hi - lo, 0.0)

    # STREAM THE NODE AXIS.  Materializing (4, n_nodes) here is multiplied by the outer
    # phi, distance, time and sample batches.  At the production floor the accurate
    # fallback policy asks for 896 nodes, which would turn the documented 48-node live
    # slab into ~67 GiB even at sample chunk one.  A rolled scan keeps only
    # U_NODE_STREAM_CHUNK nodes live while accumulating the identical trapezoid sum.
    n_nodes = int(n_nodes)
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    n_blocks = int(np.ceil(n_nodes / U_NODE_STREAM_CHUNK))
    local_idx = jnp.arange(U_NODE_STREAM_CHUNK)

    def _node_block(block_i, log_sum):
        idx = block_i * U_NODE_STREAM_CHUNK + local_idx
        live = idx < n_nodes
        s = idx / float(n_nodes - 1)
        uu = lo[:, None] + width[:, None] * s[None, :]
        gg = _g_u(a, c1, c2, uu, 0)
        endpoint = (idx == 0) | (idx == n_nodes - 1)
        log_trap = jnp.where(endpoint, -jnp.log(2.0), 0.0)
        terms = jnp.where(live[None, :], gg + log_trap[None, :], -jnp.inf)
        block = jax.scipy.special.logsumexp(terms, axis=-1)
        return jnp.logaddexp(log_sum, block)

    cell_sum = lax.fori_loop(0, n_blocks, jax.checkpoint(_node_block),
                             jnp.full(4, -jnp.inf))
    log_scale = jnp.log(jnp.where(width > 0, width, 1.0)) - jnp.log(n_nodes - 1)
    cell = cell_sum + log_scale
    cell = jnp.where(width > 0, cell, -jnp.inf)
    return jax.scipy.special.logsumexp(cell)


def _joint_table(C_A, C_B, x):
    """``x A - x^2/2 B`` with A zero-padded into B's (larger) bidegree."""
    ksa = (C_A.shape[1] - 1) // 2
    ksb = (C_B.shape[1] - 1) // 2
    out = (-0.5 * x * x) * C_B
    return out.at[:C_A.shape[0], ksb - ksa:ksb + ksa + 1].add(x * C_A)


def joint_lnL_phi_dense(C_A, C_B, x_grid, log_w_grid, n_phi=256,
                        phi_chunk=PHI_CHUNK_DEFAULT,
                        n_nodes=None):
    """Distance-, phi- and psi-marginalized value at one ``(sample, time)``.

    Same normalization as ``anglemarg.fused_log_likelihood_distphipsimarg_*``: uniform
    priors ``dphi/2pi`` and ``dpsi/pi``, which in ``u = 2 psi`` is ``(2 pi)^-2`` times
    the torus integral.

    ``phi`` is a dense grid scanned in chunks; ``u`` is exact per the cell partition.
    """
    if n_nodes is None:
        n_nodes = u_nodes_in_use()
    C_A = jnp.asarray(C_A, dtype=jnp.complex128)
    C_B = jnp.asarray(C_B, dtype=jnp.complex128)
    x_grid = jnp.asarray(x_grid, dtype=jnp.float64).ravel()
    log_w_grid = jnp.asarray(log_w_grid, dtype=jnp.float64).ravel()
    KS = (C_B.shape[1] - 1) // 2

    tables = jax.vmap(lambda x: _joint_table(C_A, C_B, x))(x_grid)   # (nx, KP, 2KS+1)

    phis = jnp.linspace(0.0, 2.0 * jnp.pi, n_phi, endpoint=False)
    n_chunk = int(np.ceil(n_phi / phi_chunk))
    pad = n_chunk * phi_chunk - n_phi
    phis_p = jnp.concatenate([phis, jnp.zeros(pad)])
    live = jnp.concatenate([jnp.ones(n_phi, bool), jnp.zeros(pad, bool)])

    def one_phi(phi):
        a, c1, c2 = jax.vmap(lambda T: _a_c1_c2(T, jnp.atleast_1d(phi)))(tables)
        return jax.vmap(log_inner_u_integral, in_axes=(0, 0, 0, None))(
            a[:, 0], c1[:, 0], c2[:, 0], n_nodes)                    # (nx,)

    def step(carry, args):
        ph, lv = args
        vals = jax.vmap(one_phi)(ph)                                 # (chunk, nx)
        vals = jnp.where(lv[:, None], vals, -jnp.inf)
        return carry, vals

    # jax.checkpoint on the scan body, as the shipped exact scheme does.  Without it a
    # REVERSE-mode pass keeps every chunk's intermediates: the wrapper's Hessian tried to
    # allocate 135 GB and died RESOURCE_EXHAUSTED, so --fisher-precondition would have
    # OOMed rather than run.  Forward evaluation was never affected, which is exactly why
    # this was invisible until a second derivative was taken.
    _, out = lax.scan(jax.checkpoint(step), None,
                      (phis_p.reshape(n_chunk, phi_chunk),
                       live.reshape(n_chunk, phi_chunk)))
    vals = out.reshape(n_chunk * phi_chunk, -1)[:n_phi]           # (n_phi, nx)

    # phi is a periodic trapezoid == plain mean; then the distance sum; then (2pi)^-2
    per_x = jax.scipy.special.logsumexp(vals, axis=0) - jnp.log(n_phi) \
        + jnp.log(2.0 * jnp.pi)
    return jax.scipy.special.logsumexp(per_x + log_w_grid) - 2.0 * jnp.log(2.0 * jnp.pi)


# ------------------------------------------------------- phi localization

#: phi seeds.  These are SEEDS, not a quadrature grid: Newton moves each to a maximum of
#: the profile and overlapping windows merge, so the count sets how many distinct modes
#: can be found, not the accuracy.  It does not scale with amplitude -- the number of
#: maxima of F is set by the bidegree, which is mode content, not SNR.
PHI_SEEDS = 32

#: phi window half-width in units of the profile's local sigma, and nodes per region.
#: Same Poisson-summation argument as U_NODES_PER_CELL: at +-12 sigma with 96 nodes the
#: spacing is sigma/4 and the trapezoid error on a Gaussian is ~1e-137.
PHI_WINDOW_SIGMA = 12.0
PHI_NODES_PER_REGION = 96

#: Grid on which the phi omitted-mass bound is evaluated.  Not a tuning knob: it sets the
#: half-spacing ``delta`` of the second-order lift, so a coarser grid gives a LOOSER
#: (still valid) bound and more declines, never a wrong accept.
PHI_BOUND_GRID = 256

#: Accept when the certified mass outside the covered phi regions is this many nats below
#: the value.  Same number and same meaning as the numpy reference's OUTSIDE_TOL_NATS.
OUTSIDE_TOL_NATS = -23.0


def phi_derivative_bound(C, order=0):
    """TRUE bound on ``|d^order_phi g|`` by the triangle inequality on the table.

    The one construction here that cannot be a fit -- the 1-D phi analogue of the numpy
    reference's :func:`~RIFT.likelihood.joint_angle_peak_local.derivative_bound`.
    """
    KP = C.shape[0]
    k = jnp.arange(KP)[:, None]
    w = jnp.where(k > 0, 2.0, 1.0)          # k>0 stored once, counted twice (real field)
    return (w * jnp.abs(C) * (jnp.abs(k) ** order)).sum()


def profile_derivative_bounds(C):
    """Exact bounds ``(M1F, M2F)`` on ``|F'|`` and ``|F''|`` for the u-profile ``F``.

    The envelope identities are ``F' = E[d_phi g]`` and ``F'' = E[d^2_phi g] +
    Var(d_phi g)``, the expectation being under the normalized ``exp(g) du``.  So
    ``|F'| <= sup|d_phi g| <= M10`` and, since a variable confined to a range of width
    ``2 M10`` has variance at most ``M10^2``, ``|F''| <= M20 + M10^2``.  Both follow from
    the coefficient table alone -- no sample, no fit, and in particular NOT the measured
    ``F''`` at a point, which is what an estimate-promoted-to-bound would use here.
    """
    m10 = phi_derivative_bound(C, 1)
    m20 = phi_derivative_bound(C, 2)
    return m10, m20 + m10 * m10


def eval_g2(C, phi, u, order=(0, 0)):
    """``d^a_phi d^b_u g`` at matching ``(phi, u)``, from the 2-D table."""
    KP = C.shape[0]
    KS = (C.shape[1] - 1) // 2
    k = jnp.arange(KP)[None, :, None]
    q = jnp.arange(-KS, KS + 1)[None, None, :]
    w = jnp.where(jnp.arange(KP) > 0, 2.0, 1.0)[None, :, None]
    a, b = order
    phi = jnp.atleast_1d(phi)
    u = jnp.atleast_1d(u)
    E = jnp.exp(1j * (phi[:, None, None] * k + u[:, None, None] * q))
    return (E * ((1j * k) ** a) * ((1j * q) ** b) * (w * C[None])).sum((1, 2)).real


def u_profile(C, phi, n_nodes=U_NODES_PER_CELL, window_sigma=U_WINDOW_SIGMA):
    """``F(phi) = log int du exp(g)``, its first two EXACT phi-derivatives, and the
    number of u cells that fell back to whole-cell integration.

    Differentiating under the integral gives them from the SAME nodes at no extra
    evaluation cost:

        F'  = E[d_phi g]              F'' = E[d^2_phi g] + Var(d_phi g)

    the expectation being under the normalized ``exp(g) du`` on the u axis.  That
    variance term is why phi cannot inherit the u axis's economy: it grows with
    amplitude, so ``F`` sharpens as the signal does even though ``g`` does not.
    """
    KP = C.shape[0]
    KS = (C.shape[1] - 1) // 2
    k = jnp.arange(KP)
    w = jnp.where(k > 0, 2.0, 1.0)
    ph = jnp.exp(1j * phi * k) * w
    D = lambda q: (ph * C[:, KS + q]).sum()
    a = D(0).real
    c1 = D(1) + jnp.conj(D(-1))
    c2 = D(2) + jnp.conj(D(-2))

    u = jnp.sort(u_stationary_roots(c1, c2))
    mid = 0.5 * (u + jnp.roll(u, -1) + jnp.where(jnp.arange(4) == 3, 2 * jnp.pi, 0.0))
    lo_c = jnp.roll(mid, 1) - jnp.where(jnp.arange(4) == 0, 2 * jnp.pi, 0.0)

    def _newton(uc, _):
        g1 = _g_u(a, c1, c2, uc, 1)
        g2 = _g_u(a, c1, c2, uc, 2)
        step = jnp.where(jnp.abs(g2) > 0, -g1 / jnp.where(jnp.abs(g2) > 0, g2, 1.0), 0.0)
        return jnp.clip(uc + jnp.clip(step, -0.5, 0.5), lo_c, mid), None

    ustar, _ = lax.scan(_newton, u, None, length=8)
    g1s = _g_u(a, c1, c2, ustar, 1)
    g2s = _g_u(a, c1, c2, ustar, 2)
    # A CLIPPED NEWTON POINT IS NOT A PEAK, however negative the curvature -- the SAME
    # defect log_inner_u_integral already gates, reintroduced here because this function
    # was written as a fresh copy of that iteration rather than as a call to it.  The
    # iteration is clamped to [lo_c, mid], so it can come to rest ON a boundary with a
    # large stationary residual; curvature alone then centres a +-window_sigma window on a
    # non-stationary point, sizes sigma from the wrong curvature, and can EXCLUDE the true
    # maximum -- underestimating F while the docstring calls the derivatives exact.
    # Measured in the numpy twin: 18% of cells that g'' < 0 accepts fail this gate, worst
    # at |g_u|/M_1 = 0.33.  Require stationarity against the axis's own exact derivative
    # bound AND interior placement; a cell failing either is integrated WHOLE.
    m1u = jnp.abs(c1) + 2.0 * jnp.abs(c2)          # exact bound on |d g / du|
    edge = 1e-9 * jnp.max(mid - lo_c)
    peaked = ((g2s < 0.0)
              & (jnp.abs(g1s) <= 1e-8 * jnp.maximum(m1u, 1e-300))
              & (ustar > lo_c + edge) & (ustar < mid - edge))
    sig = jnp.where(peaked, 1.0 / jnp.sqrt(jnp.where(peaked, -g2s, 1.0)), jnp.inf)
    lo = jnp.where(peaked, jnp.maximum(ustar - window_sigma * sig, lo_c), lo_c)
    hi = jnp.where(peaked, jnp.minimum(ustar + window_sigma * sig, mid), mid)
    width = jnp.maximum(hi - lo, 0.0)

    s = jnp.linspace(0.0, 1.0, n_nodes)
    uu = (lo[:, None] + width[:, None] * s[None, :]).ravel()          # (4n,)
    pp = jnp.full(uu.shape, phi)
    gg = eval_g2(C, pp, uu, (0, 0))
    gp = eval_g2(C, pp, uu, (1, 0))
    gpp = eval_g2(C, pp, uu, (2, 0))
    wq = jnp.full(n_nodes, 1.0 / (n_nodes - 1)).at[0].mul(0.5).at[-1].mul(0.5)
    lw = (jnp.log(jnp.where(width > 0, width, 1e-300))[:, None]
          + jnp.log(wq)[None, :]).ravel()
    lw = jnp.where(jnp.repeat(width > 0, n_nodes), lw, -jnp.inf)

    m = gg.max()
    wt = jnp.exp(gg - m + lw)
    Z = wt.sum()
    e1 = (wt * gp).sum() / Z
    F = m + jnp.log(Z)
    ddF = (wt * (gpp + gp * gp)).sum() / Z - e1 * e1
    # how many of the four cells were integrated WHOLE rather than windowed.  Reported
    # because a fallback cell spreads the same static node count over a wider interval, so
    # it is the one place F itself can be inaccurate -- and no bound on this axis can see
    # that, since the omitted-mass certificate covers what is outside the regions.
    n_fallback = (~peaked).sum()
    return F, e1, ddF, n_fallback


def _merge_sorted_intervals(lo, hi, n):
    """Merge overlapping 1-D intervals under jit, without data-dependent shapes.

    Sorting by ``lo`` makes merging a running maximum: a new group starts exactly where
    an interval begins beyond the running max of the ``hi`` seen so far.  Group ids are
    then a cumsum, and the merged bounds are segment reductions over a FIXED number of
    slots.  Empty slots come back as an inverted interval and are dropped by the
    ``width > 0`` mask downstream, so nothing needs compaction.

    This is the jittable form of the reference's ``_merge_boxes``; merging is not
    tidiness but what stops the mass between two windows being counted twice.
    """
    idx = jnp.argsort(lo)
    lo, hi = lo[idx], hi[idx]
    run = jax.lax.cummax(hi)
    fresh = jnp.concatenate([jnp.array([True]), lo[1:] > run[:-1]])
    gid = jnp.cumsum(fresh) - 1
    seg_lo = jax.ops.segment_min(lo, gid, num_segments=n, indices_are_sorted=True)
    seg_hi = jax.ops.segment_max(hi, gid, num_segments=n, indices_are_sorted=True)
    return seg_lo, seg_hi


def phi_local_lnI(C, n_seed=PHI_SEEDS, w_sigma=PHI_WINDOW_SIGMA,
                  n_nodes=PHI_NODES_PER_REGION, u_nodes=U_NODES_PER_CELL,
                  n_bound=PHI_BOUND_GRID, tol_nats=OUTSIDE_TOL_NATS):
    """``log int dphi int du exp(g)`` with BOTH axes localized, jittable.

    Returns ``(value, ok, info)``.  ``ok`` is False when the omitted-mass bound on the phi
    axis could not be made small enough; the value is returned either way for diagnosis,
    but a value with ``ok=False`` is NOT to be used.

    u is exact on the cell partition; phi is localized around the maxima of the profile
    ``F`` using its exact derivatives (see :func:`u_profile`).  phi has no algebraic
    completeness warrant -- ``F`` is a log-integral, not a trig polynomial -- so the seeds
    are targeting only and correctness rests on the certificate below.

    READ THIS BEFORE PROMOTING THIS PATH -- AND THE COST ARGUMENT BELOW IS WITHDRAWN.

    THIS FUNCTION LOCALIZES ON THE WRONG OBJECT.  It Newton-iterates on the maxima of
    ``F(phi) = log int du exp(g)`` from ``PHI_SEEDS`` arbitrary seeds.  ``F`` is a
    log-integral and has no completeness warrant -- but ``g`` ITSELF DOES, and it is the
    same warrant psi has.  The orbital phase enters the modes as ``e^{-i m phi}``, so ``A``
    carries phi-harmonics to ``m_max`` and ``B``, being quadratic, to ``2 m_max``.  The
    combined table's ``k_max = KP-1 = 2 m_max`` is therefore EXACT, and ``dg/dphi = 0``
    under ``z = e^{i phi}`` is a polynomial of degree ``2 k_max``.  Knowing the mode
    content fixes the stationary count; the 2-D system with ``dg/du = 0`` has a
    mixed-volume bound of ``16 k_max``.  The numpy reference already does this --
    :func:`~RIFT.likelihood.joint_angle_peak_local.enumerate_modes` solves the algebraic
    system -- and it finds MORE maxima than the seeded search: 13 against 8 at ``KP=5``,
    30 against 20 at ``KP=13``.

    So the earlier conclusion here -- that certifying phi costs more than the dense grid
    because ``n_bound ~ A`` -- was reasoning about a construction chosen in this file, not
    about the phi axis.  Seeded algebraically the region count is mode-order-bounded and
    provable, and the cost comparison has to be redone on that basis.  It is NOT restated
    here in a corrected form, because two successive versions of it were wrong; the
    measurements are on the PR and the argument needs rebuilding, not patching.

    MEASURED LIMITS OF THE SHIPPED CONSTANTS (Blackwell, jax 0.9.2, x64):
      * ``PHI_SEEDS = 32`` is an undocumented assumption about mode content.  At
        ``m_max = 2`` the region count plateaus by 32 seeds (7-8 regions, unchanged at
        64 and 128).  At ``m_max = 6`` it does NOT: 32 seeds find 14-19 regions where 64+
        find 19-21.  FAIL-CLOSED -- every such case declines, none returns an accepted
        wrong value -- and the missed regions are subdominant, changing the value by less
        than 1e-5.  At ``m_max = 6`` the rule declines universally, so high mode content
        is outside its reach for reasons beyond the seed count.
      * 94-97% of the phi work is on EMPTY slots: ``2 * PHI_SEEDS = 64`` static slots are
        allocated and 96 nodes evaluated in every one, while production tables use 2-4.
        That is the price of static shapes without an enumeration; it is not recoverable
        by shrinking the allocation, because shrinking starves the seeds as well and
        converts silent waste into declines (measured: 2 regions accept at 8 seeds and
        decline at 4).
      * Per-evaluation device memory is 0.098 GiB against the dense path's 0.001 GiB, and
        it scales LINEARLY with the vmap product because nothing here chunks.
        :func:`joint_lnL_phi_dense` bounds its own memory with ``lax.scan`` over
        ``phi_chunk`` and is flat in ``n_phi`` (0.39 GiB at 256, 1024 and 4096 alike).
        The (2,+-2) tables carry an EXACT ORDER-4 SYMMETRY, which reduces the bound grid to a
    QUARTER domain -- worth 4x against a shortfall of 80x, real but not the answer.
    Measured on the exponent itself, which is the object this code evaluates, and not on
    the coefficient table it is built from:

        S : (phi, u) -> (phi + pi/2, u + pi)      generator, order 4

        rung 1   S^1..S^4 deviations   2.2e-15  2.6e-15  4.0e-15  3.8e-15
        rung 3                         2.8e-15  2.8e-15  4.6e-15  4.3e-15

    ``S^2 = (phi + pi, u)`` is therefore also exact, which is where the phi half-period
    comes from; ``(phi, u + pi)`` and ``(phi + pi, u + pi)`` are NOT symmetries (relative
    deviation 1.32 each), so there is no u half-period on its own.  Every maximum carries
    exactly FOUR copies and the enumeration confirms it: rung 3's four maxima are ONE orbit
    of four (one distinct maximum, which is why they are exactly degenerate), and rung 1's
    eight are TWO orbits of four, matching its two distinct exponent values.

    TWO EARLIER VERSIONS OF THIS NOTE WERE WRONG HERE, in opposite directions, and both
    times the CONCLUSION that ``F`` is pi-periodic survived: first ``(phi+pi, u+pi)`` with
    multiplicity four, taken from a coefficient parity measured in another convention; then
    ``(phi+pi, u)`` with multiplicity two, from testing only the shifts I had thought to
    list.  The generator was never among them.  Enumerate the group from the maxima's own
    offsets rather than guessing which shifts to test.
        """
    prof = lambda p: u_profile(C, p, n_nodes=u_nodes)
    seeds = jnp.linspace(0.0, 2.0 * jnp.pi, n_seed, endpoint=False)

    def _newton(p, _):
        _, d1, d2, _ = jax.vmap(prof)(p)
        step = jnp.where(d2 < 0, -d1 / jnp.where(d2 < 0, d2, -1.0), 0.0)
        return jnp.mod(p + jnp.clip(step, -0.3, 0.3), 2.0 * jnp.pi), None

    p, _ = lax.scan(_newton, seeds, None, length=24)
    F, d1, d2, n_fb = jax.vmap(prof)(p)
    peaked = d2 < 0.0
    sig = jnp.where(peaked, 1.0 / jnp.sqrt(jnp.where(peaked, -d2, 1.0)), 0.0)

    # non-maxima are pushed past every real interval so they form empty groups; no
    # tolerance decides membership, which is deliberate -- a threshold on |F'| would be
    # exactly the estimate-promoted-to-bound this design refuses.
    big = 1.0e6
    lo = jnp.where(peaked, p - w_sigma * sig, big)
    hi = jnp.where(peaked, p + w_sigma * sig, big)

    # SPLIT AT THE SEAM BEFORE MERGING, for the reason the numpy reference had to: a
    # linear merge never joins a window near 0 to one near 2 pi, yet every region is
    # integrated at mod(., 2 pi), so both cover both peaks and the mass is counted twice
    # (+log 2, accepted, because the error is inside the regions).  Each interval yields
    # AT MOST two pieces, so 2*n_seed slots is a static bound and nothing has to be
    # compacted; a piece that does not exist is emitted empty and drops out downstream.
    wdt = jnp.clip(hi - lo, 0.0, 2.0 * jnp.pi)
    a0 = jnp.where(peaked, jnp.mod(lo, 2.0 * jnp.pi), big)
    crosses = peaked & (a0 + wdt > 2.0 * jnp.pi)
    lo2 = jnp.concatenate([a0,
                           jnp.where(crosses, 0.0, big)])
    hi2 = jnp.concatenate([jnp.where(crosses, 2.0 * jnp.pi, a0 + wdt),
                           jnp.where(crosses, a0 + wdt - 2.0 * jnp.pi, big)])
    seg_lo, seg_hi = _merge_sorted_intervals(lo2, hi2, 2 * n_seed)
    n_seed = 2 * n_seed
    # There are always more slots than groups, and an EMPTY slot comes back from the
    # segment reductions as (+inf, -inf).  Masking its weight is not enough: the node
    # positions are still built from it, jnp.mod(inf, 2 pi) is NaN, and NaN * 0 is NaN,
    # so the poison reaches the sum through a term that was supposed to be switched off.
    # Neutralize the POSITION, not just the weight.
    seg_lo = jnp.where(jnp.isfinite(seg_lo), seg_lo, 0.0)
    seg_hi = jnp.where(jnp.isfinite(seg_hi), seg_hi, 0.0)
    width = jnp.clip(seg_hi - seg_lo, 0.0, 2.0 * jnp.pi)

    # CLAMP TO ONE CIRCUIT.  At low amplitude sigma is huge and the windows span more
    # than 2 pi; integrating that literally wraps the circle and counts the same mass
    # repeatedly (measured +1.84 nats, a factor of 6.3, on real tables in the numpy
    # reference -- and ACCEPTED, because a region covering everything leaves nothing
    # outside for the certificate to object to).
    # close the circle: if some piece ends at 2 pi and another starts at 0 they are one
    # region.  Left unjoined they are still DISJOINT, so nothing is double-counted -- the
    # only cost is one extra region and a seam the quadrature treats as an edge.
    total = width.sum()
    wrapped = total >= 2.0 * jnp.pi
    seg_lo = jnp.where(wrapped, jnp.where(jnp.arange(n_seed) == 0, 0.0, big), seg_lo)
    width = jnp.where(wrapped,
                      jnp.where(jnp.arange(n_seed) == 0, 2.0 * jnp.pi, 0.0), width)

    s = jnp.linspace(0.0, 1.0, n_nodes)
    pp = (seg_lo[:, None] + width[:, None] * s[None, :]).ravel()
    Fv, _, _, _ = jax.vmap(prof)(jnp.mod(pp, 2.0 * jnp.pi))
    wq = jnp.full(n_nodes, 1.0 / (n_nodes - 1)).at[0].mul(0.5).at[-1].mul(0.5)
    lw = (jnp.log(jnp.where(width > 0, width, 1e-300))[:, None]
          + jnp.log(wq)[None, :]).ravel()
    lw = jnp.where(jnp.repeat(width > 0, n_nodes), lw, -jnp.inf)
    value = jax.scipy.special.logsumexp(Fv + lw)

    # ---------------------------------------------------------------- the phi certificate
    # WITHOUT THIS THE RETURN VALUE IS AN ESTIMATE WEARING A LIKELIHOOD'S CLOTHES.  The
    # seeds are targeting, not an enumeration -- phi has no algebraic completeness warrant
    # because F is a log-integral, not a trig polynomial -- so a missed maximum or an
    # unconverged seed is silently omitted and a finite number comes back regardless.
    # External review found this exposed with no bound, no validity result and no fallback
    # signal, and it is the house rule of this whole family violated in its own code.
    #
    # The bound: mass outside the covered regions is at most
    #     area_outside * exp(sup_outside F),
    # and sup_outside F is obtained from a grid of F values LIFTED by a true remainder,
    # never from the grid maximum itself -- a grid max is a LOWER bound on a supremum and
    # the gap grows with amplitude.  Both F and F' come back from u_profile at no extra
    # cost, so the lift is second order:
    #     F(x) <= F(x_i) + |F'(x_i)| * delta + M2F * delta^2 / 2,   delta = half spacing
    # with M2F from profile_derivative_bounds, i.e. from the coefficient table alone.
    # A first-order Lipschitz lift was tried first in the numpy twin and is USELESS at
    # amplitude -- it put the bound above the integral by +1225 nats.
    gb = jnp.linspace(0.0, 2.0 * jnp.pi, n_bound, endpoint=False)
    delta = jnp.pi / n_bound                      # half of the grid spacing
    Fb, d1b, _, _ = jax.vmap(prof)(gb)
    m1f, m2f = profile_derivative_bounds(C)
    ub = Fb + jnp.abs(d1b) * delta + 0.5 * m2f * delta * delta

    # A GRID POINT COUNTS AS OUTSIDE UNLESS ITS WHOLE delta-BALL IS COVERED.  Testing the
    # point alone leaves a band of width delta beside every region boundary belonging to
    # no test at all, and the bound would then be a bound on the wrong set.  Regions are
    # therefore ERODED by delta before the test, which over-estimates the outside -- the
    # safe direction.  A region already spanning the circle stays covering: that is the
    # low-amplitude case where the rule has degenerated into the dense grid on purpose,
    # and eroding it would report an uncovered band and decline every such row.
    full = width >= 2.0 * jnp.pi - 1e-12
    eff_lo = jnp.where(full, -1.0, seg_lo + delta)
    eff_hi = jnp.where(full, 2.0 * jnp.pi + 1.0, seg_lo + width - delta)
    d = gb[None, :] - eff_lo[:, None]
    covered = (((d >= 0.0) & (gb[None, :] <= eff_hi[:, None]))
               | ((d + 2.0 * jnp.pi >= 0.0)
                  & (gb[None, :] + 2.0 * jnp.pi <= eff_hi[:, None]))).any(axis=0)

    area_outside = jnp.clip(2.0 * jnp.pi - width.sum(), 0.0, 2.0 * jnp.pi)
    sup_outside = jnp.max(jnp.where(covered, -jnp.inf, ub))
    outside = jnp.where(area_outside > 0.0,
                        jnp.log(jnp.where(area_outside > 0.0, area_outside, 1.0))
                        + sup_outside,
                        -jnp.inf)
    margin = outside - value
    ok = margin < tol_nats

    info = {"margin": margin,
            "area_outside": area_outside,
            "sup_outside": sup_outside,
            "n_phi_regions": (width > 0).sum(),
            # INTERNAL accuracy, which the certificate above CANNOT see: it bounds the
            # mass left OUTSIDE the regions and says nothing about the quadrature inside
            # one.  Reported separately and never folded into `margin`.
            "n_u_fallback": n_fb.sum()}
    return value, ok, info
