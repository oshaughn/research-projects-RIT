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
``_merge_boxes`` on cannot arise.  Everything is then static: 4 roots, 4 candidate
cells, a fixed number of quadrature nodes in each.

WHY THE ROOTS ARE TAKEN WITHOUT A ``|z| = 1`` FILTER.  At exact multiplicity the
computed roots smear off the unit circle by ``eps^(1/m)`` -- measured 4.6e-6 for a
triple root -- so a fixed tolerance drops real modes in precisely the degenerate regime
that is normal here.  Every root contributes its angle; a spurious one produces a
zero-length or redundant cell, which is harmless, whereas a dropped one loses mass.

WHAT SCALES WITH AMPLITUDE AND WHAT DOES NOT.  The stationary points of ``g`` do not
move when the data amplitude grows -- ``g -> lambda g`` leaves them fixed -- so the
CELLS are amplitude-independent, while the peak inside each cell narrows as
``A^-1/2``.  The local window is therefore sized from the local curvature and clipped
to the cell, which keeps the node count fixed.  This is the u axis's whole economy: the
shipped dense scheme spends ``~sqrt(A)`` points on this axis, and this spends a
constant.

THAT ECONOMY IS CLAIMED ONLY FOR WINDOWED CELLS.  A cell whose Newton centre is rejected
is integrated WHOLE (see :func:`log_inner_u_integral`), and a whole cell is not narrow --
it inherits the dense ``~sqrt(A)`` requirement.  Which cells fall back is data-dependent
and the node count is static, so the production caller sizes for the fallback case with
:func:`required_u_nodes` and declines when :func:`u_nodes_capped` says the sizing cannot
be met.  The honest cost statement is therefore: constant on this axis wherever every
cell is windowed, and ``~sqrt(A)`` where the caller must insure against a fallback.

SCOPE OF THIS KERNEL.  The u axis is localized; the phi axis is a dense grid, scanned
in chunks.  That is deliberately the same cost shape as the shipped ``laplace`` scheme
(``~sqrt(A)`` on phi) and a strict improvement on its u treatment, which uses a blended
O(1/A) width model rather than the exact stationary points.  Localizing phi as well --
the (phi localized, psi localized) cell of the family -- needs the profile ``F(phi)``
and its envelope derivative, and is not attempted here.

MEMORY.  Bounded by ``phi_chunk`` through ``lax.scan``, never by the grid: the largest
transient is ``(phi_chunk, n_x, 4, n_u)``.  It is a cost knob and cannot change the
result beyond floating-point reassociation.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "required_n_phi",
    "required_u_nodes",
    "u_nodes_capped",
    "U_WINDOW_SIGMA",
    "U_NODES_PER_CELL",
    "U_NODES_CAP",
    "PHI_CHUNK_DEFAULT",
    "u_stationary_roots",
    "log_inner_u_integral",
    "joint_lnL_phi_dense",
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
#: +-12 sigma.  The numpy twin measured 1.7e-03 nats of inner-u error that way, so this
#: default resolves WINDOWED cells at any amplitude and nothing more.  WHICH cells fall
#: back is data-dependent and cannot be known at trace time, so a caller that may hit
#: one -- every production caller -- must size the count for the fallback case with
#: :func:`required_u_nodes` rather than take this default; the production entry point
#: :func:`~RIFT.likelihood.jax_ile.anglemarg.fused_log_likelihood_distphipsimarg_peaklocal`
#: does exactly that, and declines when the sizing cannot be met.
U_NODES_PER_CELL = 48

#: Cost ceiling on the derived fallback node count.  This is a REFUSAL threshold and not
#: a clamp to fall back on: see :func:`u_nodes_capped`.
U_NODES_CAP = 2048

#: phi points per scan step.
PHI_CHUNK_DEFAULT = 16


def _u_nodes_needed(amplitude, pts_per_sigma=3.0):
    """The derived requirement, BEFORE any cap and before the windowed floor."""
    a = max(float(amplitude), 1.0)
    return int(np.ceil(2.0 * np.pi * np.sqrt(5.0 * a) * float(pts_per_sigma))) + 1


def required_u_nodes(amplitude, pts_per_sigma=3.0, cap=U_NODES_CAP):
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

    ``cap`` bounds the cost, and the value returned when it binds is NOT adequate -- test
    :func:`u_nodes_capped` and decline, do not integrate with it.  The certificate cannot
    absorb the difference: the omitted-mass bound covers the mass OUTSIDE the cover and
    the inner-u error lives INSIDE it, so a ``-23`` nat margin says nothing whatever about
    a quadrature error of 2.2e-04 nats (log-relative -8.4, six orders of magnitude larger
    than exp(-23) of the mass).  An under-resolved cell is a declined row, not a caveat.
    """
    return int(min(max(_u_nodes_needed(amplitude, pts_per_sigma),
                       U_NODES_PER_CELL), int(cap)))


def u_nodes_capped(amplitude, pts_per_sigma=3.0, cap=U_NODES_CAP):
    """True when ``cap`` binds, i.e. :func:`required_u_nodes` returns LESS than derived.

    The one question a caller has to ask before using the returned count: below the cap
    the fallback cells are resolved by construction, at the cap they are under-resolved
    by an amount nothing downstream can measure.
    """
    return bool(_u_nodes_needed(amplitude, pts_per_sigma) > int(cap))


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
    # exposed to the caller as required_u_nodes(), and the production caller passes it
    # for EVERY cell: it cannot know at trace time which cells will fall back, so it
    # insures all of them and declines when the derived count exceeds U_NODES_CAP.  The
    # default here resolves the windowed case only and is not a production setting.
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

    s = jnp.linspace(0.0, 1.0, n_nodes)                          # (n,)
    uu = lo[:, None] + width[:, None] * s[None, :]               # (4, n)
    gg = _g_u(a, c1, c2, uu, 0)
    wq = jnp.full(n_nodes, 1.0 / (n_nodes - 1))
    wq = wq.at[0].mul(0.5).at[-1].mul(0.5)
    logw = jnp.log(wq)[None, :] + jnp.log(jnp.where(width > 0, width, 1.0))[:, None]
    cell = jax.scipy.special.logsumexp(gg + logw, axis=-1)       # (4,)
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
                        n_nodes=U_NODES_PER_CELL):
    """Distance-, phi- and psi-marginalized value at one ``(sample, time)``.

    Same normalization as ``anglemarg.fused_log_likelihood_distphipsimarg_*``: uniform
    priors ``dphi/2pi`` and ``dpsi/pi``, which in ``u = 2 psi`` is ``(2 pi)^-2`` times
    the torus integral.

    ``phi`` is a dense grid scanned in chunks; ``u`` is exact per the cell partition.
    """
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
