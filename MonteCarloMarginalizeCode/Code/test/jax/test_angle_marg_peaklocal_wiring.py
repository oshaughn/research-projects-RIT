"""The 'peak-local' angle-marg scheme, as wired into the ILE likelihood.

These test the WIRING -- that the option reaches the likelihood, produces the same
answer as the scheme it is meant to replace, records its provenance, refuses what it
cannot honour, and does NOT change any default.  The kernel's own numerics are tested in
test_joint_anglemarg_peaklocal.py.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood
from test_angle_marg_exact import make_synth, RA, DEC, INCL, INTERP


def test_peak_local_is_an_offered_choice_and_reaches_the_cli():
    """optparse builds --angle-marg-scheme's choices from ANGLE_MARG_CHOICES, so being
    in that tuple IS the CLI wiring; a scheme absent from it is unreachable."""
    assert "peak-local" in AM.ANGLE_MARG_CHOICES
    assert hasattr(AM, "fused_log_likelihood_distphipsimarg_peaklocal")


def test_peak_local_is_NOT_reachable_from_auto():
    """A scheme that changes the likelihood must not become reachable by default on the
    strength of unit tests.  'auto' must keep choosing among the schemes that have
    campaign evidence until a head-to-head pilot says otherwise."""
    for amp in (1.0, 50.0, 500.0, 5.0e4, 5.0e6):
        scheme, _ = AM.choose_angle_marg_scheme(amp)
        assert scheme != "peak-local", (amp, scheme)


@pytest.mark.parametrize("boost", [1.0, 30.0])
def test_wrapper_peak_local_matches_exact(boost):
    """The wiring's whole claim: asking for it by name gives the same likelihood as the
    scheme it parallels."""
    data = make_synth(scale=2.0, kappa_boost=boost)
    kw = dict(nphi=32, npsi=8, interp=INTERP)
    ex = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, angle_marg="exact", **kw)
    pl = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, angle_marg="peak-local", **kw)
    assert pl.angle_marg_scheme == "peak-local"
    a = np.asarray(ex._batched(jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL)))
    b = np.asarray(pl._batched(jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL)))
    assert np.abs(a - b).max() < 1e-4, (boost, a, b)


def test_peak_local_records_its_provenance():
    """This pipeline has a documented history of silently-inert flags, so the scheme
    actually used must be visible in the run record, not inferred from the request."""
    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                       interp=INTERP, angle_marg="peak-local")
    info = like.angle_marg_info
    assert info["scheme"] == "peak-local"
    assert info["requested"] == "peak-local"
    assert "amp_sizing" in info, info


def test_peak_local_refuses_the_adaptive_distance_quadrature():
    """Refuse rather than silently ignore: this kernel sums the caller's distance grid
    and implements no psi-marginal node placement, exactly as the laplace branch does
    not.  The incompatibility is a property of the scheme, declared once."""
    from RIFT.likelihood.jax_ile import core as _core
    data = make_synth(scale=2.0)
    old = _core._DISTMARG_GH_N
    try:
        _core._DISTMARG_GH_N = 8
        with pytest.raises(ValueError, match="peak-local"):
            AM.fused_log_likelihood_distphipsimarg_peaklocal(
                data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
                jnp.linspace(0.4, 2.0, 8), jnp.zeros(8), interp=INTERP,
                amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)
    finally:
        _core._DISTMARG_GH_N = old


def test_peak_local_requires_amp_sizing_rather_than_guessing_it():
    """There is deliberately no default: the phi axis is still dense here and must be
    SIZED from the data, and a silently-undersized grid is the defect this module family
    exists to remove."""
    data = make_synth(scale=2.0)
    with pytest.raises(Exception):
        AM.fused_log_likelihood_distphipsimarg_peaklocal(
            data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
            jnp.linspace(0.4, 2.0, 8), jnp.zeros(8), interp=INTERP)


def test_the_driver_CLI_accepts_it_and_rejects_a_typo():
    """Deliberately a SUBPROCESS: the flag's job is to travel from a command line into
    the likelihood, and optparse builds its choices from ANGLE_MARG_CHOICES at import
    time -- a test that imports the module cannot see the CLI wiring.  A misspelling
    must be loud, because a scheme name absorbed as 'not recognised' would silently run
    a different likelihood."""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    driver = os.path.join(root, "bin", "integrate_likelihood_extrinsic_jax")
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_PLATFORMS"] = "cpu"

    def run(value):
        p = subprocess.run([sys.executable, driver, "--angle-marg-scheme", value],
                           env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=600)
        return p.stdout.decode("utf-8", "replace")

    good = run("peak-local")
    assert "invalid choice" not in good, good[-1500:]

    bad = run("peaklocal")            # the plausible misspelling
    assert "invalid choice" in bad, bad[-1500:]
    assert "peak-local" in bad, bad[-1500:]


# --------------------------------------------------------- review findings

def test_peak_local_runs_the_runtime_amplitude_failsafe():
    """P1 from review.  The u axis is localized and needs no sizing, but THE PHI AXIS IS
    STILL DENSE and is sized from amp_sizing, which estimate_angle_amplitude is explicit
    about being an estimator and NOT a proven bound.  A hotter sampled sky location can
    therefore under-resolve phi exactly as it can for exact/laplace, and skipping the
    check would publish that silently."""
    data = make_synth(scale=2.0, kappa_boost=50.0)
    x = jnp.linspace(0.4, 2.0, 8)
    lw = jnp.zeros(8)
    AM.reset_amp_failsafe()
    # size for a much quieter target than the data actually is: the check must notice
    AM.fused_log_likelihood_distphipsimarg_peaklocal(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL), x, lw,
        interp=INTERP, amp_sizing=1.0)
    st = AM.amp_failsafe_state(barrier=True)
    assert st.get("tripped"), st
    assert st.get("scheme") == "peak-local", st
    AM.reset_amp_failsafe()


def test_peak_local_is_capped_by_the_batch_memory_rule():
    """P1 from review.  peak-local still nests sample/time vmaps over the distance grid,
    phi chunks, four cells and the streamed u-node block, and its scan returns every
    ``(phi,distance)`` value, so the batch multiplies the same way the dense schemes do.
    Leaving it out of the cap kept an uncapped 8000-sample batch."""
    from RIFT.likelihood.jax_ile import samplers as S

    class _Data(object):
        npts = 614

    class _Like(object):
        data = _Data()
        angle_marg_scheme = "peak-local"
        x_grid = np.zeros(256)

    class _Exact(_Like):
        angle_marg_scheme = "exact"

    class _NoScheme(object):
        pass

    capped = S.angle_marg_eval_chunk(_Like(), 8000)
    assert capped < 8000
    # NOT "same cap as exact" -- that was the earlier assertion and review rightly
    # objected that it pins the wrong invariant.  peak-local carries the WHOLE distance
    # grid inside every phi chunk and stacks the full phi-scan result; its production-
    # floor model is ~216x the dense model's 8192 bytes/sample/time-point.  A cap equal
    # to exact's would look protective and would not be.  The scheme-specific model must
    # therefore be STRICTLY tighter.
    assert capped < S.angle_marg_eval_chunk(_Exact(), 8000), capped
    # and it must scale with the distance grid, which is what makes it a model rather
    # than a constant
    class _Wide(_Like):
        x_grid = np.zeros(1024)
    # At this width the corrected body+scan model exceeds the fallback target
    # even at S=1.  Returning a cap of one would claim protection it cannot give.
    with pytest.raises(MemoryError, match="resource preflight"):
        S.angle_marg_eval_chunk(_Wide(), 8000)
    # the "grid" sentinel means "runs no dense angle scheme" and must stay uncapped
    assert S.angle_marg_eval_chunk(_NoScheme(), 8000) == 8000


def test_known_four_gib_device_uses_configured_fraction(monkeypatch):
    """The unknown-device 4-GiB reserve must never become a known-device floor."""
    from RIFT.likelihood.jax_ile import samplers as S

    class _Device(object):
        platform = "gpu"

        def memory_stats(self):
            return {"bytes_limit": 4 << 30,
                    "largest_free_block_bytes": 4 << 30}

    monkeypatch.setattr(S.jax, "devices", lambda: [_Device()])
    monkeypatch.setattr(S, "_ANGLE_MARG_BUFFER_FRACTION", 0.5)
    assert S._angle_marg_buffer_target() == (2 << 30)


@pytest.mark.parametrize("amplitude,n_phi", [(450.0, 352), (12500.0, 1792)])
def test_peak_local_model_includes_streamed_body_and_scan_output(
        amplitude, n_phi):
    """The cap must account for both source-visible peak-local payloads."""
    from RIFT.likelihood.jax_ile import samplers as S

    class _Data(object):
        npts = 1193
        lms = ((2, 2), (2, -2))

    class _Like(object):
        data = _Data()
        angle_marg_scheme = "peak-local"
        x_grid = np.zeros(256)
        angle_marg_info = {"amp_sizing": amplitude}

    per_point = S._peaklocal_bytes_per_sample_pt(_Like())
    assert per_point == 16 * 256 * 4 * 8 * 8 + n_phi * 256 * 8


def test_peak_local_resource_preflight_refuses_an_unfit_single_sample(
        monkeypatch):
    """A=12500 needs 5.242 GiB/sample; a cap of one would still OOM 4 GiB."""
    from RIFT.likelihood.jax_ile import samplers as S

    class _Data(object):
        npts = 1193
        lms = ((2, 2), (2, -2))

    class _Like(object):
        data = _Data()
        angle_marg_scheme = "peak-local"
        x_grid = np.zeros(256)
        angle_marg_info = {"amp_sizing": 12500.0}

    monkeypatch.setattr(S, "_angle_marg_buffer_target", lambda: 4 << 30)
    with pytest.raises(MemoryError, match="reducing the outer evaluation chunk"):
        S.angle_marg_eval_chunk(_Like(), 8000)


def test_peak_local_floor_amplitude_fits_one_sample_at_two_gib(monkeypatch):
    """A=450 needs 1.966 GiB/sample, so the known-4-GiB target admits only one."""
    from RIFT.likelihood.jax_ile import samplers as S

    class _Data(object):
        npts = 1193
        lms = ((2, 2), (2, -2))

    class _Like(object):
        data = _Data()
        angle_marg_scheme = "peak-local"
        x_grid = np.zeros(256)
        angle_marg_info = {"amp_sizing": 450.0}

    monkeypatch.setattr(S, "_angle_marg_buffer_target", lambda: 2 << 30)
    assert S.angle_marg_eval_chunk(_Like(), 8000) == 1


def test_peak_local_artifacts_carry_the_standing_best_effort_label():
    """P1 from review.  A scheme missing from the label's list publishes output with NO
    standing statement at all -- and silence is precisely what a reader six months later
    would misread as verification.  peak-local's phi grid is amp-sized, so its artifacts
    are entitled to no more confidence than exact's."""
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "bin", "integrate_likelihood_extrinsic_jax")
    spec = importlib.util.spec_from_loader("_ile_jax_driver",
                                           importlib.machinery.SourceFileLoader(
                                               "_ile_jax_driver", path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AM.reset_amp_failsafe()
    note = mod.angle_grid_suspect_note("peak-local")
    assert note.startswith("ANGLE-GRID-CHECK=BEST-EFFORT"), note
    assert mod.angle_grid_suspect_note("grid") == ""


def test_kernel_and_memory_guard_read_the_same_node_count():
    """Review P2.  ``u_nodes_in_use`` was introduced as the single source of truth for the
    u-node count, and its docstring said both the kernel and the batch-memory guard call
    it -- but only the guard did.  ``joint_lnL_phi_dense`` still defaulted straight to
    ``U_NODES_PER_CELL`` and the fused caller passed no ``n_nodes``, so an
    amplitude-dependent change would have moved the guard and left the kernel behind.  A
    single source of truth that only one side reads is not one.

    The invariant is NOT "both currently equal 48" -- production uses the uncapped
    derived count.  Both sides must read the same amplitude, while the guard models only
    the streamed live block rather than the total quadrature work.

    The shape is deliberately NOT the production one.  At npts=614 with 256 distance nodes
    the cap is already pinned at its floor of 1 -- the measured "peak-local batches one
    sample" result -- so quadrupling the node count cannot move it, and the guard assertion
    would read ``1 < 1`` and fail while the wiring was correct.  A saturated observable
    cannot test the thing it saturates on.  npts=64 with 32 distance nodes stays clear of
    both the floor and the 8000 ceiling.
    """
    from RIFT.likelihood.jax_ile import samplers as S
    from RIFT.likelihood.jax_ile import joint_anglemarg_peaklocal as JP

    class _Data(object):
        npts = 64

    class _Like(object):
        data = _Data()
        angle_marg_scheme = "peak-local"
        x_grid = np.zeros(32)
        angle_marg_info = {"amp_sizing": 450.0}

    seen = []
    real_helper = JP.u_nodes_in_use
    real_inner = JP.log_inner_u_integral

    def _spy_inner(a, c1, c2, n_nodes=JP.U_NODES_PER_CELL, **kw):
        seen.append(int(n_nodes))
        return real_inner(a, c1, c2, n_nodes, **kw)

    baseline_cap = S.angle_marg_eval_chunk(_Like(), 8000)
    assert 1 < baseline_cap < 8000, baseline_cap      # the observable is not saturated

    helper_args = []
    def _raised_policy(amp_sizing=None):
        helper_args.append(amp_sizing)
        return 4 * real_helper(amp_sizing)

    JP.u_nodes_in_use = _raised_policy
    JP.log_inner_u_integral = _spy_inner
    try:
        # The guard must consult the helper with the production amplitude.  Its cap does
        # not shrink because the extra total work is streamed through the same live block.
        raised_cap = S.angle_marg_eval_chunk(_Like(), 8000)
        assert raised_cap == baseline_cap, (baseline_cap, raised_cap)
        assert 450.0 in helper_args, helper_args

        # the KERNEL must follow it too, via n_nodes=None resolving through the helper
        rng = np.random.default_rng(0)
        C_A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        C_B = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        C_B[0, 2] = abs(C_B[0, 2].real) + 3.0
        x_grid = jnp.asarray(np.linspace(0.5, 2.0, 8))
        lw = jnp.zeros(8)
        JP.joint_lnL_phi_dense(jnp.asarray(C_A), jnp.asarray(C_B), x_grid, lw, n_phi=8)
        assert seen, "kernel never reached log_inner_u_integral"
        assert set(seen) == {4 * real_helper(None)}, (seen, real_helper(None))
    finally:
        JP.u_nodes_in_use = real_helper
        JP.log_inner_u_integral = real_inner

    # restoring the helper restores the cap exactly -- no hidden state
    assert S.angle_marg_eval_chunk(_Like(), 8000) == baseline_cap
