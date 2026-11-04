#!/usr/bin/env python
"""
test_NFlow_portfolio_contract.py

Pins the two things mcsamplerNFlow has to provide before mcsamplerPortfolio can
carry it as a member.  Both were missing, and both are reachable from production
CLI (``--sampler-portfolio NFlow`` on util_ConstructIntrinsicPosterior_GenericCoordinates
and util_ConstructEOSPosterior).

1. RETURN ORDER.  ``draw_simplified`` used to end with ``return rv, p_s, p_prior``,
   while MCSamplerGeneric and every other implementation (mcsamplerGPU,
   mcsamplerAdaptiveVolume, mcsamplerEnsemble, the unreliable_oracle members)
   return ``(p_s, p_prior, rv)``.  mcsamplerPortfolio.draw() unpacks
   ``joint_p_s_here, joint_p_prior_here, rv_here = member.draw_simplified(...)``,
   so an NFlow member had the SAMPLES assigned to joint_p_s.  Measured on
   junior/rift_O4d @94f352ad8 with a real nflows install, [AV, NFlow] portfolio,
   unit Gaussian in a [-5,5]^d box:
     * d=2: ValueError "could not broadcast input array from shape (2,100) into
       shape (100,)" out of mcsamplerPortfolio.draw.
     * d=1: NO error.  The (1,n) sample array broadcasts into the (n,) density
       slot, and the run COMPLETES with ln Z = -4.458 against a true -1.384 --
       3.07 nats low, silently.  (Control: the same portfolio with the NFlow
       member replaced by a second AV returns -1.273, 0.11 nats.)  So the
       failure is not reliably loud, which is why this is a pinned test and not
       a comment.

2. sampling_density.  mcsamplerPortfolio builds its balance-heuristic mixture
   denominator q_mix = sum_m frac_m q_m from ``member.sampling_density(X)``.
   NFlow had no such method, so an NFlow member forced the portfolio onto the
   legacy stratified per-member density -- which is only valid when every member
   shares one support, and which PR #356 ("portfolio: make sampling_density the
   member p_s contract") turns into a hard refusal.

NO nflows/torch REQUIRED.  mcsamplerNFlow imports torch and nflows at module
scope, which is why test_NF_reuse.py is rostered OPTDEP.  The code paths pinned
here -- the uniform ``self.nf_flow is None`` draw and its matching
sampling_density branch -- are pure numpy, so this file imports the module
behind import-time stubs when the real packages are absent.  It uses the REAL
packages when they are installed, so a dev box exercises the genuine import.

Usage:
  python -m pytest -q MonteCarloMarginalizeCode/Code/test/integrators/test_NFlow_portfolio_contract.py
"""
from __future__ import print_function

import ast
import inspect
import io
import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------- import stubs

_SUBMODULES = [
    "torch", "torch.optim", "torch.optim.lr_scheduler", "torch.utils", "torch.utils.data",
    "nflows", "nflows.flows", "nflows.flows.base", "nflows.utils",
    "nflows.distributions", "nflows.distributions.normal",
    "nflows.transforms", "nflows.transforms.normalization", "nflows.transforms.base",
    "nflows.transforms.autoregressive", "nflows.transforms.permutations",
    "nflows.transforms.standard", "nflows.transforms.lu",
    "nflows.nn", "nflows.nn.nets",
]


def _make_stub(name):
    """A module whose every attribute is a fresh permissive CLASS.

    A class (not a function) is required: mcsamplerNFlow does
    ``class TanhTransform(Transform)`` at module scope, so the stand-in has to be
    usable as a base.  This stub only has to survive IMPORT -- every path this
    file exercises is numpy-only -- so it deliberately does not try to imitate
    torch or nflows behaviour.  Anything that actually needs them belongs in
    test_NF_reuse.py, which is rostered OPTDEP.
    """
    mod = types.ModuleType(name)

    def __getattr__(attr):
        return type(str(attr), (object,), {
            "__init__": lambda self, *a, **k: None,
            "__call__": lambda self, *a, **k: None,
        })

    mod.__getattr__ = __getattr__
    if name == "torch":
        _install_torch_shim(mod)
    return mod


class _ShimTensor(object):
    """What `torch.as_tensor` returns under the stub: an array that answers
    .numpy() and .detach(), which is all the flow stand-ins below consume."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def detach(self):
        return self

    def numpy(self):
        return self._arr


class _ShimNoGrad(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_torch_shim(mod):
    """The three torch attributes sampling_density's trained branch actually uses.

    The permissive class-returning stub is not enough for these: `torch.no_grad()`
    has to be a context manager and `torch.as_tensor` has to hand the flow something
    it can read back.  This shim is NARROW on purpose -- it implements exactly these
    three names and nothing else, so it cannot quietly absorb a call this file did
    not intend to make.  It also means the stubbed arm does not prove the real torch
    path works; that is covered by running this file under a real torch+nflows
    install, and by test_NF_reuse.py (rostered OPTDEP).
    """
    mod.no_grad = lambda: _ShimNoGrad()
    mod.as_tensor = lambda x, dtype=None: _ShimTensor(x)
    mod.get_default_dtype = lambda: None


@pytest.fixture(scope="module")
def NF():
    """mcsamplerNFlow, imported with real torch/nflows if present, else stubbed.

    Restores sys.modules afterwards so a stub cannot leak into another test file
    sharing the interpreter.
    """
    try:
        import torch  # noqa: F401
        import nflows  # noqa: F401
        real = True
    except Exception:
        real = False

    saved = {}
    if not real:
        for name in _SUBMODULES:
            saved[name] = sys.modules.get(name)
            sys.modules[name] = _make_stub(name)
    saved["RIFT.integrators.mcsamplerNFlow"] = sys.modules.pop(
        "RIFT.integrators.mcsamplerNFlow", None)
    try:
        import RIFT.integrators.mcsamplerNFlow as mod
        mod._test_used_real_deps = real
        yield mod
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


# ------------------------------------------------------------------- fixtures

# Asymmetric box, and a prior that is NOT the sampling density.  Both matter: with
# a uniform prior on a symmetric box, p_s and p_prior are the same number and a
# swapped return order is undetectable.
_BOX = [(-5.0, 5.0), (0.0, 4.0), (-1.0, 3.0)]
_PARAMS = ["x0", "x1", "x2"]
_VOL = float(np.prod([hi - lo for lo, hi in _BOX]))


def _ramp(lo, hi):
    """Normalized linear ramp on [lo,hi]: integrates to 1, varies point to point."""
    w = hi - lo
    return np.vectorize(lambda x, lo=lo, w=w: 2.0 * (x - lo) / (w * w))


def _build(NF):
    s = NF.MCSampler(n_chunk=64)
    for p, (lo, hi) in zip(_PARAMS, _BOX):
        s.add_parameter(p, np.vectorize(lambda x, w=(hi - lo): 1.0 / w),
                        prior_pdf=_ramp(lo, hi),
                        left_limit=lo, right_limit=hi,
                        adaptive_sampling=True)
    assert s.nf_flow is None, "fixture must stay on the untrained (uniform) branch"
    return s


# ----------------------------------------------------------------- the contract

def test_draw_simplified_returns_ps_prior_rv(NF):
    """(p_s, p_prior, rv), positionally -- the order mcsamplerPortfolio unpacks.

    Each slot is identified by a property only it has, so EVERY permutation of the
    three fails rather than just the one that shipped:
      slot 0  constant 1/V   (the uniform sampling density)
      slot 1  varies, and equals prod(prior_pdf) at the returned samples
      slot 2  shape (ndim, n), inside the box
    """
    np.random.seed(20260916)
    s = _build(NF)
    n = 37
    out = s.draw_simplified(n, save_no_samples=True)

    assert isinstance(out, tuple) and len(out) == 3, \
        "draw_simplified must return a 3-tuple, got {!r}".format(type(out))
    p_s, p_prior, rv = out

    # slot 2: the samples
    rv = np.asarray(rv)
    assert rv.shape == (len(_PARAMS), n), \
        ("slot 2 must be the SAMPLES with shape (ndim, n)=({}, {}), got {}."
         "  A (n,) array here means the return order is (rv, p_s, p_prior)."
         .format(len(_PARAMS), n, rv.shape))
    for indx, (lo, hi) in enumerate(_BOX):
        assert np.all(rv[indx] >= lo) and np.all(rv[indx] <= hi)

    # slot 0: the sampling density, constant 1/V on the untrained branch
    p_s = np.asarray(p_s)
    assert p_s.shape == (n,), "slot 0 must be p_s with shape (n,), got {}".format(p_s.shape)
    assert np.allclose(p_s, 1.0 / _VOL), \
        "slot 0 must be the uniform sampling density 1/V={:g}, got {!r}".format(1.0 / _VOL, p_s[:4])

    # slot 1: the prior at those samples -- varies, so it cannot be confused with slot 0
    p_prior = np.asarray(p_prior)
    assert p_prior.shape == (n,), "slot 1 must be p_prior with shape (n,), got {}".format(p_prior.shape)
    expect = np.ones(n)
    for indx, (lo, hi) in enumerate(_BOX):
        expect *= _ramp(lo, hi)(rv[indx])
    assert np.allclose(p_prior, expect), "slot 1 must be prod(prior_pdf) at the returned samples"
    assert p_prior.std() > 0, "the ramp prior must vary, else this test cannot tell slot 1 from slot 0"


class _Tensorish(object):
    """The two methods the draw path calls on a flow's output: .detach().numpy()."""

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def numpy(self):
        return self._arr


class _StrictIntFlow(object):
    """A flow stand-in that is STRICT about the one thing being tested.

    It reproduces nflows.distributions.base.Distribution.sample's check verbatim
    (``check.is_positive_int`` -> ``isinstance(n, int) and n > 0``), because a
    permissive stand-in here would accept the numpy int and pass a defect that
    the real nflows rejects.  Everything else it does is the minimum the draw
    path consumes.
    """

    def __init__(self, ndim, seed=0):
        self.ndim = ndim
        self.rng = np.random.RandomState(seed)
        self.saw = []

    def sample_and_log_prob(self, num_samples):
        self.saw.append(type(num_samples))
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise TypeError("Number of samples must be a positive integer.")
        x = np.column_stack([self.rng.uniform(lo, hi, size=num_samples) for lo, hi in _BOX])
        return _Tensorish(x), _Tensorish(self._logq(x))

    def log_prob(self, t):
        return _Tensorish(self._logq(t.numpy()))

    def _logq(self, x):
        return np.full(x.shape[0], -np.log(_VOL))


def test_draw_simplified_accepts_a_numpy_int_on_the_flow_branch(NF):
    """mcsamplerPortfolio hands members n_samples_per_member[i], a numpy int64.

    nflows type-checks its sample count with isinstance(n, int), which a numpy
    integer FAILS ("Number of samples must be a positive integer"), so NFlow was
    the one member with a stricter signature than the portfolio's own call.  The
    untrained branch cannot see this -- numpy sizes accept a numpy int happily --
    so drive the flow branch through a stand-in that keeps nflows' check.

    This also pins the return order on the TRAINED branch, independently of the
    uniform-branch test above.
    """
    np.random.seed(11)
    s = _build(NF)
    n = np.array([23], dtype=np.int64)[0]
    assert not isinstance(n, int), "vacuous unless a numpy int is not a python int"

    s.nf_flow = _StrictIntFlow(len(_PARAMS), seed=4)
    p_s, p_prior, rv = s.draw_simplified(n, save_no_samples=True, enforce_bounds=True)

    assert s.nf_flow.saw, "the flow branch was not taken -- this test checked nothing"
    assert s.nf_flow.saw[0] is int, \
        ("draw_simplified passed {} to the flow; nflows requires a python int."
         .format(s.nf_flow.saw[0]))

    rv = np.asarray(rv)
    assert rv.shape == (len(_PARAMS), 23), \
        "slot 2 must be the (ndim, n) samples on the trained branch too, got {}".format(rv.shape)
    assert np.allclose(np.asarray(p_s), 1.0 / _VOL), "slot 0 must be the flow density exp(log_prob)"
    assert np.asarray(p_prior).shape == (23,) and np.asarray(p_prior).std() > 0, \
        "slot 1 must be the (varying) prior at those samples"


def test_sampling_density_exists_and_is_the_uniform_box_density(NF):
    """The method mcsamplerPortfolio needs for q_mix = sum_m frac_m q_m."""
    s = _build(NF)
    assert hasattr(s, "sampling_density"), \
        ("mcsamplerNFlow needs sampling_density(X): mcsamplerPortfolio builds its "
         "balance-heuristic mixture denominator from it, and PR #356 makes a member "
         "without one a hard error.")

    rng = np.random.RandomState(3)
    X = np.column_stack([rng.uniform(lo, hi, size=200) for lo, hi in _BOX])
    q = s.sampling_density(X)
    assert q is not None, "sampling_density must not be None once the box is registered"
    q = np.asarray(q)
    assert q.shape == (200,)
    assert np.allclose(q, 1.0 / _VOL)

    # it must agree with the p_s draw_simplified reports for its OWN draws -- that
    # agreement IS the member p_s contract the portfolio relies on
    np.random.seed(5)
    p_s, _, rv = s.draw_simplified(50, save_no_samples=True)
    assert np.allclose(np.asarray(s.sampling_density(np.asarray(rv).T)), np.asarray(p_s))

    # (ndim, N) is tolerated, as for mcsamplerAdaptiveVolume/mcsamplerEnsemble
    assert np.allclose(np.asarray(s.sampling_density(X.T)), q)

    # zero outside the box: enforce_bounds means no accepted draw lands there, so a
    # nonzero density out there would overstate this member's share of q_mix
    X_out = X.copy()
    X_out[:7, 0] = _BOX[0][1] + 1.0
    q_out = np.asarray(s.sampling_density(X_out))
    assert np.all(q_out[:7] == 0.0)
    assert np.allclose(q_out[7:], q[7:])


def test_internal_call_sites_unpack_in_contract_order(NF):
    """Source check: nothing inside mcsamplerNFlow may keep the old order.

    The behavioural tests above pin the single ``return`` statement, which both
    the trained and untrained branches share.  They cannot see the module's OWN
    call site in integrate_log -- reaching it needs a real trained flow -- and
    that call used to carry a "Beware reversed order of rv" comment precisely
    because it compensated for the defect.  Flipping the return without flipping
    the call site would leave NFlow's own integrator broken, so pin it here.
    """
    path = inspect.getsourcefile(NF)
    assert path and path.endswith(".py"), \
        "could not locate mcsamplerNFlow source (got {!r})".format(path)
    tree = ast.parse(io.open(path, encoding="utf-8").read())

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Tuple) or not isinstance(value, ast.Call):
            continue
        fn = value.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "draw_simplified"):
            continue
        names = [e.id if isinstance(e, ast.Name) else "<expr>" for e in target.elts]
        sites.append((node.lineno, names))

    assert sites, \
        ("found no tuple-unpacking call to draw_simplified in {}; this test has stopped "
         "checking anything -- re-point it at the real call site.".format(path))

    for lineno, names in sites:
        assert len(names) == 3, "line {}: expected a 3-tuple, got {}".format(lineno, names)
        # the samples must be LAST; they are the only slot whose name carries 'rv'
        assert "rv" in names[2], \
            ("{}:{} unpacks draw_simplified as {} -- the samples must be the THIRD "
             "element, matching (p_s, p_prior, rv)."
             .format(path, lineno, tuple(names)))
        for indx in (0, 1):
            assert "rv" not in names[indx], \
                ("{}:{} unpacks draw_simplified as {} -- slot {} holds the samples, but "
                 "the contract is (p_s, p_prior, rv)."
                 .format(path, lineno, tuple(names), indx))

class _SupersetFlow(object):
    """Flow stand-in that is EXACTLY uniform over a superset of the parameter box.

    Axis 0 is spread over a box-width-doubled interval and the other axes stay inside,
    so the acceptance mass is a closed-form A = 1/2 and the density is a closed-form
    constant 1/V_super.  That makes the NORMALIZATION checkable in absolute terms:
    the accepted draws are uniform on the box, so a correctly reported p_s must be
    1/V_box, and E[prior/p_s] over fresh draws must be exactly 1.

    Nothing here is position-dependent; `_GradedFlow` covers alignment.
    """

    def __init__(self, seed=0, width=2.0):
        self.rng = np.random.RandomState(seed)
        self.n_asked = []
        self.n_drawn = 0
        lo0, hi0 = _BOX[0]
        # axis 0 spread over `width` times the box width, so A == 1/width exactly
        self.super_lo0, self.super_hi0 = lo0, lo0 + width * (hi0 - lo0)
        self.v_super = (self.super_hi0 - self.super_lo0) * float(
            np.prod([hi - lo for lo, hi in _BOX[1:]]))
        self.A_exact = _VOL / self.v_super                        # == 1/width

    def _sample(self, n):
        x = np.empty((n, len(_BOX)))
        x[:, 0] = self.rng.uniform(self.super_lo0, self.super_hi0, size=n)
        for indx, (lo, hi) in enumerate(_BOX[1:], start=1):
            x[:, indx] = self.rng.uniform(lo, hi, size=n)
        return x

    def logq_at(self, X):
        return np.full(np.asarray(X).shape[0], -np.log(self.v_super))

    def sample_and_log_prob(self, num_samples):
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise TypeError("Number of samples must be a positive integer.")
        self.n_asked.append(num_samples)
        self.n_drawn += num_samples
        x = self._sample(num_samples)
        return _Tensorish(x), _Tensorish(self.logq_at(x))

    def log_prob(self, t):
        return _Tensorish(self.logq_at(t.numpy()))

    def parameters(self):
        return iter(())


class _GradedFlow(_SupersetFlow):
    """As _SupersetFlow, but with a log-density that VARIES WITH POSITION.

    A constant stand-in density cannot tell row i of `rv` from row j, so it cannot see
    a misalignment between the samples and the numbers reported for them.  With a
    strictly monotone log_q, any permutation or off-by-slice between `rv`, `log_ps`
    and `log_p` shows up elementwise.  Measured with a constant stand-in instead: a
    `[:, -n_to_get:]` slice on `rv` alone left p_s wrong by 37x and the gate green.

    log_q is not normalized, and does not need to be: these tests compare the reported
    p_s against this same function at the SAME rows.  `_SupersetFlow` covers absolute
    normalization.
    """

    SLOPE = 0.37

    def logq_at(self, X):
        X = np.atleast_2d(np.asarray(X))
        mid = 0.5 * (self.super_lo0 + self.super_hi0)
        return -np.log(self.v_super) + self.SLOPE * (X[:, 0] - mid)


class _DeadFlow(_SupersetFlow):
    """Every sample lands outside the box, so no refill can ever succeed."""

    def _sample(self, n):
        return np.full((n, len(_BOX)), _BOX[0][1] + 10.0)


def _prior_at(rv):
    """prod(prior_pdf) evaluated at the columns of an (ndim, n) sample array."""
    out = np.ones(np.asarray(rv).shape[1])
    for indx, (lo, hi) in enumerate(_BOX):
        out *= _ramp(lo, hi)(np.asarray(rv)[indx])
    return out


def test_trained_flow_rows_stay_aligned_with_their_density_and_prior(NF):
    """p_s and p_prior must belong to the SAMPLES returned beside them.

    `log_ps`, `log_p` and `rv` are concatenated across refill passes and then sliced to
    n_to_get.  Three independent slices means three chances to misalign, and a
    misalignment is invisible in any aggregate: every row still looks like a plausible
    density.  A graded stand-in density makes it elementwise-visible.
    """
    np.random.seed(21)
    s = _build(NF)
    flow = _GradedFlow(seed=1)
    s.nf_flow = flow
    n = 400
    p_s, p_prior, rv = s.draw_simplified(n, save_no_samples=True)
    rv = np.asarray(rv)

    A = s.flow_acceptance()
    expect_ps = np.exp(flow.logq_at(rv.T)) / A
    assert np.allclose(np.asarray(p_s), expect_ps, rtol=1e-9), \
        ("p_s does not match q/A AT THE RETURNED SAMPLES; worst relative error {:.3g}. "
         "The rows of log_ps and rv have come apart."
         .format(float(np.max(np.abs(np.asarray(p_s) - expect_ps) / expect_ps))))
    assert np.allclose(np.asarray(p_prior), _prior_at(rv), rtol=1e-9), \
        "p_prior does not match prod(prior_pdf) at the returned samples"
    # and the graded density must actually vary, or none of the above discriminates
    assert np.ptp(np.asarray(p_s)) / np.mean(np.asarray(p_s)) > 0.5, \
        "the stand-in density is nearly constant, so this test cannot see a misalignment"


def test_reported_density_is_normalized_over_the_box(NF):
    """E[prior/p_s] == 1 exactly when p_s is a normalized density on the box.

    This is the support/normalization probe, and it does not go through the 1/A
    algebra, so it is independent of the implementation it checks.  Before the fix it
    read 1/A: measured 6.29, 4.21, 2.88 and 2.18 on real trained flows at d = 2, 4, 6
    and 8.  The stand-in here has a closed-form A = 1/2.
    """
    np.random.seed(22)
    s = _build(NF)
    flow = _SupersetFlow(seed=2)
    s.nf_flow = flow
    p_s, p_prior, rv = s.draw_simplified(4000, save_no_samples=True)

    assert abs(s.flow_acceptance() - flow.A_exact) < 0.05, \
        "stand-in acceptance should be near the closed-form {:.3g}".format(flow.A_exact)
    # accepted draws are uniform on the box, so the reported density must be 1/V_box
    assert np.allclose(np.asarray(p_s), 1.0 / _VOL, rtol=0.05), \
        ("p_s should be 1/V_box = {:g}; got {:g}.  Without the 1/A it would be {:g}."
         .format(1.0 / _VOL, float(np.mean(np.asarray(p_s))), 1.0 / flow.v_super))
    est = float(np.mean(np.asarray(p_prior) / np.asarray(p_s)))
    assert abs(est - 1.0) < 0.05, \
        ("E[prior/p_s] = {:.4f}, not 1, so the reported p_s is not a normalized "
         "density on the box (1/A would give {:.3f})".format(est, 1.0 / flow.A_exact))


def test_sampling_density_is_zero_outside_the_box_on_a_trained_flow(NF):
    """Only the UNTRAINED branch was checked for this, and it is the trained branch
    that matters to the portfolio.

    A member whose q_m is nonzero where it cannot draw inflates q_mix on every other
    member's samples and biases ln Z low with no diagnostic.  Measured with the
    `inside &` guard removed: sampling_density returned 0.241 outside the box.
    """
    np.random.seed(23)
    s = _build(NF)
    s.nf_flow = _GradedFlow(seed=3)
    s.draw_simplified(200, save_no_samples=True)   # establishes the acceptance

    rng = np.random.RandomState(4)
    X = np.column_stack([rng.uniform(lo, hi, size=50) for lo, hi in _BOX])
    q_in = np.asarray(s.sampling_density(X))
    assert np.all(q_in > 0)
    X_out = X.copy()
    X_out[:12, 0] = _BOX[0][1] + 5.0
    q_out = np.asarray(s.sampling_density(X_out))
    assert np.all(q_out[:12] == 0.0), \
        ("sampling_density returned {!r} outside the box on a trained flow; the draws "
         "are truncated there, so the density must be 0".format(q_out[:3]))
    assert np.allclose(q_out[12:], q_in[12:])


def test_zero_draws_returns_empty_arrays(NF):
    """mcsamplerPortfolio.draw can allocate a member zero draws.

    Its comment says the per-member count "Can be zero" and it has an explicit
    n_samples_per_member[-1] = 0 branch.  The refill loop never runs at n_to_get == 0,
    and np.concatenate([]) raises "need at least one array to concatenate".
    mcsamplerAdaptiveVolume returns empty arrays here, so match it.
    """
    np.random.seed(24)
    s = _build(NF)
    s.nf_flow = _GradedFlow(seed=5)
    for n in (0, np.int64(0)):
        p_s, p_prior, rv = s.draw_simplified(n, save_no_samples=True)
        assert np.asarray(p_s).shape == (0,), "p_s should be empty, got {}".format(np.asarray(p_s).shape)
        assert np.asarray(p_prior).shape == (0,)
        assert np.asarray(rv).shape == (len(_PARAMS), 0), \
            "rv should be (ndim, 0), got {}".format(np.asarray(rv).shape)


def test_enforce_bounds_false_draws_exactly_what_was_asked(NF):
    """With no truncation, one pass of exactly n_to_get is necessary and sufficient.

    Sizing by 1/A in this mode over-drew 8x, and consulting flow_acceptance() took a
    BOUNDS-ENFORCING probe batch on behalf of a caller that asked for bounds not to be
    enforced.
    """
    np.random.seed(25)
    s = _build(NF)
    flow = _GradedFlow(seed=6)
    s.nf_flow = flow
    n = 500
    p_s, p_prior, rv = s.draw_simplified(n, save_no_samples=True, enforce_bounds=False)

    assert np.asarray(rv).shape == (len(_PARAMS), n)
    assert flow.n_drawn == n, \
        "drew {} rows for a request of {}; nothing is discarded in this mode".format(flow.n_drawn, n)
    assert flow.n_asked == [n], "expected one pass of exactly n, got asks {}".format(flow.n_asked)
    # no 1/A here: the draws are NOT truncated, so q itself is the density
    assert np.allclose(np.asarray(p_s), np.exp(flow.logq_at(np.asarray(rv).T)), rtol=1e-9)
    # and no bounds-enforcing statistics were invented for this generation
    assert s.flow_acceptance_history()[0] == 0, \
        "an enforce_bounds=False batch must not contribute to the acceptance statistics"


def test_acceptance_is_per_flow_generation_and_reset_on_retrain(NF):
    """A is a property of ONE flow.

    Pooling it across retrains normalizes each chunk by the wrong number (measured:
    ln Z biased -0.179 nats over 6 seeds).  Using only the latest batch is unbiased but
    noisy (0.13 nats per chunk at ~46 draws, which is what a floored portfolio member
    gets).  So it accumulates WITHIN a generation and resets when a flow is installed.
    """
    np.random.seed(26)
    s = _build(NF)
    s.nf_flow = _DeadFlow(seed=7)
    try:
        s.draw_simplified(50, save_no_samples=True)
    except Exception:
        pass                       # expected; see the exhaustion test
    drawn_dead = s.flow_acceptance_history()[0]
    assert drawn_dead > 0, "the dead flow should have left statistics behind"

    # installing a new flow must discard them
    s.nf_flow = _SupersetFlow(seed=8)
    s._reset_flow_acceptance()
    assert s.flow_acceptance_history() == (0, 0, None), \
        "a new flow generation must start with no acceptance statistics"

    s.draw_simplified(300, save_no_samples=True)
    n_drawn_1, n_kept_1, A1 = s.flow_acceptance_history()
    assert 0.3 < A1 < 0.7, "acceptance should be near the closed-form 0.5, got {!r}".format(A1)
    # a second batch ACCUMULATES rather than replacing
    s.draw_simplified(300, save_no_samples=True)
    n_drawn_2, n_kept_2, A2 = s.flow_acceptance_history()
    assert n_drawn_2 > n_drawn_1 and n_kept_2 > n_kept_1, \
        "a second batch from the same flow must add to the generation's statistics"
    assert 0.3 < A2 < 0.7


def test_every_flow_install_site_resets_the_acceptance(NF):
    """Testing _reset_flow_acceptance() does not prove the INSTALL SITES call it.

    There are three places self.nf_flow is replaced -- setup(), the end of
    update_sampling_prior(), and the warm-load in integrate_log().  A cached A from the
    previous flow normalizes the new one by the wrong number: measured on that path,
    flow_acceptance() reported 0.8849 for a flow whose true A was 0.9295, so
    sampling_density() was 1.05x off.
    """
    path = inspect.getsourcefile(NF)
    tree = ast.parse(io.open(path, encoding="utf-8").read())

    installs, resets = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and t.attr == 'nf_flow' \
                    and not (isinstance(node.value, ast.Constant) and node.value.value is None):
                installs.append(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == '_reset_flow_acceptance':
            resets.append(node.lineno)

    assert len(installs) >= 3, \
        "expected at least 3 nf_flow install sites, found {}".format(installs)
    # every install must have a reset call within a few lines of it
    for lineno in installs:
        assert any(abs(r - lineno) <= 4 for r in resets), \
            ("{}:{} installs a new flow with no _reset_flow_acceptance() nearby "
             "(resets at {})".format(path, lineno, resets))


def test_trained_flow_draw_returns_exactly_what_was_asked(NF):
    """mcsamplerPortfolio.draw copies member output into a FIXED-WIDTH slice.

    enforce_bounds discards the flow samples outside the box, so one batch returns
    fewer than requested and the portfolio died with "could not broadcast input
    array from shape (45,) into shape (114,)".  draw_simplified refills instead.
    """
    np.random.seed(2)
    s = _build(NF)
    flow = _GradedFlow(seed=11)
    s.nf_flow = flow
    n = 500
    p_s, p_prior, rv = s.draw_simplified(n, save_no_samples=True)

    assert np.asarray(rv).shape == (len(_PARAMS), n), \
        ("draw_simplified must return exactly n_to_get samples; got {} for n={}. "
         "A short batch aborts mcsamplerPortfolio.draw."
         .format(np.asarray(rv).shape, n))
    assert np.asarray(p_s).shape == (n,) and np.asarray(p_prior).shape == (n,)
    # it had to draw MORE than it returned, which is the whole point
    assert flow.n_drawn > n, \
        "with half the mass outside the box, filling n={} must draw more than n".format(n)
    # and every returned sample is inside the box
    for indx, (lo, hi) in enumerate(_BOX):
        assert np.all(np.asarray(rv)[indx] >= lo) and np.all(np.asarray(rv)[indx] <= hi)


def test_refill_exhaustion_raises_with_the_measured_acceptance(NF):
    """A flow whose mass has left the box cannot be refilled at any price.

    Bail with a message naming the acceptance rather than spinning, and rather than
    returning short and reaching the portfolio as a broadcast error.
    """
    np.random.seed(5)
    s = _build(NF)
    s.nf_flow = _DeadFlow()
    with pytest.raises(Exception) as excinfo:
        s.draw_simplified(100, save_no_samples=True)
    msg = str(excinfo.value)
    assert "acceptance" in msg and "mcsamplerNFlow" in msg, \
        "the exhaustion error must name itself and the measured acceptance: {!r}".format(msg)


def test_sampling_density_matches_reported_ps_on_a_trained_flow(NF):
    """q_m in the portfolio mixture must be the same quantity as the member's p_s.

    Both carry the 1/A; if only one does, q_mix is wrong by 1/A on this member's
    share of every chunk.
    """
    np.random.seed(6)
    s = _build(NF)
    s.nf_flow = _GradedFlow(seed=12)
    p_s, _, rv = s.draw_simplified(300, save_no_samples=True)
    q = s.sampling_density(np.asarray(rv).T)
    assert q is not None
    assert np.allclose(np.asarray(q), np.asarray(p_s)), \
        "sampling_density and draw_simplified's p_s must agree on the trained branch"


def test_declares_its_joint_p_s_is_a_normalized_density(NF):
    """The class attribute mcsamplerPortfolio reads at setup().

    Without it the portfolio treats the member as "scale unknown" and refuses to
    pool it with mcsamplerAdaptiveVolume, which declares False.  It is only true
    BECAUSE draw_simplified divides by A; the two must move together.
    """
    assert NF.MCSampler.joint_p_s_is_normalized_density is True


def test_portfolio_accepts_an_nflow_member_at_setup(NF):
    """End of the road for both defects: the real contract check, run for real.

    mcsamplerPortfolio.setup() refuses a member that has no sampling_density and is
    pooled with one whose joint_p_s is not a declared density (mcsamplerAdaptiveVolume
    declares False).  [AV, NFlow] was exactly that combination.
    """
    from RIFT.integrators import mcsamplerAdaptiveVolume as AVmod
    from RIFT.integrators import mcsamplerPortfolio as Pmod

    av = AVmod.MCSampler()
    assert getattr(AVmod.MCSampler, 'joint_p_s_is_normalized_density', None) is False, \
        "this test is vacuous unless AV still declares its joint_p_s is NOT a density"
    port = Pmod.MCSampler(portfolio=[av, NF.MCSampler()], n_chunk=32)
    for p, (lo, hi) in zip(_PARAMS, _BOX):
        port.add_parameter(p, np.vectorize(lambda x, w=(hi - lo): 1.0 / w),
                           prior_pdf=_ramp(lo, hi),
                           left_limit=lo, right_limit=hi, adaptive_sampling=True)
    port.setup(portfolio_breakpoints=None, n=32)   # must not raise


def test_one_parameter_flow_has_a_trainable_layer(NF):
    """num_layers = int(d/2) is 0 at d=1, so the transform had no trainable weights
    and training died in torch with "optimizer got an empty parameter list".

    d=1 is the case worth guarding: it is where the original return-order defect
    completed silently instead of raising.
    """
    s = NF.MCSampler(n_chunk=16)
    lo, hi = _BOX[0]
    s.add_parameter(_PARAMS[0], np.vectorize(lambda x, w=(hi - lo): 1.0 / w),
                    prior_pdf=_ramp(lo, hi), left_limit=lo, right_limit=hi,
                    adaptive_sampling=True)
    s.setup()
    assert s.num_layers >= 1, \
        "a 1-parameter flow needs at least one autoregressive layer; got {}".format(s.num_layers)


def test_affine_mean_scale_is_shape_safe_at_one_parameter(NF):
    """update_sampling_prior's PointwiseAffine pre-conditioning step.

    It computed np.diag(np.cov(samples_train)).  np.cov of a (1, n) array is 0-d and
    np.diag then raises "Input must be 1- or 2-d", so a ONE-parameter sampler crashed
    there as soon as it had enough history to train -- after the portfolio had already
    spent a chunk of likelihood.  d=1 is also where the return-order defect was silent
    rather than loud, so it is the configuration worth being able to run.
    """
    rng = np.random.RandomState(9)
    for ndim in (1, 2, 5):
        samples = rng.normal(size=(ndim, 400))
        mean, scale = NF.MCSampler._affine_mean_scale(samples)
        assert np.asarray(mean).shape == (ndim,), \
            "mean must be per-axis, shape (ndim,); got {}".format(np.asarray(mean).shape)
        assert np.asarray(scale).shape == (ndim,), \
            ("scale must be per-axis, shape (ndim,); got {} at ndim={}"
             .format(np.asarray(scale).shape, ndim))
        assert np.all(np.asarray(scale) > 0)
        assert np.allclose(mean, np.mean(samples, axis=1))
        if ndim > 1:
            # unchanged where it already worked -- BIT-identical, not merely close
            assert np.array_equal(np.asarray(scale), np.diag(np.cov(samples))), \
                "ndim>=2 must be bit-identical to the original np.diag(np.cov(...))"


def test_call_site_uses_the_shape_safe_helper(NF):
    """Testing the helper does not prove the CALL SITE stopped doing it inline.

    One grep for the exact original expression: a reintroduced
    np.diag(np.cov(samples_train)) anywhere in the module brings the d=1 crash back
    while every test above still passes.
    """
    path = inspect.getsourcefile(NF)
    tree = ast.parse(io.open(path, encoding="utf-8").read())

    def _attr(node, name):
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == name

    # AST, not a grep: the prose in _affine_mean_scale's own docstring names the old
    # expression, and a string search cannot tell that from code.
    offenders = [n.lineno for n in ast.walk(tree)
                 if _attr(n, 'diag') and n.args and _attr(n.args[0], 'cov')]
    assert not offenders, \
        ("{} calls diag(cov(...)) directly at line(s) {}; that raises \"Input must be "
         "1- or 2-d\" at ndim=1.  Use _affine_mean_scale(), which wraps cov in "
         "atleast_2d.".format(path, offenders))

    # and the call site must actually route through the helper
    upd = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == 'update_sampling_prior']
    assert len(upd) == 1, "expected one update_sampling_prior, found {}".format(len(upd))
    assert any(_attr(n, '_affine_mean_scale') for n in ast.walk(upd[0])), \
        "update_sampling_prior must get its affine mean/scale from _affine_mean_scale()"


def test_refill_takes_several_passes_when_acceptance_is_low(NF):
    """_NF_REFILL_MAX_PASSES has to be larger than 1 for the refill to be worth having.

    One pass asks for 1.15/A times what is needed, so at a high acceptance it already
    suffices and capping the loop at a single pass changes nothing.  The bound only
    does work at a LOW acceptance, where the per-pass memory cap
    (_NF_REFILL_MAX_BATCH_FACTOR * n_to_get) stops one pass from covering 1/A.  This
    drives that corner: A = 0.1 exactly, so one capped pass cannot fill the chunk.
    """
    np.random.seed(27)
    s = _build(NF)
    flow = _SupersetFlow(seed=13, width=10.0)      # A == 0.1
    s.nf_flow = flow
    n = 500
    p_s, p_prior, rv = s.draw_simplified(n, save_no_samples=True)

    assert np.asarray(rv).shape == (len(_PARAMS), n)
    # the probe is one ask; the refill passes are the rest, and there must be several
    refill_asks = [a for a in flow.n_asked if a != 4000]
    assert len(refill_asks) >= 2, \
        ("at A=0.1 with the per-pass cap at {}x n, filling n={} needs more than one "
         "pass; asks were {}".format(4, n, flow.n_asked))
    assert abs(s.flow_acceptance() - 0.1) < 0.03, \
        "acceptance should be near the closed-form 0.1, got {!r}".format(s.flow_acceptance())
    assert np.allclose(np.asarray(p_s), 1.0 / _VOL, rtol=0.1), \
        "p_s must still be the normalized 1/V_box after a multi-pass refill"


def test_num_layers_is_unchanged_above_one_parameter(NF):
    """The max(1, ...) fix must be a d=1 fix ONLY.

    The claim made for this change is that it alters nothing at d>=2; anything else
    would retrain every existing flow architecture silently.  So pin the old formula
    for d>=2 rather than just asserting num_layers >= 1, which a +1 anywhere would
    also satisfy.
    """
    for ndim in (1, 2, 3, 4, 6, 8):
        s = NF.MCSampler(n_chunk=16)
        for indx in range(ndim):
            lo, hi = -1.0 - indx, 2.0 + indx
            s.add_parameter("p{}".format(indx),
                            np.vectorize(lambda x, w=(hi - lo): 1.0 / w),
                            prior_pdf=np.vectorize(lambda x, w=(hi - lo): 1.0 / w),
                            left_limit=lo, right_limit=hi, adaptive_sampling=True)
        s.setup()
        if ndim == 1:
            assert s.num_layers == 1, \
                "d=1 needs one layer to have any trainable weights, got {}".format(s.num_layers)
        else:
            assert s.num_layers == int(ndim / 2), \
                ("d={} must keep the original int(d/2)={} layers, got {}"
                 .format(ndim, int(ndim / 2), s.num_layers))
