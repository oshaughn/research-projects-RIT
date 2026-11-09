"""--eos-affine-parameter central: dM-per-branch prior and branch Monte Carlo."""
import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import lal
import numpy as np
import pytest

from RIFT import lalsimutils
from RIFT.physics import eos_affine

CODE = str(Path(__file__).resolve().parents[1])
DRIVER = os.path.join(CODE, "bin", "util_ConstructIntrinsicPosterior_GenericCoordinates.py")


def _msun(m):
    return m / lal.MSUN_SI if m > 1e15 else m


class FakeTwinEOS:
    """Two stable branches overlapping on [1.6, 2.0] Msun; Lambda known per branch.

    Mass mode sees the primary-branch projection (as nmbseq v1 does):
    mMaxMsun is the primary maximum and lambda_from_m(m) ignores branch 1.
    """
    bounds = ((0.5, 2.0), (1.6, 2.4))
    mMinMsun = 0.5
    mMaxMsun = 2.0
    mMaxMsun_all = 2.4

    @staticmethod
    def branch_lambda(m, b):
        return (400.0, 40.0)[b] * (3.0 - m)

    def branches_for_m(self, m):
        m = _msun(m)
        return [b for b, (lo, hi) in enumerate(self.bounds) if lo <= m <= hi]

    def lambda_from_m(self, m, branch_id=None):
        m = _msun(m)
        if branch_id is None:
            if m > 0.999 * self.mMaxMsun:
                return 1e-8
            return self.branch_lambda(m, 0)
        lo, hi = self.bounds[branch_id]
        if not lo <= m <= hi:
            raise ValueError("mass outside branch")
        return self.branch_lambda(m, branch_id)

    def lambda_from_m_vector(self, m, branch_id=None):
        return np.array([self.lambda_from_m(x, branch_id=branch_id) for x in np.atleast_1d(m)])


class FakeSingleEOS(FakeTwinEOS):
    bounds = ((0.5, 2.0),)
    mMaxMsun_all = 2.0


class FailingTwinEOS(FakeTwinEOS):
    """Raises or returns NaN at selected masses; neighbours must survive."""
    def branches_for_m(self, m):
        if np.isclose(_msun(m), 1.3):
            raise RuntimeError("synthetic branch enumeration failure")
        return super().branches_for_m(m)

    def lambda_from_m(self, m, branch_id=None):
        m = _msun(m)
        if np.isclose(m, 1.4):
            raise RuntimeError("synthetic native failure")
        if np.isclose(m, 1.8) and branch_id == 1:
            return np.nan
        return super().lambda_from_m(m, branch_id=branch_id)


def _central(eos, x, u, coord_names=("m1", "m2"), low=("m1", "m2"), **kw):
    return eos_affine.convert_waveform_coordinates_with_eos_central(
        np.asarray(x, float), np.asarray(u, float), coord_names=list(coord_names),
        low_level_coord_names=list(low), eos_class=eos, **kw)


def _protect(fn):
    def guarded(rows):
        rows = np.atleast_2d(rows)
        good = np.all(np.isfinite(rows), axis=1)
        out = np.full(len(rows), -np.inf)
        if good.any():
            out[good] = fn(rows[good])
        return out
    return guarded


# ---------------------------------------------------------------- branch MC

def test_branch_selection_and_weights_for_twin_masses():
    eos = FakeTwinEOS()
    x = [[1.8, 1.2], [1.8, 1.2], [2.2, 1.7], [1.9, 1.9]]
    u = [[0.1, 0.9], [0.9, 0.9], [0.5, 0.2], [0.99, 1.0]]
    x_out, log_w, branches = _central(eos, x, u, coord_names=("lambda1", "lambda2"))
    np.testing.assert_array_equal(branches, [[0, 0], [1, 0], [1, 0], [1, 1]])
    np.testing.assert_allclose(log_w, np.log([2, 2, 2, 4]))
    expect = [[FakeTwinEOS.branch_lambda(m, b) for m, b in zip(row, br)]
              for row, br in zip(x, branches)]
    np.testing.assert_allclose(x_out, expect)


def test_branch_monte_carlo_reproduces_exact_branch_sum():
    eos = FakeTwinEOS()
    fit = _protect(lambda X: -0.5 * ((X[:, 0] - 300.) / 150.) ** 2 - 0.5 * ((X[:, 1] - 60.) / 80.) ** 2)
    conv = lambda x, u: _central(eos, x, u, coord_names=("lambda1", "lambda2"))[:2]
    lnL, _ = eos_affine.make_central_log_likelihood(fit, conv, lambda x: np.zeros(len(x)), 2)

    def exact(m1, m2):
        return sum(np.exp(fit(np.array([[eos.branch_lambda(m1, b1), eos.branch_lambda(m2, b2)]])))[0]
                   for b1 in eos.branches_for_m(m1) for b2 in eos.branches_for_m(m2))

    # Fixed masses, stratified u: the branch average is exact, not just unbiased.
    for m1, m2 in [(1.8, 1.7), (1.8, 1.2), (2.2, 1.9), (1.0, 1.1)]:
        uu = np.array([(i + .5) / 4 for i in range(4)])
        U1, U2 = [a.ravel() for a in np.meshgrid(uu, uu)]
        vals = np.exp(lnL(np.full(16, m1), np.full(16, m2), U1, U2))
        assert np.mean(vals) == pytest.approx(exact(m1, m2), rel=1e-12)

    # Masses and branch labels sampled together: Monte Carlo agrees with the
    # branch sum integrated by quadrature over the mass box.
    rng = np.random.default_rng(3)
    n = 200000
    lo, hi = 1.2, 2.3
    m1, m2 = rng.uniform(lo, hi, (2, n))
    vals = np.exp(lnL(m1, m2, *rng.uniform(0, 1, (2, n))))
    mc, err = vals.mean(), vals.std() / np.sqrt(n)
    grid = np.linspace(lo, hi, 221)[:-1] + (hi - lo) / 440
    quad = np.mean([exact(a, b) for a in grid for b in grid])
    assert abs(mc - quad) < 4 * err, (mc, quad, err)


def test_single_branch_eos_central_equals_mass_exactly():
    eos = FakeSingleEOS()
    rng = np.random.default_rng(11)
    m1 = rng.uniform(0.6, 2.3, 400)
    m2 = rng.uniform(0.6, 2.3, 400)
    x = np.column_stack([m1, m2])
    u = rng.uniform(0, 1, (400, 2))
    names = ["mc", "eta", "LambdaTilde", "DeltaLambdaTilde"]
    mass = lalsimutils.convert_waveform_coordinates_with_eos(
        x, coord_names=names, low_level_coord_names=["m1", "m2"], eos_class=eos)
    central, log_w, branches = _central(eos, x, u, coord_names=names)
    assert np.isneginf(mass).any() and np.isfinite(mass).any()
    np.testing.assert_array_equal(central, mass)
    np.testing.assert_array_equal(log_w, 0.)
    ok = np.all(np.isfinite(mass), axis=1)
    np.testing.assert_array_equal(branches[ok], 0)

    fit = _protect(lambda X: -0.5 * ((X[:, 2] - 400.) / 300.) ** 2 - X[:, 0])
    conv = lambda xx, uu: _central(eos, xx, uu, coord_names=names)[:2]
    lnL_c, _ = eos_affine.make_central_log_likelihood(fit, conv, lambda xx: np.zeros(len(xx)), 2)
    lnL_m = fit(mass)
    np.testing.assert_array_equal(lnL_c(m1, m2, u[:, 0], u[:, 1]), lnL_m)
    assert lnL_c(*map(float, (m1[0], m2[0], u[0, 0], u[0, 1]))) == lnL_m[0]


def test_failing_evaluations_reject_samples_and_keep_neighbours():
    eos = FailingTwinEOS()
    x = [[1.2, 1.1], [1.3, 1.1], [1.4, 1.1], [1.8, 1.1], [1.8, 1.1], [2.5, 1.1], [1.9, 0.4], [1.7, 1.0]]
    u = [[.5, .5], [.5, .5], [.5, .5], [.9, .5], [.1, .5], [.5, .5], [.5, .5], [.5, 1.5]]
    x_out, log_w, branches = _central(eos, x, u, coord_names=("lambda1", "lambda2"))
    bad = ~np.all(np.isfinite(x_out), axis=1)
    # 1.3: enumeration raises; 1.4: native raises; (1.8, branch 1): NaN;
    # 2.5: above every branch; 0.4: below every branch; u outside [0, 1].
    np.testing.assert_array_equal(bad, [False, True, True, True, False, True, True, True])
    assert np.isneginf(x_out[bad]).all()
    np.testing.assert_array_equal(log_w[bad], 0.)
    np.testing.assert_array_equal(branches[bad], eos_affine.NO_BRANCH)
    assert x_out[4, 0] == pytest.approx(eos.branch_lambda(1.8, 0))
    assert log_w[4] == pytest.approx(np.log(2))


def test_no_matter_component_has_unit_weight():
    eos = FakeTwinEOS()
    x_out, log_w, br = _central(eos, [[5.0, 1.8]], [[.3, .7]], coord_names=("lambda1", "lambda2"), no_matter1=True)
    np.testing.assert_allclose(x_out, [[0., eos.branch_lambda(1.8, 1)]])
    assert log_w[0] == pytest.approx(np.log(2))
    np.testing.assert_array_equal(br, [[eos_affine.NO_BRANCH, 1]])


def test_real_eosmanager_multibranch_family(monkeypatch):
    """EOSConcrete + fake reviewed-multibranch lalsim: the surface central mode uses."""
    from RIFT.physics import EOSManager

    class Lalsim:
        bounds = tuple((a * lal.MSUN_SI, b * lal.MSUN_SI) for a, b in ((1.0, 2.0), (1.3, 3.0)))
        def SimNeutronStarEOSFromFilePhaseTransition(self, fname): return "eos"
        def CreateSimNeutronStarFamilyPT(self, eos, min_fam): return "fam"
        def SimNeutronStarFamNumberOfBranches(self, fam): return 2
        def SimNeutronStarFamBranchMinMass(self, b, fam): return self.bounds[b][0]
        def SimNeutronStarFamBranchMaxMass(self, b, fam): return self.bounds[b][1]
        def SimNeutronStarFamBranchRadius(self, m, b, fam): return (12e3, 9e3)[b]
        def SimNeutronStarFamBranchLoveNumberK2(self, m, b, fam): return (0.1, 0.05)[b]
        def SimNeutronStarFamBranchCentralPressure(self, m, b, fam): return 1.
    monkeypatch.setattr(EOSManager, "lalsim", Lalsim())
    eos = EOSManager.EOSLALSimulationFromFile("twin.dat", phase_transition_aware=True)
    assert eos_affine.require_branch_surface(eos) == pytest.approx(3.0)
    x_out, log_w, br = _central(eos, [[1.5, 1.2], [1.5, 1.2], [2.5, 0.9]], [[.2, .2], [.7, .2], [.5, .5]],
                                coord_names=("lambda1", "lambda2"))
    np.testing.assert_array_equal(br, [[0, 0], [1, 0], [eos_affine.NO_BRANCH] * 2])
    np.testing.assert_allclose(log_w, [np.log(2), np.log(2), 0.])
    assert x_out[1, 0] < x_out[0, 0]   # the compact branch has the smaller Lambda
    np.testing.assert_allclose(x_out[0, 0], eos.lambda_from_m(1.5, branch_id=0))
    assert np.isneginf(x_out[2]).all()   # 0.9 Msun lies below every branch


# --------------------------------------------- evidence through RIFT samplers

BOX = (1.2, 2.3)
# n(m) = 1 on [1.2,1.6), 2 on [1.6,2.0], 1 on (2.0,2.3]; mass mode stops at 2.0.
ANALYTIC_RATIO = ((0.4 + 2 * 0.4 + 0.3) / (2.0 - 1.2)) ** 2
# CIP-style (sampler, use_lnL) pairs: adaptive_cartesian rejects --internal-use-lnL,
# AV forces it, GMM allows both.
CONSTANT_CASES = [("adaptive_cartesian", False), ("GMM", False), ("GMM", True)]
PEAKED_CASES = [("adaptive_cartesian", False), ("GMM", True), ("AV", True)]
PEAK_NAMES = ["m1", "m2", "lambda1"]


def _peak(X):
    return (-0.5 * ((X[:, 0] - 1.85) / 0.12) ** 2 - 0.5 * ((X[:, 1] - 1.75) / 0.12) ** 2
            - 0.5 * ((X[:, 2] - 200.) / 150.) ** 2)


def _peaked_quadrature(central, npts=330):
    eos = FakeTwinEOS()
    lo, hi = BOX
    g = np.linspace(lo, hi, npts + 1)[:-1] + (hi - lo) / (2 * npts)
    A, B = [a.ravel() for a in np.meshgrid(g, g, indexing="ij")]
    member = lambda m, b: (eos.bounds[b][0] <= m) & (m <= eos.bounds[b][1])
    if not central:
        ok = (A < eos.mMaxMsun) & (B < eos.mMaxMsun)
        vals = np.exp(_peak(np.column_stack([A, B, eos.branch_lambda(A, 0)])))
        return np.mean(np.where(ok, vals, 0.))
    total = np.zeros_like(A)
    for b1 in (0, 1):
        n2 = member(B, 0).astype(float) + member(B, 1)
        total += member(A, b1) * n2 * np.exp(_peak(np.column_stack([A, B, eos.branch_lambda(A, b1)])))
    return np.mean(total)


def _sampler_evidence(kind, central, use_lnL, peaked=False, nmax=60000):
    import RIFT.integrators.mcsampler as mcsampler
    eos = FakeTwinEOS()
    names_fit = PEAK_NAMES if peaked else ["m1", "m2"]
    fit = _protect(_peak if peaked else (lambda X: np.zeros(len(X))))
    names = ["m1", "m2"]
    if central:
        conv = lambda x, u: _central(eos, x, u, coord_names=names_fit)[:2]
        lnL, _ = eos_affine.make_central_log_likelihood(fit, conv, lambda x: np.zeros(len(x)), 2)
        names = names + list(eos_affine.CENTRAL_AUX_NAMES)
    else:
        lnL = lambda m1, m2: fit(lalsimutils.convert_waveform_coordinates_with_eos(
            np.column_stack([m1, m2]), coord_names=names_fit, low_level_coord_names=["m1", "m2"], eos_class=eos))
    width = BOX[1] - BOX[0]
    extra = {}
    if kind == "GMM":
        import RIFT.integrators.mcsamplerEnsemble as module
        sampler = module.MCSampler()
        # As CIP builds it: native groups, then one 1d group per branch variate.
        extra["gmm_dict"] = {(k,): None for k in range(len(names))}
        extra.update(n_comp=1, max_iter=40, L_cutoff=1e-10)
    elif kind == "AV":
        import RIFT.integrators.mcsamplerAdaptiveVolume as module
        sampler = module.MCSampler()
    else:
        sampler = mcsampler.MCSampler()
    for p in ["m1", "m2"]:
        sampler.add_parameter(p, pdf=np.vectorize(lambda x: 1. / width), prior_pdf=lambda x: np.ones_like(x) / width,
                              left_limit=BOX[0], right_limit=BOX[1], adaptive_sampling=True)
    if central:
        for p in eos_affine.CENTRAL_AUX_NAMES:
            sampler.add_parameter(p, pdf=np.vectorize(lambda x: 1.), prior_pdf=eos_affine.unit_uniform_prior,
                                  left_limit=0., right_limit=1., adaptive_sampling=True)
    fn = lnL
    if use_lnL:
        extra.update(use_lnL=True, return_lnI=True)
    else:
        fn = lambda *a: np.exp(lnL(*a))
    np.random.seed(5)
    res, var, neff, _ = sampler.integrate(fn, *names, nmax=nmax, n=4000 if peaked else 2000, neff=1e9,
                                          save_intg=True, no_protect_names=True, tempering_exp=0.1,
                                          floor_level=1e-3, enforce_bounds=True, verbose=False, **extra)
    return float(res) if use_lnL else float(np.log(res))


@pytest.fixture
def float128_alias(monkeypatch):
    # The samplers' lnL path spells np.float128, absent on macOS arm64 numpy.
    if not hasattr(np, "float128"):
        monkeypatch.setattr(np, "float128", np.longdouble, raising=False)


@pytest.mark.parametrize("kind,use_lnL", CONSTANT_CASES)
def test_constant_likelihood_evidence_ratio_is_branch_count_prior(kind, use_lnL, float128_alias):
    ln_c = _sampler_evidence(kind, True, use_lnL)
    ln_m = _sampler_evidence(kind, False, use_lnL)
    # 5% is several MC sigma here, and far below the 3.5x effect being tested.
    assert np.exp(ln_c - ln_m) == pytest.approx(ANALYTIC_RATIO, rel=0.05)
    assert np.exp(ln_m) == pytest.approx(((2.0 - BOX[0]) / (BOX[1] - BOX[0])) ** 2, rel=0.05)


@pytest.mark.parametrize("kind,use_lnL", PEAKED_CASES)
def test_peaked_lambda_likelihood_evidence_matches_branch_sum(kind, use_lnL, float128_alias):
    """Lambda-dependent likelihood: each sampler recovers the exact branch-summed evidence."""
    for central in (True, False):
        ln_z = _sampler_evidence(kind, central, use_lnL, peaked=True, nmax=100000)
        assert np.exp(ln_z) == pytest.approx(_peaked_quadrature(central), rel=0.03), (kind, central)


# ------------------------------------------------- mass-mode regression

# Values from the driver at base commit 259baea8d (before this option existed),
# for the inputs below; the mass path must reproduce them bit for bit.
NEGINF = -np.inf
MASS_MODE_CONVERTED_259baea8d = [[0.9999999999999999, 0.25, 740.5206580011862, -0.0], [1.0937499999999996, 0.24984375, 697.4042385086856, -13.745305341594364], [1.1874999999999998, 0.24937500000000007, 653.5067756367829, -24.078053110026023], [1.2812500000000002, 0.24859375, 607.75483662792, -30.84848760934423], [1.3749999999999998, 0.2475, 559.0440841738501, -33.918381956221126], [1.4687499999999998, 0.24609375000000003, 506.238199054308, -33.16345723309518], [NEGINF, NEGINF, NEGINF, NEGINF], [NEGINF, NEGINF, NEGINF, NEGINF], [NEGINF, NEGINF, NEGINF, NEGINF]]
MASS_MODE_LNL_259baea8d = [-6.025730626572479, -4.548262540149953, -3.282728558373963, -2.2232771059578154, -1.368966351274582, -0.7246065165976505, NEGINF, NEGINF, NEGINF]


def _mass_mode_driver_namespace():
    tree = ast.parse(open(DRIVER).read())
    conv = next(n for n in tree.body if isinstance(n, ast.If) and "using_eos" in ast.dump(n.test)
                and any(isinstance(i, ast.FunctionDef) and i.name == "convert_coords" for i in n.orelse))
    uni = next(n for n in tree.body if isinstance(n, ast.If) and "eccentric_phase_coordinates" in ast.dump(n.test)
               and any(isinstance(i, ast.FunctionDef) and i.name == "log_likelihood_function" for i in n.body))
    guard = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "protect_fit_against_out_of_support"]
    ns = dict(np=np, lalsimutils=lalsimutils, my_eos=FakeTwinEOS(), source_redshift=0.,
              opts=types.SimpleNamespace(using_eos="lal_X", no_matter1=False, no_matter2=False,
                                         downselect_enforce_kerr=False, fref=20., eccentric_phase_coordinates=False),
              coord_names=["mc", "eta", "LambdaTilde", "DeltaLambdaTilde"], low_level_coord_names=["mc", "delta_mc"],
              my_fit=lambda X: -0.5 * ((X[:, 2] - 300.) / 200.) ** 2 - 10 * (X[:, 0] - 1.6) ** 2 + X[:, 3] * 1e-3,
              my_log_prior_scale=lambda X: 0.1 * X[:, 1])
    exec(compile(ast.Module(body=guard + [conv, uni], type_ignores=[]), DRIVER, "exec"), ns)
    return ns


def test_mass_mode_integrand_unchanged_from_base_commit():
    ns = _mass_mode_driver_namespace()
    x = np.column_stack([np.linspace(1.0, 1.75, 9), np.linspace(0., 0.2, 9)])
    np.testing.assert_array_equal(ns["convert_coords"](x), MASS_MODE_CONVERTED_259baea8d)
    np.testing.assert_array_equal(ns["log_likelihood_function"](x[:, 0], x[:, 1]), MASS_MODE_LNL_259baea8d)
    source = open(DRIVER).read()
    assert 'parser.add_argument("--eos-affine-parameter", default="mass"' in source
    # Every central-only statement sits under a top-level eos_affine_central test.
    tree = ast.parse(source)
    central_ifs = [n for n in tree.body if isinstance(n, ast.If) and ast.dump(n.test) ==
                   ast.dump(ast.parse("eos_affine_central", mode="eval").body)]
    assert len(central_ifs) >= 5


# ------------------------------------------------------------ end to end

SITE = """
import os
if os.environ.get("CIP_FAKE_TWIN"):
    import numpy as np
    if not hasattr(np, "float128"):
        np.float128 = np.longdouble
    np.random.seed(3)
    import sys
    sys.path.insert(0, os.environ["CIP_FAKE_TWIN"])
    from test_eos_affine_central import FakeTwinEOS
    import RIFT.physics.EOSManager as manager
    manager.EOSLALSimulation = lambda name: FakeTwinEOS()
"""


def _ile_file(path):
    rng = np.random.default_rng(7)
    n = 600
    mc = rng.uniform(1.55, 1.70, n)
    eta = 0.25 * (1 - rng.uniform(0, 0.3, n) ** 2)
    M = mc / eta ** 0.6
    m1 = M * (1 + np.sqrt(1 - 4 * eta)) / 2
    l1, l2 = rng.uniform(0, 800, (2, n))
    lt = (8. / 13) * ((1 + 7 * eta - 31 * eta ** 2) * (l1 + l2)
                      + np.sqrt(1 - 4 * eta) * (1 + 9 * eta - 11 * eta ** 2) * (l1 - l2))
    lnL = 20 - 0.5 * ((mc - 1.62) / 0.01) ** 2 - 0.5 * ((lt - 300.) / 250.) ** 2
    z = np.zeros(n)
    np.savetxt(path, np.column_stack([np.arange(n), m1, M - m1, z, z, z, z, z, z, l1, l2, lnL,
                                      np.full(n, .01), np.full(n, 1000.), np.full(n, 100.)]))


def test_driver_end_to_end_central_fake_twin(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(SITE)
    _ile_file(tmp_path / "ile.dat")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([CODE, str(tmp_path)]), CIP_FAKE_TWIN=str(Path(__file__).parent),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(tmp_path / "mpl"))
    result = subprocess.run([sys.executable, DRIVER, "--fname", str(tmp_path / "ile.dat"), "--input-tides",
                             "--using-eos", "lal_FAKETWIN", "--eos-affine-parameter", "central",
                             "--parameter", "mc", "--parameter", "delta_mc", "--parameter-implied", "LambdaTilde",
                             "--mc-range", "[1.55,1.70]", "--fit-method", "quadratic", "--n-max", "100000",
                             "--n-eff", "500", "--n-chunk", "4000", "--n-output-samples", "300",
                             "--fname-output-integral", "result", "--fname-output-samples", "samples", "--no-plots"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert np.isfinite(np.loadtxt(tmp_path / "result.dat"))
    P = lalsimutils.xml_to_ChooseWaveformParams_array(str(tmp_path / "samples.xml.gz"))
    B = np.atleast_2d(np.loadtxt(tmp_path / "samples_eos_branch.dat", dtype=int))
    assert len(B) == len(P) > 0
    eos = FakeTwinEOS()
    for p, (b1, b2, n1, n2) in zip(P, B):
        for m, lam, b, n in ((p.m1 / lal.MSUN_SI, p.lambda1, b1, n1), (p.m2 / lal.MSUN_SI, p.lambda2, b2, n2)):
            assert b in eos.branches_for_m(m) and n == len(eos.branches_for_m(m))
            assert lam == pytest.approx(eos.branch_lambda(m, b), rel=1e-5)   # XML stores float32
    assert (B[:, :2] == 1).any() and (B[:, :2] == 0).any()


# ---------------------------------------------------------------- guards

@pytest.mark.parametrize("kwargs,match", [
    (dict(using_eos=None), "requires --using-eos"),
    (dict(using_eos_branch=0), "--using-eos-branch"),
    (dict(tabular_eos_file="seq.h5"), "ordering coordinate"),
    (dict(using_eos_for_prior=True), "--using-eos-for-prior"),
    (dict(assume_eos_but_primary_bh=True), "--no-matter1"),
    (dict(oracle_reference_sample_file="ref.xml.gz"), "oracle"),
    (dict(sampler_oracle=["RS"]), "oracle"),
])
def test_central_option_guards(kwargs, match):
    args = dict(using_eos="lal_SLy")
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        eos_affine.validate_eos_affine_request("central", **args)
    # mass mode never rejects on these grounds
    eos_affine.validate_eos_affine_request("mass", **args)
    eos_affine.validate_eos_affine_request("central", using_eos="lal_SLy",
                                           assume_eos_but_primary_bh=True, no_matter1=True)


def test_eos_surface_guards():
    class NoBranches:
        mMaxMsun = 2.
        def lambda_from_m(self, m): return 1.
    class NoBranchId(NoBranches):
        mMaxMsun_all = 2.
        def branches_for_m(self, m): return [0]
    class NoMax:
        def branches_for_m(self, m): return [0]
        def lambda_from_m(self, m, branch_id=None): return 1.
    with pytest.raises(TypeError, match="branches_for_m"):
        eos_affine.require_branch_surface(NoBranches())
    with pytest.raises(TypeError, match="branch_id"):
        eos_affine.require_branch_surface(NoBranchId())
    with pytest.raises(TypeError, match="all-branch maximum"):
        eos_affine.require_branch_surface(NoMax())
    class LalsimLike(NoMax):
        mMaxMsun = 2.7   # lalsim: already the all-branch maximum
    assert eos_affine.require_branch_surface(LalsimLike()) == 2.7
    with pytest.raises(ValueError):
        eos_affine.validate_eos_affine_request("enthalpy", using_eos="x")
    assert eos_affine.require_branch_surface(FakeTwinEOS()) == 2.4


@pytest.mark.parametrize("extra,match", [
    (["--using-eos-branch", "0"], "--using-eos-branch"),
    (["--tabular-eos-file", "seq.h5", "--input-eos-index"], "ordering coordinate"),
])
def test_driver_rejects_incompatible_central_runs(tmp_path, extra, match):
    result = subprocess.run([sys.executable, DRIVER, "--fname", "unused.dat", "--using-eos", "lal_SLy",
                             "--eos-affine-parameter", "central", "--no-plots"] + extra,
                            cwd=tmp_path, env=dict(os.environ, PYTHONPATH=CODE, MPLBACKEND="Agg"),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert match in result.stderr


def test_driver_central_blocks_use_the_tested_helpers():
    """The driver wires the branch variates into the sampler, GMM groups and integrand."""
    source = open(DRIVER).read()
    for needle in ("eos_affine.make_central_log_likelihood(",
                   "sampling_coord_names = list(sampling_coord_names) + list(eos_affine.CENTRAL_AUX_NAMES)",
                   "gmm_dict[(k,)] = None",
                   "_eos_branch.dat",
                   "eos_affine.require_branch_surface(my_eos)"):
        assert needle in source, needle
    ast.parse(source)


# ------------------------------------------- nmbseq (needs the NMB branch API)

def _nmb_fixture(tmp_path):
    nmb = pytest.importorskip("test_nmb_sequence_branches",
                              reason="NMB sequence branch API (EOSManager e28fb9340) not present")
    from RIFT.physics import EOSManager
    if not hasattr(EOSManager.EOSSequenceSingleIndex, "branches_for_m"):
        pytest.skip("EOSSequenceSingleIndex has no branch API")
    return nmb._write_nss(tmp_path / "twin.nss.h5", nmb._synthetic_sequence())


def test_nmbseq_central_branch_sum(tmp_path):
    from RIFT.physics import EOSManager
    path = _nmb_fixture(tmp_path)
    eos = EOSManager.EOSSequenceSingleIndex(fname=path, index=1)   # twin_high
    m_all = eos_affine.require_branch_surface(eos)
    assert m_all == pytest.approx(eos.mMaxMsun_all) and m_all > eos.mMaxMsun
    rng = np.random.default_rng(2)
    m = rng.uniform(0.5, 2.3, (300, 2))
    u = rng.uniform(0, 1, (300, 2))
    x_out, log_w, br = _central(eos, m, u, coord_names=("lambda1", "lambda2"))
    for row in range(len(m)):
        ns = [eos.branches_for_m(mi) for mi in m[row]]
        if not all(ns) or max(m[row]) >= m_all:
            assert np.isneginf(x_out[row]).all()
            continue
        assert log_w[row] == pytest.approx(np.log(len(ns[0]) * len(ns[1])))
        for c in range(2):
            assert br[row, c] in ns[c]
            assert x_out[row, c] == eos.lambda_from_m(m[row, c] * lal.MSUN_SI, branch_id=int(br[row, c]))
    assert (log_w > 0).any() and (br == 1).any()


def test_nmbseq_driver_end_to_end_central(tmp_path):
    path = _nmb_fixture(tmp_path)
    (tmp_path / "sitecustomize.py").write_text(
        "import numpy as np\nif not hasattr(np, 'float128'):\n    np.float128 = np.longdouble\nnp.random.seed(3)\n")
    _ile_file(tmp_path / "ile.dat")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([CODE, str(tmp_path)]), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"))
    common = [sys.executable, DRIVER, "--fname", str(tmp_path / "ile.dat"), "--input-tides",
              "--using-eos", "nmbseq:{}:1".format(path), "--parameter", "mc", "--parameter", "delta_mc",
              "--parameter-implied", "LambdaTilde", "--mc-range", "[1.55,1.70]", "--fit-method", "quadratic",
              "--n-max", "100000", "--n-eff", "500", "--n-chunk", "4000", "--n-output-samples", "300", "--no-plots"]
    lnz = {}
    for mode in ("mass", "central"):
        result = subprocess.run(common + ["--eos-affine-parameter", mode, "--fname-output-integral", "res_" + mode,
                                          "--fname-output-samples", "samp_" + mode],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        lnz[mode] = float(np.loadtxt(tmp_path / ("res_" + mode + ".dat")))
    B = np.atleast_2d(np.loadtxt(tmp_path / "samp_central_eos_branch.dat", dtype=int))
    assert (B[:, :2] == 1).any()
    assert not (tmp_path / "samp_mass_eos_branch.dat").exists()
    # The data favour component masses near 1.86 Msun, inside the twin region.
    assert lnz["central"] > lnz["mass"]
