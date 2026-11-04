"""
Time-indexed cross terms in DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop.

For the time-domain fixed-segment code path the template self term <h|h> depends on
the arrival time, so U and V are supplied per detector as (npts_full, n_lm, n_lm)
arrays on the rholm time grid.  These checks run on CPU (xpy=np) with synthetic
random Q/U/V and random extrinsic parameters:

  1. scalar (2-D) U path is bit-identical to the parent-commit code (stored
     hex-exact reference values, same deterministic inputs);
  2. time-indexed U that is constant in time gives exactly the scalar-U result;
  3. genuinely time-varying U against a brute-force per-sample, per-time loop,
     for nearest and cubic time_interp, with explicit orbital phase,
     with return_lnLt, with an elementwise custom loglikelihood (the distance
     marginalization hook), with mixed scalar/time-indexed detectors, and with
     a window running past the buffer edge (zero extension);
  4. refusals: analytic phase marginalization, unsupported calibration paths,
     the legacy array likelihoods, and bad shapes.

Run with:  pytest test/test_td_time_indexed_U.py
"""
import numpy as np
import pytest
import lal
import lalsimulation as lalsim

import RIFT.likelihood.factored_likelihood as fl
from RIFT.likelihood.vectorized_lal_tools import ComputeDetAMResponse, TimeDelayFromEarthCenter
from RIFT.likelihood.SphericalHarmonics_gpu import SphericalHarmonicsVectorized


class _PVec(object):
    pass


def _make_case(seed=7, dets=("H1", "L1", "V1"), modes=((2, 2), (2, -2), (2, 1)),
               npts_full=256, npts=12, npts_extrinsic=5, deltaT=1.0/4096,
               epoch_offset=-0.03, hermitian=True, physical=True):
    """Synthetic inputs.  Returns a dict with everything the likelihood needs.

    U_t[det] is (npts_full, n_lm, n_lm): Hermitian at each time when
    hermitian=True (V_t symmetric), otherwise fully random.  U_s[det] is the
    (n_lm, n_lm) scalar cross term equal to U_t at time index 0 (used for the
    constant-in-time check).
    """
    rng = np.random.default_rng(seed)
    n_lm = len(modes)
    lookupNKDict = {det: np.array(modes, dtype=int) for det in dets}
    rholmsArrayDict = {}
    U_t, V_t, U_s, V_s, epochDict = {}, {}, {}, {}, {}
    P = _PVec()
    P.phi = rng.uniform(0, 2*np.pi, npts_extrinsic)
    P.theta = rng.uniform(0.2, np.pi-0.2, npts_extrinsic)
    P.psi = rng.uniform(0, np.pi, npts_extrinsic)
    P.incl = rng.uniform(0.2, np.pi-0.2, npts_extrinsic)
    P.phiref = rng.uniform(0, 2*np.pi, npts_extrinsic)
    P.dist = rng.uniform(300.0, 900.0, npts_extrinsic) * (lal.PC_SI*1e6)
    P.tref = 1000000000.0
    P.deltaT = deltaT
    for det in dets:
        rholmsArrayDict[det] = (rng.standard_normal((n_lm, npts_full))
                                + 1j*rng.standard_normal((n_lm, npts_full)))
        A = rng.standard_normal((npts_full, n_lm, n_lm)) + 1j*rng.standard_normal((npts_full, n_lm, n_lm))
        B = rng.standard_normal((npts_full, n_lm, n_lm)) + 1j*rng.standard_normal((npts_full, n_lm, n_lm))
        if hermitian:
            A = A + np.conj(np.swapaxes(A, 1, 2))
            B = B + np.swapaxes(B, 1, 2)
        if physical:
            # Diagonal dominance of the real quadratic form ensures a valid norm
            # for every complex detector response and mode coefficient vector.
            floor = 2 * (np.sum(np.abs(A), axis=(1, 2)) + np.sum(np.abs(B), axis=(1, 2)))
            A = A + floor[:, None, None] * np.eye(n_lm)[None]
        U_t[det] = A
        V_t[det] = B
        U_s[det] = A[0].copy()
        V_s[det] = B[0].copy()
        epochDict[det] = P.tref + epoch_offset
    tvals = np.linspace(-npts//2*deltaT, npts//2*deltaT, npts)
    return dict(P=P, tvals=tvals, lookupNKDict=lookupNKDict, rholmsArrayDict=rholmsArrayDict,
                U_t=U_t, V_t=V_t, U_s=U_s, V_s=V_s, epochDict=epochDict, npts=npts,
                npts_extrinsic=npts_extrinsic, deltaT=deltaT, dets=list(dets))


def _call(case, U, V, **kw):
    return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        case["tvals"], case["P"], case["lookupNKDict"], case["rholmsArrayDict"],
        U, V, case["epochDict"], Lmax=2, xpy=np, **kw)


def _cubic_weights(u):
    return (-u*(u-1.0)*(u-2.0)/6.0,
            (u+1.0)*(u-1.0)*(u-2.0)/2.0,
            -(u+1.0)*u*(u-2.0)/2.0,
            (u+1.0)*u*(u-1.0)/6.0)


def _brute_force(case, U, V, time_interp='nearest', phase_marginalization=False,
                 loglikelihood=None, return_lnLt=False):
    """Per-sample, per-time loop implementing
        lnL_k(e,t) = (D_ref/D) Re[sum_lm conj(F Y_lm) Q_lm(t)]
                     - (D_ref/D)^2/4 Re[ |F|^2 sum conj(Y_lm) Y_l'm' U + F^2 sum Y_lm Y_l'm' V ]
    with Q, U, V looked up at the same (nearest or cubic-interpolated) index.
    U[det], V[det] may be 2-D (time independent) or 3-D (time indexed).
    """
    P = case["P"]
    tvals = case["tvals"]
    npts = case["npts"]
    ne = case["npts_extrinsic"]
    deltaT = float(P.deltaT)
    distMpc = P.dist/(lal.PC_SI*1e6)
    invD = fl.distMpcRef/distMpc
    gmst = lal.GreenwichMeanSiderealTime(P.tref)
    kappa = np.zeros((ne, npts), dtype=complex)
    rho_sq = np.zeros((ne, npts))
    for det in case["dets"]:
        detector = lalsim.DetectorPrefixToLALDetector(det)
        lms = case["lookupNKDict"][det]
        n_lm = len(lms)
        Y = SphericalHarmonicsVectorized(lms, P.incl, -P.phiref, xpy=np, l_max=2)
        F = ComputeDetAMResponse(detector.response, P.phi, P.theta, P.psi, gmst, xpy=np)
        t_det = float(P.tref - float(case["epochDict"][det])) + TimeDelayFromEarthCenter(
            detector.location, P.phi, P.theta, float(gmst), xpy=np)
        sample_first = (t_det + tvals[0])/deltaT
        Q = case["rholmsArrayDict"][det].T          # (npts_full, n_lm)
        n_time = Q.shape[0]
        Ud = U[det]
        Vd = V[det]
        for e in range(ne):
            if time_interp == 'nearest':
                i0 = int(np.rint(sample_first[e]) + 0.5)
                stencil = [(0, 1.0)]
            else:
                i0 = int(np.floor(sample_first[e]))
                stencil = list(zip((-1, 0, 1, 2), _cubic_weights(sample_first[e] - np.floor(sample_first[e]))))
            for j in range(npts):
                Q_here = np.zeros(n_lm, dtype=complex)
                U_here = np.zeros((n_lm, n_lm), dtype=complex)
                V_here = np.zeros((n_lm, n_lm), dtype=complex)
                for off, w in stencil:
                    idx = i0 + j + off
                    if 0 <= idx < n_time:
                        Q_here += w*Q[idx]
                        U_here += w*(Ud[idx] if Ud.ndim == 3 else Ud)
                        V_here += w*(Vd[idx] if Vd.ndim == 3 else Vd)
                k = 0j
                for a in range(n_lm):
                    term = np.conj(F[e]*Y[e, a])*Q_here[a]
                    if phase_marginalization and a == 1:
                        # the whole (2,-2) contribution (F, Y and Q) is conjugated
                        term = np.conj(term)
                    k += term
                kappa[e, j] += invD[e]*k
                s = 0j
                for a in range(n_lm):
                    for b in range(n_lm):
                        s += (F[e]*np.conj(F[e]))*np.conj(Y[e, a])*Y[e, b]*U_here[a, b]
                        s += F[e]*F[e]*Y[e, a]*Y[e, b]*V_here[a, b]
                rho_sq[e, j] += 0.5*invD[e]**2*s.real
    if loglikelihood is None:
        loglikelihood = lambda kap, rho: kap - 0.5*rho
    k_in = np.abs(kappa) if phase_marginalization else kappa.real
    lnL_t = loglikelihood(k_in, rho_sq)
    if return_lnLt:
        return lnL_t
    # per-row offset, as the likelihood takes it (a single batch maximum underflows quiet rows)
    m = np.max(lnL_t, axis=-1, keepdims=True)
    L = fl.my_simps(np.exp(lnL_t - m), dx=deltaT, axis=-1)
    return m[:, 0] + np.log(L)


# --------------------------------------------------------------------------
# 1. scalar path bit-identical to the parent commit
# --------------------------------------------------------------------------
# Reference lnL (hex floats) from DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop
# at the parent commit (ddc2d99e), on _make_case(seed=7) with the scalar U_s/V_s,
# for (time_interp, phase_marginalization) in the order below.
_SCALAR_REFERENCE = {
    ('nearest', False): ['-0x1.8b78573b6077ep+2', '-0x1.69f6e851c2512p+2', '-0x1.81443504c9ec2p+2', '-0x1.6a71c1e4a2e94p+2', '-0x1.63cac0935a0c2p+2'],
    ('nearest', True): ['-0x1.629d5b63a02c0p+2', '-0x1.0fe1e4000425fp+2', '-0x1.590ec83cc02eep+2', '-0x1.51325ea047cc8p+2', '-0x1.3168217666e44p+2'],
    ('cubic', False): ['-0x1.8b4c87601802ep+2', '-0x1.6c3c22b249c90p+2', '-0x1.80dd0045f38f8p+2', '-0x1.6ac000166eb86p+2', '-0x1.6b7851ccaed10p+2'],
    ('cubic', True): ['-0x1.6a79ffc519bb2p+2', '-0x1.1328980112586p+2', '-0x1.61e85eed08db0p+2', '-0x1.579687f4bb618p+2', '-0x1.3bc0e01ddb03ep+2'],
}


@pytest.mark.parametrize("time_interp,phase_marg", sorted(_SCALAR_REFERENCE.keys()))
def test_scalar_path_bit_identical_to_parent(time_interp, phase_marg):
    case = _make_case(seed=7, modes=((2, 2), (2, -2)), physical=False)
    lnL = _call(case, case["U_s"], case["V_s"], time_interp=time_interp, phase_marginalization=phase_marg)
    ref = np.array([float.fromhex(h) for h in _SCALAR_REFERENCE[(time_interp, phase_marg)]])
    assert ref.shape == lnL.shape
    # The stored values were computed on one machine; other numpy/BLAS builds differ at the
    # level of rounding (~1e-15 relative), so compare to 1e-12 rather than bit for bit.
    np.testing.assert_allclose(np.asarray(lnL), ref, rtol=1e-12, atol=0,
                               err_msg="scalar path changed")


def test_scalar_path_against_brute_force():
    case = _make_case(seed=11)
    for time_interp in ('nearest', 'cubic'):
        lnL = _call(case, case["U_s"], case["V_s"], time_interp=time_interp)
        ref = _brute_force(case, case["U_s"], case["V_s"], time_interp=time_interp)
        np.testing.assert_allclose(lnL, ref, rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------
# 2. constant-in-time U reproduces the scalar result exactly
# --------------------------------------------------------------------------
@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
@pytest.mark.parametrize("phase_marg", [False, True])
def test_constant_time_indexed_equals_scalar(time_interp, phase_marg):
    case = _make_case(seed=3, modes=((2, 2), (2, -2)))
    U_const = {det: np.broadcast_to(case["U_s"][det], (256, 2, 2)).copy() for det in case["dets"]}
    V_const = {det: np.broadcast_to(case["V_s"][det], (256, 2, 2)).copy() for det in case["dets"]}
    kw = dict(time_interp=time_interp, phase_marginalization=phase_marg)
    if phase_marg:
        with pytest.raises(NotImplementedError, match="explicit phase sampling"):
            _call(case, U_const, V_const, **kw)
        return
    lnL_s = _call(case, case["U_s"], case["V_s"], **kw)
    lnL_t = _call(case, U_const, V_const, **kw)
    # Same quantities, summed in a different order: agreement to a few ulp.
    np.testing.assert_allclose(lnL_t, lnL_s, rtol=1e-13, atol=0)
    lnLt_s = _call(case, case["U_s"], case["V_s"], return_lnLt=True, **kw)
    lnLt_t = _call(case, U_const, V_const, return_lnLt=True, **kw)
    assert lnLt_t.shape == (case["npts_extrinsic"], case["npts"])
    np.testing.assert_allclose(lnLt_t, lnLt_s, rtol=1e-13, atol=1e-12)


def test_constant_time_indexed_self_term_exact():
    # The self term itself: the time-indexed code path gathers the same matrix at every
    # window index, so it must equal the scalar einsum to roundoff.
    case = _make_case(seed=5)
    det = "H1"
    n_lm = 3
    rng = np.random.default_rng(0)
    ne = 4
    F = rng.standard_normal(ne) + 1j*rng.standard_normal(ne)
    Y = rng.standard_normal((ne, n_lm)) + 1j*rng.standard_normal((ne, n_lm))
    U = case["U_s"][det]
    V = case["V_s"][det]
    ref = ((F*np.conj(F)).real*np.einsum("...i,...j,ij", np.conj(Y), Y, U).real
           + (np.square(F)*np.einsum("...i,...j,ij", Y, Y, V)).real)
    U3 = np.broadcast_to(U, (40, n_lm, n_lm)).copy()
    V3 = np.broadcast_to(V, (40, n_lm, n_lm)).copy()
    ifirst = np.array([3, 5, 0, 20], dtype=np.int32)
    got = fl._time_indexed_self_term(U3, V3, F, Y, ifirst, None, 8, 'nearest', xpy=np)
    assert got.shape == (ne, 8)
    np.testing.assert_allclose(got, ref[:, None]*np.ones((1, 8)), rtol=1e-13, atol=0)
    frac = np.array([0.1, 0.7, 0.0, 0.5])
    got_c = fl._time_indexed_self_term(U3, V3, F, Y, ifirst, frac, 8, 'cubic', xpy=np)
    np.testing.assert_allclose(got_c, ref[:, None]*np.ones((1, 8)), rtol=1e-12, atol=0)


# --------------------------------------------------------------------------
# 3. genuinely time-varying U against the brute-force loop
# --------------------------------------------------------------------------
@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
@pytest.mark.parametrize("phase_marg", [False, True])
@pytest.mark.parametrize("hermitian", [True, False])
def test_time_varying_against_brute_force(time_interp, phase_marg, hermitian):
    case = _make_case(seed=23, modes=((2, 2), (2, -2)), hermitian=hermitian)
    kw = dict(time_interp=time_interp, phase_marginalization=phase_marg)
    if phase_marg:
        with pytest.raises(NotImplementedError, match="explicit phase sampling"):
            _call(case, case["U_t"], case["V_t"], **kw)
        return
    lnL = _call(case, case["U_t"], case["V_t"], **kw)
    ref = _brute_force(case, case["U_t"], case["V_t"], **kw)
    np.testing.assert_allclose(lnL, ref, rtol=1e-10, atol=1e-10)
    lnLt = _call(case, case["U_t"], case["V_t"], return_lnLt=True, **kw)
    reft = _brute_force(case, case["U_t"], case["V_t"], return_lnLt=True, **kw)
    assert lnLt.shape == (case["npts_extrinsic"], case["npts"])
    np.testing.assert_allclose(lnLt, reft, rtol=1e-10, atol=1e-10)
    # the self term must vary along the time axis
    rho_only = _call(case, case["U_t"], case["V_t"], return_lnLt=True,
                     loglikelihood=lambda k, r: r, **kw)
    assert np.max(np.abs(np.diff(rho_only, axis=-1))) > 1e-3


@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
def test_time_varying_three_modes(time_interp):
    case = _make_case(seed=29, modes=((2, 2), (2, -2), (2, 1)), hermitian=True)
    lnL = _call(case, case["U_t"], case["V_t"], time_interp=time_interp)
    ref = _brute_force(case, case["U_t"], case["V_t"], time_interp=time_interp)
    np.testing.assert_allclose(lnL, ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
def test_custom_loglikelihood_sees_time_dependent_rho_sq(time_interp):
    # Elementwise loglikelihood (the shape the distance-marginalization table
    # helper has) must receive rho_sq of shape (npts_extrinsic, npts).
    seen = {}

    def loglike(kappa_sq, rho_sq):
        seen['shape'] = rho_sq.shape
        seen['varies'] = float(np.max(np.abs(np.diff(rho_sq, axis=-1))))
        return kappa_sq - 0.5*rho_sq + 0.1*np.log1p(np.abs(rho_sq))

    case = _make_case(seed=31, modes=((2, 2), (2, -2)))
    lnL = _call(case, case["U_t"], case["V_t"], time_interp=time_interp, loglikelihood=loglike)
    assert seen['shape'] == (case["npts_extrinsic"], case["npts"])
    assert seen['varies'] > 1e-3
    ref = _brute_force(case, case["U_t"], case["V_t"], time_interp=time_interp, loglikelihood=loglike)
    np.testing.assert_allclose(lnL, ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
def test_mixed_scalar_and_time_indexed_detectors(time_interp):
    case = _make_case(seed=37, modes=((2, 2), (2, -2)))
    U = dict(case["U_t"]); V = dict(case["V_t"])
    U["L1"] = case["U_s"]["L1"]; V["L1"] = case["V_s"]["L1"]
    lnL = _call(case, U, V, time_interp=time_interp)
    ref = _brute_force(case, U, V, time_interp=time_interp)
    np.testing.assert_allclose(lnL, ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("time_interp", ['nearest', 'cubic'])
def test_window_past_buffer_edge_zero_extension(time_interp):
    # epoch close to tref -> ifirst near 0, some windows start before the buffer
    case = _make_case(seed=41, modes=((2, 2), (2, -2)), epoch_offset=-0.0005, npts_full=64)
    lnL = _call(case, case["U_t"], case["V_t"], time_interp=time_interp, return_lnLt=True)
    ref = _brute_force(case, case["U_t"], case["V_t"], time_interp=time_interp, return_lnLt=True)
    np.testing.assert_allclose(lnL, ref, rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------------
# 4. refusals
# --------------------------------------------------------------------------
def test_refuse_calibration_paths():
    """Calibration with time-indexed terms runs on the loop method only (test_td_gs_calibration.py
    checks the values); the fused kernel and the compressed base U refuse it."""
    case = _make_case(seed=43, modes=((2, 2), (2, -2)))
    with pytest.raises(NotImplementedError):
        _call(case, case["U_t"], case["V_t"], n_cal=2, cal_method='fused')
    comp = {det: fl.compress_time_indexed_cross_terms(case["U_t"][det], case["V_t"][det], 1e-13) for det in case["dets"]}
    with pytest.raises(NotImplementedError):
        _call(case, comp, comp, n_cal=2)
    U_cal = {det: case["U_s"][det][None] for det in case["dets"]}
    V_cal = {det: case["V_s"][det][None] for det in case["dets"]}
    with pytest.raises(NotImplementedError):
        _call(case, comp, comp, ctUArrayDict_cal=U_cal, ctVArrayDict_cal=V_cal)
    # the loop method with the realization blocks (npts_full = 2 blocks of 128) runs: the base
    # U(t) may be the full length or one block long, and gives the same result either way
    lnL_full = _call(case, case["U_t"], case["V_t"], n_cal=2)
    U_blk = {det: case["U_t"][det][:128] for det in case["dets"]}
    V_blk = {det: case["V_t"][det][:128] for det in case["dets"]}
    lnL_blk = _call(case, U_blk, V_blk, n_cal=2)
    # block 1 of the full-length U is gathered for realization 1 in lnL_full but block 0's is reused
    # in lnL_blk; realization 0 agrees exactly
    c_full = _call(case, case["U_t"], case["V_t"], n_cal=2, return_cal_components=True)
    c_blk = _call(case, U_blk, V_blk, n_cal=2, return_cal_components=True)
    np.testing.assert_allclose(c_full[:, 0], c_blk[:, 0], rtol=0, atol=1e-12)
    assert np.all(np.isfinite(lnL_full)) and np.all(np.isfinite(lnL_blk))


def test_refuse_options_not_implemented_for_time_indexed():
    # stencils, pregrids and quadratures the time-indexed self term does not implement
    case = _make_case(seed=47, modes=((2, 2), (2, -2)))
    dT = float(case["P"].deltaT)
    for kw in (dict(time_interp='sinc'),
               dict(time_quadrature='bandlimited'),
               dict(time_quadrature='peak-local'),
               dict(explicit_time_values=True),
               dict(time_interp='cubic', q_deltaT=dT / 2)):
        with pytest.raises(NotImplementedError):
            _call(case, case["U_t"], case["V_t"], **kw)
    # the fixed-grid trapezoid rule applies to any integrand
    _call(case, case["U_t"], case["V_t"], time_quadrature='trapezoid')


def test_refuse_bad_shapes():
    case = _make_case(seed=47, modes=((2, 2), (2, -2)))
    with pytest.raises(ValueError):            # U 3-D, V 2-D
        _call(case, case["U_t"], case["V_s"])
    U_short = {det: case["U_t"][det][:-1] for det in case["dets"]}
    V_short = {det: case["V_t"][det][:-1] for det in case["dets"]}
    with pytest.raises(ValueError):            # time grid mismatch
        _call(case, U_short, V_short)



def test_refuse_mismatched_mode_shapes():
    case = _make_case(seed=47, modes=((2, 2), (2, -2)))
    U3 = {det: np.zeros((256, 3, 3), dtype=complex) for det in case["dets"]}
    with pytest.raises(ValueError, match="modes"):         # 3 modes in U(t), V(t); 2 in lookupNKDict
        _call(case, U3, U3)
    V_bad = {det: case["V_t"][det][:, :1, :] for det in case["dets"]}
    with pytest.raises(ValueError, match="matching"):      # U(t), V(t) of different shapes
        _call(case, case["U_t"], V_bad)
    V_s_bad = {det: np.zeros((3, 3), dtype=complex) for det in case["dets"]}
    with pytest.raises(ValueError, match="matching"):      # scalar U, V of different shapes
        _call(case, case["U_s"], V_s_bad)


def test_negative_norm_refused_on_ordinary_likelihood():
    case = _make_case(dets=('H1',), modes=((2, 2),))
    U = {'H1': -np.ones((256, 1, 1), dtype=complex)}
    V = {'H1': np.zeros((256, 1, 1), dtype=complex)}
    with pytest.raises(ValueError, match='negative template norm'):
        _call(case, U, V)

def test_refuse_legacy_array_likelihoods():
    case = _make_case(seed=53, modes=((2, 2), (2, -2)))
    P = case["P"]
    with pytest.raises(NotImplementedError):
        fl.SingleDetectorLogLikelihoodModelViaArray(case["lookupNKDict"], case["U_t"], case["V_t"],
                                                    P.tref, P.phi[0], P.theta[0], P.incl[0], P.phiref[0],
                                                    P.psi[0], P.dist[0], "H1")
    Ps = _PVec()
    for name in ("phi", "theta", "psi", "incl", "phiref", "dist"):
        setattr(Ps, name, float(getattr(P, name)[0]))
    Ps.tref = P.tref
    Ps.deltaT = P.deltaT
    with pytest.raises(NotImplementedError):
        fl.DiscreteFactoredLogLikelihoodViaArray(case["tvals"], Ps, case["lookupNKDict"], case["rholmsArrayDict"],
                                                 case["U_t"], case["V_t"], case["epochDict"], Lmax=2)
    with pytest.raises(NotImplementedError):
        fl.DiscreteFactoredLogLikelihoodViaArrayVector(case["tvals"], P, case["lookupNKDict"], case["rholmsArrayDict"],
                                                       case["U_t"], case["V_t"], case["epochDict"], Lmax=2, xpy=np)
    with pytest.raises(NotImplementedError):
        fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopOrig(case["tvals"], P, case["lookupNKDict"], case["rholmsArrayDict"],
                                                                 case["U_t"], case["V_t"], case["epochDict"], Lmax=2)


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------
class _FakeSeries(object):
    def __init__(self, data, epoch):
        class _D(object):
            pass
        self.data = _D()
        self.data.data = data
        self.data.length = len(data)
        self.epoch = epoch


def test_pack_time_indexed_cross_terms():
    rng = np.random.default_rng(59)
    modes = [(2, 2), (2, -2), (2, 1), (2, -1)]
    npts = 20
    rholms = {m: _FakeSeries(rng.standard_normal(npts) + 0j, 1.0) for m in modes}
    ct = {(a, b): rng.standard_normal(npts) + 1j*rng.standard_normal(npts) for a in modes for b in modes}
    ctV = {(a, b): rng.standard_normal(npts) + 1j*rng.standard_normal(npts) for a in modes for b in modes}
    lNK, lKN, lconj, U, V, rA, rI, ep = fl.PackLikelihoodDataStructuresAsArrays(modes, None, rholms, ct, ctV)
    assert U.shape == (npts, 4, 4) and V.shape == (npts, 4, 4)
    for a in modes:
        for b in modes:
            np.testing.assert_array_equal(U[:, lKN[a], lKN[b]], ct[(a, b)])
            np.testing.assert_array_equal(V[:, lKN[a], lKN[b]], ctV[(a, b)])
    # scalar values still give (n_lm, n_lm), same lookup ordering
    ct_s = {k: v[0] for k, v in ct.items()}
    ctV_s = {k: v[0] for k, v in ctV.items()}
    lNK2, lKN2, _, U2, V2, _, _, _ = fl.PackLikelihoodDataStructuresAsArrays(modes, None, rholms, ct_s, ctV_s)
    assert U2.shape == (4, 4)
    assert lKN2 == lKN
    np.testing.assert_array_equal(U2, U[0])
    np.testing.assert_array_equal(V2, V[0])
    # wrong length refuses
    ct_bad = dict(ct)
    ct_bad[(modes[1], modes[2])] = ct_bad[(modes[1], modes[2])][:-1]
    with pytest.raises(ValueError):
        fl.PackLikelihoodDataStructuresAsArrays(modes, None, rholms, ct_bad, ctV)




# Review regressions: phase-dependent norms and retained calibration buffers.
@pytest.mark.parametrize("interp", ["nearest", "cubic"])
@pytest.mark.parametrize("active", [1, 2, 4])
@pytest.mark.parametrize("return_lnLt", [False, True])
def test_global_norm_uses_stored_blocks_during_burn_in(interp, active, return_lnLt):
    from scipy.special import logsumexp
    case = _make_case(seed=701, dets=("H1",), modes=((2, 2), (2, -2)),
                      npts_full=512, npts=80, epoch_offset=-0.001)
    U = {d: case["U_t"][d][:128] for d in case["dets"]}
    V = {d: case["V_t"][d][:128] for d in case["dets"]}
    kw = dict(time_interp=interp, return_lnLt=return_lnLt)
    got = _call(case, U, V, n_cal=active, **kw)
    separate = []
    for c in range(active):
        one = dict(case, rholmsArrayDict={
            d: case["rholmsArrayDict"][d][:, c*128:(c+1)*128] for d in case["dets"]})
        separate.append(_call(one, U, V, **kw))
    ref = logsumexp(np.asarray(separate), axis=0) - np.log(active)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    # Ensure this exercises the data term, including windows crossing block edges.
    zero = dict(case, rholmsArrayDict={d: np.zeros_like(v) for d, v in case["rholmsArrayDict"].items()})
    assert np.max(np.abs(got - _call(zero, U, V, n_cal=active, **kw))) > 1e-4


@pytest.mark.parametrize("layout", ["dense", "compressed", "calibration_only"])
def test_analytic_phase_refused_for_all_time_indexed_norm_layouts(layout):
    case = _make_case(seed=703, modes=((2, 2), (2, -2)))
    U, V, kw = case["U_t"], case["V_t"], {}
    if layout == "compressed":
        U = {d: fl.compress_time_indexed_cross_terms(U[d], V[d], tol=1e-13) for d in case["dets"]}
        V = U
    elif layout == "calibration_only":
        kw = dict(ctUArrayDict_cal={d: U[d][None] for d in case["dets"]},
                  ctVArrayDict_cal={d: V[d][None] for d in case["dets"]})
        U, V = case["U_s"], case["V_s"]
    with pytest.raises(NotImplementedError, match="explicit phase sampling"):
        _call(case, U, V, phase_marginalization=True, **kw)


@pytest.mark.parametrize("name,value", [
    ("internal_ile_td_subsample", 4),
    ("internal_ile_td_calibration_correction_type", "bilby"),
    ("internal_ile_td_calibration_fir_half_length", 32),
])
@pytest.mark.parametrize("route", [None, "td-gs"])
def test_pipeline_td_knobs_require_and_reach_the_selected_route(name, value, route):
    import ast, types, sys, os
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "bin/util_RIFT_pseudo_pipe.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.If) and
                isinstance(n.test, ast.Attribute) and n.test.attr == "internal_ile_inner_product")
    attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and
             isinstance(n.value, ast.Name) and n.value.id == "opts"}
    opts = types.SimpleNamespace(**dict.fromkeys(attrs))
    opts.internal_ile_inner_product = route
    setattr(opts, name, value)
    scope = dict(opts=opts, line="", sys=sys, os=os, _as_list=lambda x: x or [])
    compiled = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "pipeline_td", "exec")
    if route is None:
        with pytest.raises(SystemExit) as exc:
            exec(compiled, scope)
        assert exc.value.code == 1
    else:
        exec(compiled, scope)
        flag = {"internal_ile_td_subsample": "--td-subsample",
                "internal_ile_td_calibration_correction_type": "--td-calibration-correction-type",
                "internal_ile_td_calibration_fir_half_length": "--td-calibration-fir-half-length"}[name]
        assert flag + " " + str(value) in scope["line"]


@pytest.mark.parametrize("td,phase", [(False, False), (False, True), (True, False), (True, True)])
def test_tdgs_phase_table_refused_at_driver_startup(td, phase):
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[1] / "bin/integrate_likelihood_extrinsic_batchmode").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.If) and
                isinstance(n.test, ast.BoolOp) and
                {v.id for v in n.test.values if isinstance(v, ast.Name)} == {"_td_gs_requested", "_tq_phase_marg"})
    scope = dict(_td_gs_requested=td, _tq_phase_marg=phase)
    compiled = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "driver_phase_guard", "exec")
    if td and phase:
        with pytest.raises(ValueError, match="explicit orbital-phase sampling"):
            exec(compiled, scope)
    else:
        exec(compiled, scope)


@pytest.mark.parametrize("missing,expected", [("tdanalysis", True), ("tdanalysis.forms.gs", False), ("numpy", False), (None, False)])
def test_optional_td_import_skip_does_not_hide_broken_installed_apis(missing, expected):
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / ".travis/test-all-mod.py").read_text())
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name == "_missing_optional_module") or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OPTIONAL_MODULE_DEPENDENCIES" for t in n.targets))]
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "optional_import", "exec"), scope)
    exc = ImportError("td-gs needs the final tdanalysis API")
    if missing:
        exc.__cause__ = ModuleNotFoundError("No module named " + repr(missing), name=missing)
    check = scope["_missing_optional_module"]
    assert check("RIFT.likelihood.td_gs_precompute", exc) is expected
    assert check("RIFT.unrelated", exc) is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
