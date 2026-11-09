"""Branch-aware accessors on NuclearMatter-Backend sequence EOS objects.

The primary-branch (v1) results are compared bitwise against reference
implementations copied from the pre-change reader (rift_O4c 259baea8d).
"""

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
lal = pytest.importorskip("lal")

from RIFT.physics import EOSManager
from RIFT.physics.lalsim_eos_compat import evaluate_eos_lambda

FIELDS = ["hc", "M", "R", "Lambda", "M_b", "I", "rho_c", "cs2_c", "stable"]


# ---- reference implementations copied from 259baea8d -----------------------

def _ref_stable_branches(M, R, Lam, stable):
    M = np.asarray(M); R = np.asarray(R); Lam = np.asarray(Lam)
    valid = (np.isfinite(M) & np.isfinite(R) & np.isfinite(Lam)
             & (M > 0) & (np.asarray(stable) > 0.5))
    branches = []
    start = None
    for k in range(len(M)):
        continues = (valid[k] and start is not None
                     and k > 0 and valid[k - 1] and M[k] > M[k - 1])
        if valid[k] and start is None:
            start = k
        elif valid[k] and not continues:
            if k - start >= 2:
                branches.append((M[start:k], R[start:k], Lam[start:k]))
            start = k
        elif not valid[k] and start is not None:
            if k - start >= 2:
                branches.append((M[start:k], R[start:k], Lam[start:k]))
            start = None
    if start is not None and len(M) - start >= 2:
        branches.append((M[start:], R[start:], Lam[start:]))
    return branches


def _ref_interp(m, M, val):
    indx_sort = np.argsort(M)
    return np.exp(np.interp(m, M[indx_sort], np.log(val[indx_sort])))


# ---- synthetic sequences ---------------------------------------------------

def _segment(m0, m1, n):
    return list(np.linspace(m0, m1, n))


def _eos_rows(mass_segments, n_pts):
    """Build one EOS from (masses, stable) segments, padded to n_pts."""
    M = []; stable = []
    for masses, flag in mass_segments:
        M += masses; stable += [flag] * len(masses)
    assert len(M) <= n_pts
    pad = n_pts - len(M)
    M += list(np.linspace(M[-1] - 0.01, M[-1] - 0.3, pad)); stable += [0.0] * pad
    M = np.array(M); stable = np.array(stable)
    hc = np.linspace(0.05, 0.9, n_pts)
    R = 14.0 - 4.0 * hc
    Lam = 4000.0 * np.exp(-9.0 * hc)
    rows = np.zeros((n_pts, len(FIELDS)))
    rows[:, FIELDS.index("hc")] = hc
    rows[:, FIELDS.index("M")] = M
    rows[:, FIELDS.index("R")] = R
    rows[:, FIELDS.index("Lambda")] = Lam
    rows[:, FIELDS.index("M_b")] = 1.1 * M
    rows[:, FIELDS.index("I")] = 1e45
    rows[:, FIELDS.index("rho_c")] = 0.2 + hc
    rows[:, FIELDS.index("cs2_c")] = 0.3
    rows[:, FIELDS.index("stable")] = stable
    return rows


N_PTS = 48


def _synthetic_sequence():
    single = _eos_rows([(_segment(0.4, 2.2, 30), 1.0),
                        (_segment(2.18, 2.0, 5), 0.0)], N_PTS)
    # rise to 2.0, dip ~9% to 1.82, rise past the first maximum to 2.25
    twin_high = _eos_rows([(_segment(0.4, 2.0, 18), 1.0),
                           (_segment(1.95, 1.82, 5), 0.0),
                           (_segment(1.85, 2.25, 12), 1.0),
                           (_segment(2.2, 2.0, 4), 0.0)], N_PTS)
    # third family stays below the first maximum
    twin_low = _eos_rows([(_segment(0.4, 2.0, 18), 1.0),
                          (_segment(1.95, 1.82, 5), 0.0),
                          (_segment(1.84, 1.95, 10), 1.0),
                          (_segment(1.9, 1.7, 4), 0.0)], N_PTS)
    three = _eos_rows([(_segment(0.4, 2.0, 14), 1.0),
                       (_segment(1.95, 1.80, 4), 0.0),
                       (_segment(1.82, 1.95, 8), 1.0),
                       (_segment(1.9, 1.85, 3), 0.0),
                       (_segment(1.88, 2.1, 8), 1.0),
                       (_segment(2.05, 1.9, 3), 0.0)], N_PTS)
    nan_rows = _eos_rows([(_segment(0.4, 2.1, 30), 1.0),
                          (_segment(2.05, 1.9, 6), 0.0)], N_PTS)
    nan_rows[0, FIELDS.index("M")] = np.nan          # leading NaN
    nan_rows[10, FIELDS.index("Lambda")] = np.nan    # NaN inside rising run
    nan_rows[32, FIELDS.index("R")] = np.nan         # NaN in unstable tail
    nan_rows[-1, :] = np.nan                         # trailing NaN row
    return np.stack([single, twin_high, twin_low, three, nan_rows])


EXPECTED_BRANCH_COUNTS = [1, 2, 2, 3, 2]


def _write_nss(path, seq):
    with h5py.File(path, "w") as f:
        f.attrs["representation"] = "tabular_hc/1"
        f.attrs["schema_version"] = "nmbackend.nss/1"
        f.attrs["fields"] = json.dumps(FIELDS)
        f.attrs["units"] = json.dumps({"M": "Msun", "R": "km"})
        f.create_dataset("sequence", data=seq)
    return str(path)


def _write_pca(path, seq):
    """Exact (full-rank) PCA encoding of the M, R, logLambda curves."""
    M = seq[:, :, FIELDS.index("M")]
    R = seq[:, :, FIELDS.index("R")]
    L = np.log(seq[:, :, FIELDS.index("Lambda")])
    curves = np.stack([M, R, L], axis=1)
    curves = np.where(np.isfinite(curves), curves, 1.0)
    n_eos, _, n_pts = curves.shape
    mean = curves.mean(axis=0)
    comps = np.zeros((3, n_eos, n_pts)); coeffs = np.zeros((n_eos, 3, n_eos))
    for c in range(3):
        _, _, vt = np.linalg.svd(curves[:, c] - mean[c], full_matrices=False)
        comps[c] = vt[:n_eos]
        coeffs[:, c] = (curves[:, c] - mean[c]) @ vt[:n_eos].T
    with h5py.File(path, "w") as f:
        f.attrs["representation"] = "pca_hc/1"
        f.attrs["schema_version"] = "nmbackend.pca/1"
        f.attrs["channels"] = json.dumps(["M", "R", "logLambda"])
        f.create_dataset("mean", data=mean)
        f.create_dataset("components", data=comps)
        f.create_dataset("coeffs", data=coeffs)
    return str(path)


@pytest.fixture
def seq_array():
    return _synthetic_sequence()


@pytest.fixture
def nss_path(tmp_path, seq_array):
    return _write_nss(tmp_path / "synthetic.nss.h5", seq_array)


def _ref_branches(seq_array, k):
    s = seq_array[k]
    return _ref_stable_branches(s[:, 1], s[:, 2], s[:, 3], s[:, 8])


# ---- tests -----------------------------------------------------------------

def test_vectorized_splitter_matches_loop_reference():
    rng = np.random.default_rng(42)
    n_eos, n_pts = 300, 40
    M = np.cumsum(rng.normal(0.02, 0.05, (n_eos, n_pts)), axis=1) + 1.0
    M[rng.random(M.shape) < 0.03] = np.nan
    M[rng.random(M.shape) < 0.02] = -1.0
    R = np.where(rng.random(M.shape) < 0.02, np.inf, 12.0)
    Lam = np.where(rng.random(M.shape) < 0.02, np.nan, 300.0)
    stable = (rng.random(M.shape) < 0.85).astype(float)
    M[:, 5] = M[:, 4]                                   # mass plateau
    e, a, b = EOSManager.EOSSequenceNMB._stable_branch_bounds(M, R, Lam, stable)
    for k in range(n_eos):
        ref = _ref_stable_branches(M[k], R[k], Lam[k], stable[k])
        mine = [(M[k, s:t], R[k, s:t], Lam[k, s:t])
                for s, t in zip(a[e == k], b[e == k])]
        assert len(ref) == len(mine)
        for (m0, r0, l0), (m1, r1, l1) in zip(ref, mine):
            assert np.array_equal(m0, m1)
            assert np.array_equal(r0, r1)
            assert np.array_equal(l0, l1)


def test_branch_counts_and_records(nss_path, seq_array):
    seq = EOSManager.EOSSequenceFromFile(fname=nss_path, load_ns=True, no_sort=True)
    assert isinstance(seq, EOSManager.EOSSequenceNMB)
    assert list(seq.stable_branch_counts) == EXPECTED_BRANCH_COUNTS
    for k in range(len(EXPECTED_BRANCH_COUNTS)):
        ref = _ref_branches(seq_array, k)
        assert seq.n_branches_of_indx(k) == len(ref) == seq.stable_branch_counts[k]
        recs = seq.branches_of_indx(k)
        assert recs[0] is seq.eos_ns_tov["eos_{}".format(k)]
        for rec, (M, R, Lam) in zip(recs, ref):
            assert np.array_equal(rec["M"], M)
            assert np.array_equal(rec["R"], R)
            assert np.array_equal(rec["Lambda"], Lam)


def test_primary_default_is_bitwise_unchanged(nss_path, seq_array):
    seq = EOSManager.EOSSequenceFromFile(fname=nss_path, load_ns=True, no_sort=True)
    m = np.linspace(0.2, 2.6, 97)          # includes clamped extrapolation
    for k in range(len(EXPECTED_BRANCH_COUNTS)):
        M, R, Lam = _ref_branches(seq_array, k)[0]
        ref_L = _ref_interp(m, M, Lam)
        ref_R = _ref_interp(m, M, R)
        for kw in ({}, {"branch_id": None}, {"branch_id": "primary"}):
            assert np.array_equal(seq.lambda_of_m_indx(m, k, **kw), ref_L)
            assert np.array_equal(seq.R_of_m_indx(m, k, **kw), ref_R)
        assert seq.m_max_of_indx(k) == np.max(M)
        assert seq.m_max_of_indx(k, branch="primary") == np.max(M)
        assert seq.m_min_of_indx(k) == np.min(M)


def test_explicit_branches_nan_outside_range(nss_path, seq_array):
    seq = EOSManager.EOSSequenceFromFile(fname=nss_path, load_ns=True, no_sort=True)
    m = np.linspace(0.2, 2.6, 97)
    for k in range(len(EXPECTED_BRANCH_COUNTS)):
        for b, (M, R, Lam) in enumerate(_ref_branches(seq_array, k)):
            inside = (m >= M[0]) & (m <= M[-1])
            lam = seq.lambda_of_m_indx(m, k, branch_id=b)
            rad = seq.R_of_m_indx(m, k, branch_id=b)
            assert np.array_equal(lam[inside], _ref_interp(m[inside], M, Lam))
            assert np.array_equal(rad[inside], _ref_interp(m[inside], M, R))
            assert np.all(np.isnan(lam[~inside])) and np.all(np.isnan(rad[~inside]))
            assert seq.m_max_of_indx(k, branch=b) == M[-1]
            assert seq.m_min_of_indx(k, branch=b) == M[0]
            # scalar input gives a scalar
            assert np.ndim(seq.lambda_of_m_indx(float(M[1]), k, branch_id=b)) == 0
            assert np.isnan(seq.lambda_of_m_indx(float(M[-1]) + 1e-6, k, branch_id=b))
        with pytest.raises(ValueError, match="outside"):
            seq.lambda_of_m_indx(1.4, k, branch_id=EXPECTED_BRANCH_COUNTS[k])
        with pytest.raises(ValueError):
            seq.m_max_of_indx(k, branch="bogus")


def test_branches_at_m_and_max_mass_all(nss_path):
    seq = EOSManager.EOSSequenceFromFile(fname=nss_path, load_ns=True, no_sort=True)
    # single
    assert seq.branches_at_m_indx(1.4, 0) == [0]
    assert seq.branches_at_m_indx(2.3, 0) == []
    assert seq.m_max_of_indx(0, branch="all") == seq.m_max_of_indx(0)
    # twin rising past the first maximum
    assert seq.branches_at_m_indx(1.9, 1) == [0, 1]
    assert seq.branches_at_m_indx(2.1, 1) == [1]
    assert seq.m_max_of_indx(1) == pytest.approx(2.0)
    assert seq.m_max_of_indx(1, branch="all") == pytest.approx(2.25)
    # third family below the first maximum
    assert seq.branches_at_m_indx(1.9, 2) == [0, 1]
    assert seq.branches_at_m_indx(1.97, 2) == [0]
    assert seq.m_max_of_indx(2, branch="all") == seq.m_max_of_indx(2) == pytest.approx(2.0)
    # three branches
    assert seq.branches_at_m_indx(1.9, 3) == [0, 1, 2]
    assert seq.branches_at_m_indx(2.05, 3) == [2]
    assert seq.m_max_of_indx(3, branch="all") == pytest.approx(2.1)
    # NaN-split EOS: the Lambda NaN at row 10 splits the rising run
    recs = seq.branches_of_indx(4)
    assert np.all(np.isfinite(recs[0]["Lambda"])) and np.all(np.isfinite(recs[1]["M"]))
    assert seq.branches_at_m_indx(1.4, 4) == [1]


def test_sorted_order_bookkeeping(tmp_path, seq_array):
    path = _write_nss(tmp_path / "s.nss.h5", seq_array)
    unsorted = EOSManager.EOSSequenceFromFile(
        fname=path, load_ns=True, no_sort=True,
        oned_order_name="Lambda", oned_order_mass=1.3)
    srt = EOSManager.EOSSequenceFromFile(
        fname=path, load_ns=True, no_sort=False,
        oned_order_name="Lambda", oned_order_mass=1.3)
    assert srt.oned_order_sorted and not unsorted.oned_order_sorted
    m = np.linspace(0.5, 2.3, 41)
    for i in range(len(EXPECTED_BRANCH_COUNTS)):
        name = srt._name_of_indx(i)
        k = int(name.split("_")[1])
        # branch data and the primary accessor resolve to the same EOS
        assert srt.n_branches_of_indx(i) == srt.stable_branch_counts[k]
        assert np.array_equal(srt.lambda_of_m_indx(m, i),
                              unsorted.lambda_of_m_indx(m, k))
        for b in range(srt.n_branches_of_indx(i)):
            assert np.array_equal(srt.lambda_of_m_indx(m, i, branch_id=b),
                                  unsorted.lambda_of_m_indx(m, k, branch_id=b),
                                  equal_nan=True)
            assert srt.m_max_of_indx(i, branch=b) == unsorted.m_max_of_indx(k, branch=b)
        assert srt.m_max_of_indx(i, branch="all") == unsorted.m_max_of_indx(k, branch="all")
        assert srt.branches_at_m_indx(1.9, i) == unsorted.branches_at_m_indx(1.9, k)


def test_pca_path(tmp_path, seq_array):
    path = _write_pca(tmp_path / "s.pca.h5", seq_array)
    seq = EOSManager.EOSSequenceFromFile(fname=path, load_ns=True, no_sort=True)
    assert isinstance(seq, EOSManager.EOSSequencePCA)
    with h5py.File(path, "r") as f:
        mean, comps, coeffs = f["mean"][:], f["components"][:], f["coeffs"][:]
    m = np.linspace(0.3, 2.5, 61)
    for k in range(coeffs.shape[0]):
        rec = mean + np.einsum("ck,ckp->cp", coeffs[k], comps)
        M, R, Lam = rec[0], rec[1], np.exp(rec[2])
        stable = np.concatenate([[True], np.diff(M) > 0]).astype(float)
        ref = _ref_stable_branches(M, R, Lam, stable)
        assert seq.stable_branch_counts[k] == seq.n_branches_of_indx(k) == len(ref)
        assert np.array_equal(seq.lambda_of_m_indx(m, k), _ref_interp(m, *ref[0][::2]))
        for b, (Mb, Rb, Lb) in enumerate(ref):
            inside = (m >= Mb[0]) & (m <= Mb[-1])
            lam = seq.lambda_of_m_indx(m, k, branch_id=b)
            assert np.array_equal(lam[inside], _ref_interp(m[inside], Mb, Lb))
            assert np.all(np.isnan(lam[~inside]))
        assert seq.m_max_of_indx(k, branch="all") == max(np.max(x[0]) for x in ref)
    # the twin and three-branch EOS keep their extra branches through PCA
    assert seq.n_branches_of_indx(1) >= 2 and seq.n_branches_of_indx(3) >= 3


def test_single_index_surface(nss_path, seq_array):
    eos = EOSManager.EOSSequenceSingleIndex(fname=nss_path, index=1)
    M0, R0, L0 = _ref_branches(seq_array, 1)[0]
    M1, R1, L1 = _ref_branches(seq_array, 1)[1]
    assert eos.number_of_branches == 2
    assert eos.mMaxMsun == float(np.max(M0))
    assert eos.mMaxMsun_all == float(np.max(M1))
    # branches_for_m: Msun and SI kg
    assert eos.branches_for_m(1.9) == [0, 1]
    assert eos.branches_for_m(1.9 * lal.MSUN_SI) == [0, 1]
    assert eos.branches_for_m(2.1) == [1]
    assert eos.branches_for_m(3.0) == []
    # branch_id=None keeps the v1 primary projection, including twin masses
    m = 1.9
    assert eos.lambda_from_m(m) == float(_ref_interp(m, M0, L0))
    assert eos.lambda_from_m(m * lal.MSUN_SI) == eos.lambda_from_m(m)
    assert eos.lambda_from_m(2.1) == 1e-8            # above primary max
    assert eos.radius_from_m(m) == pytest.approx(1e3 * float(_ref_interp(m, M0, R0)), rel=1e-15)
    assert eos.R_from_m(m) == _ref_interp(m, M0, R0)
    # explicit branches
    assert eos.lambda_from_m(m, branch_id=1) == float(_ref_interp(m, M1, L1))
    assert eos.lambda_from_m(2.1, branch_id=1) == float(_ref_interp(2.1, M1, L1))
    assert eos.lambda_from_m(0.9995 * eos.mMaxMsun_all, branch_id=1) == 1e-8
    assert eos.lambda_from_m(2.1, branch_id=0) == 1e-8   # above branch-0 max
    with pytest.raises(ValueError, match="outside stable branch 1"):
        eos.lambda_from_m(1.4, branch_id=1)
    with pytest.raises(ValueError, match="outside"):
        eos.lambda_from_m(1.4, branch_id=2)
    assert eos.radius_from_m(m, branch_id=1) == pytest.approx(
        1e3 * float(_ref_interp(m, M1, R1)), rel=1e-15)
    with pytest.raises(ValueError):
        eos.radius_from_m(1.4, branch_id=1)
    assert np.isnan(eos.R_from_m(1.4, branch_id=1))
    # vectors: None path unchanged; explicit branch rejects with -inf
    mv = np.array([1.4, 1.9, 2.1])
    assert np.array_equal(eos.lambda_from_m_vector(mv),
                          np.array([eos.lambda_from_m(x) for x in mv]))
    vb = eos.lambda_from_m_vector(mv, branch_id=1)
    assert vb[0] == -np.inf
    assert vb[1] == eos.lambda_from_m(1.9, branch_id=1)
    assert vb[2] == eos.lambda_from_m(2.1, branch_id=1)
    assert eos.lambda_from_m_vector(1.9, branch_id=1) == eos.lambda_from_m(1.9, branch_id=1)


def test_single_index_for_branch_view(nss_path, seq_array):
    eos = EOSManager.EOSSequenceSingleIndex(fname=nss_path, index=3)
    views = [eos.for_branch(b) for b in range(3)]
    for b, view in enumerate(views):
        Mb, Rb, Lb = _ref_branches(seq_array, 3)[b]
        assert isinstance(view, EOSManager.EOSBranchView)
        assert view.branch_id == b
        assert view.name.endswith("[branch={}]".format(b))
        assert view.mMaxMsun == float(Mb[-1]) == view.mMaxMsun_all
        mid = float(0.5 * (Mb[1] + Mb[-2]))
        assert view.lambda_from_m(mid) == eos.lambda_from_m(mid, branch_id=b)
        assert view.lambda_from_m(mid * lal.MSUN_SI) == eos.lambda_from_m(mid, branch_id=b)
        assert view.radius_from_m(mid) == eos.radius_from_m(mid, branch_id=b)
        assert view.R_from_m(mid) == eos.R_from_m(mid, branch_id=b)
        assert view.branches_for_m(1.9) == [b]
        out = evaluate_eos_lambda(view, np.array([0.1, mid]), use_vector=True)
        assert out[0] == -np.inf and out[1] == eos.lambda_from_m(mid, branch_id=b)
        assert view.lambda_from_m(1.01 * float(Mb[-1])) == 1e-8
    with pytest.raises(ValueError):
        eos.for_branch(3)
    with pytest.raises(ValueError):
        eos.for_branch(-1)


def test_single_index_default_matches_reference_for_all_eos(nss_path, seq_array):
    for k in range(len(EXPECTED_BRANCH_COUNTS)):
        eos = EOSManager.EOSSequenceSingleIndex(fname=nss_path, index=k)
        M, R, Lam = _ref_branches(seq_array, k)[0]
        assert eos.mMaxMsun == float(np.max(M))
        for m in np.linspace(0.2, 2.6, 25):
            ref = 1e-8 if m > 0.999 * eos.mMaxMsun else float(_ref_interp(m, M, Lam))
            assert eos.lambda_from_m(m) == ref
