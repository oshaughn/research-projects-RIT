"""Precompute cross-term batching: the bit-identical fixes stay bit-identical, and the
opt-in batched path stays off by default and agrees with the loop it replaces.

Companion to DESIGN_precompute_crossterm_batching.md.  Three of the four changes under
test claim BIT identity, not approximate agreement, so each is checked against a replay
of the exact code it replaced rather than against a tolerance.
"""
import os

import numpy as np
import pytest

lal = pytest.importorskip("lal")

import RIFT.lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as FL

FNYQ = 512.0
DELTAF = 1.0 / 8.0
FMIN, FMAX = 20.0, 256.0
LEN1 = int(FNYQ / DELTAF) + 1
LEN2 = 2 * (LEN1 - 1)
TSPEC = 1.0            # N_spec = 1024 < LEN2/2 = 4096


def _psd_array(dead_bin=True, negative_bin=False):
    f = np.arange(LEN1) * DELTAF
    psd = np.zeros(LEN1)
    b = (f >= FMIN) & (f <= FMAX)
    psd[b] = 1e-46 * ((f[b] / 100.0) ** -4.14 + 2.0 + 0.5 * (f[b] / 100.0) ** 2)
    if dead_bin:
        psd[300] = 0.0        # a dead bin inside the band
    if negative_bin:
        psd[400] = -1e-46     # pins `!= 0` (the shipped loop) against `> 0`
    return psd


def _psd_series(arr):
    s = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0.0), 0.0, DELTAF,
                                       lsu.lsu_HertzUnit, LEN1)
    s.data.data[:] = arr
    return s


def _series(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = lal.CreateCOMPLEX16FrequencySeries("h", lal.LIGOTimeGPS(0.0), 0.0, DELTAF,
                                               lsu.lsu_DimensionlessUnit, LEN2)
        s.data.data[:] = (rng.standard_normal(LEN2) + 1j * rng.standard_normal(LEN2)) * 1e-23
        out.append(s)
    return out


# --------------------------------------------------------------------------------------
# bit-identity of the three loop -> vector rewrites
# --------------------------------------------------------------------------------------
def test_array_psd_weights_bit_identical():
    """The vectorised array-PSD fill reproduces the per-bin loop exactly, INCLUDING its
    `!= 0` mask -- the REAL8FrequencySeries branch uses `> 0`, which would silently drop a
    negative bin, and the two branches must not be conflated."""
    psd = _psd_array(negative_bin=True)
    ip = lsu.ComplexIP(FMIN, FMAX, FNYQ, DELTAF, psd, False, False, 0.0)
    ref = np.zeros(LEN1)
    for i in range(ip.minIdx, ip.maxIdx):
        if psd[i] != 0.0:
            ref[i] = 1.0 / psd[i] * 1.0
    assert np.array_equal(ip.weights, ref)
    assert ip.weights[400] < 0, "negative PSD bin must survive the `!= 0` mask"


def test_array_psd_weights_bit_identical_psi4():
    psd = _psd_array()
    ip = lsu.ComplexIP(FMIN, FMAX, FNYQ, DELTAF, psd, False, False, 0.0,
                       waveform_is_psi4=True)
    ref = np.zeros(LEN1)
    for i in range(ip.minIdx, ip.maxIdx):
        if psd[i] != 0.0:
            ew = 1.0 / (2 * np.pi * i * DELTAF) / (2 * np.pi * i * DELTAF)
            ref[i] = 1.0 / psd[i] * ew
    assert np.array_equal(ip.weights, ref)


def test_inv_spec_trunc_weights_bit_identical():
    """The slice-assignment zeroing reproduces the per-element SWIG loop exactly."""
    ser = _psd_series(_psd_array())
    ip = lsu.ComplexIP(FMIN, FMAX, FNYQ, DELTAF, ser, False, True, TSPEC)

    w = np.zeros(LEN1)
    iv = np.arange(ip.minIdx, ip.maxIdx)
    ok = iv[ser.data.data[iv] > 0]
    w[ok] = 1.0 / ser.data.data[ok] * np.ones(LEN1)[ok]
    n_spec = int(TSPEC / ip.deltaT)
    wfd = lal.CreateCOMPLEX16FrequencySeries("w", lal.LIGOTimeGPS(0.0), 0.0, DELTAF,
                                             lsu.lsu_DimensionlessUnit, ip.len1side)
    wtd = lal.CreateREAL8TimeSeries("w", lal.LIGOTimeGPS(0.0), 0.0, ip.deltaT,
                                    lsu.lsu_DimensionlessUnit, ip.len2side)
    fwd = lal.CreateForwardREAL8FFTPlan(ip.len2side, 0)
    rev = lal.CreateReverseREAL8FFTPlan(ip.len2side, 0)
    wfd.data.data[:] = np.sqrt(w)
    wfd.data.data[0] = wfd.data.data[-1] = 0.0
    lal.REAL8FreqTimeFFT(wtd, wfd, rev)
    for i in range(int(n_spec / 2), ip.len2side - int(n_spec / 2)):
        wtd.data.data[i] = 0.0
    lal.REAL8TimeFreqFFT(wfd, wtd, fwd)
    wfd.data.data[0] = wfd.data.data[-1] = 0.0
    assert np.array_equal(ip.weights, np.abs(wfd.data.data * wfd.data.data))


def test_ip_bit_identical_without_epoch_differences():
    ip = lsu.ComplexIP(FMIN, FMAX, FNYQ, DELTAF, _psd_array(), False, False, 0.0)
    h1, h2 = _series(2, 11)
    shipped = np.sum(np.conj(h1.data.data) * h2.data.data
                     * np.ones(LEN2) * ip.weights2side) * 2.0 * ip.deltaF
    assert ip.ip(h1, h2) == shipped


def test_ip_epoch_difference_path_still_reached():
    """The `include_epoch_differences` branch must still apply a phase, not silently no-op."""
    ip = lsu.ComplexIP(FMIN, FMAX, FNYQ, DELTAF, _psd_array(), False, False, 0.0)
    h1, h2 = _series(2, 12)
    h1.epoch = lal.LIGOTimeGPS(0.25)
    plain = ip.ip(h1, h2)
    shifted = ip.ip(h1, h2, include_epoch_differences=True)
    assert plain != shifted


# --------------------------------------------------------------------------------------
# band support
# --------------------------------------------------------------------------------------
def test_band_support_matches_actual_nonzeros():
    """Recorded support must be read off the weights, never assumed from (fmin, fMax):
    inverse spectrum truncation -- ON by default in the driver -- smears the band to full
    support, and a batched path that skipped to [fmin, fMax] there would drop ~1e-6 of the
    weight."""
    narrow = lsu.ComplexIP(FMIN, FMAX, FNYQ, DELTAF, _psd_array(), False, False, 0.0)
    nz = np.nonzero(narrow.weights2side)[0]
    assert (narrow.band_lo2side, narrow.band_hi2side) == (int(nz[0]), int(nz[-1]) + 1)
    assert narrow.band_hi2side - narrow.band_lo2side < narrow.len2side

    trunc = lsu.ComplexIP(FMIN, FMAX, FNYQ, DELTAF, _psd_series(_psd_array()),
                          False, True, TSPEC)
    nz = np.nonzero(trunc.weights2side)[0]
    assert (trunc.band_lo2side, trunc.band_hi2side) == (int(nz[0]), int(nz[-1]) + 1)


# --------------------------------------------------------------------------------------
# the batched path
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("trunc", [False, True])
def test_ip_matrix_matches_ip_loop(trunc):
    psd = _psd_series(_psd_array()) if trunc else _psd_array()
    ip = lsu.ComplexIP(FMIN, FMAX, FNYQ, DELTAF, psd, False, trunc, TSPEC if trunc else 0.0)
    A, B = _series(4, 21), _series(3, 22)
    loop = np.array([[ip.ip(a, b) for b in B] for a in A])
    mat = ip.ip_matrix(A, B)
    assert mat.shape == loop.shape
    assert np.max(np.abs(mat - loop)) <= 1e-13 * np.max(np.abs(loop))


def _hlms(keys, seed):
    return dict(zip(keys, _series(len(keys), seed)))


MODES = [(2, -2), (2, 0), (2, 2), (3, -3)]


def test_crossterm_batched_matches_loop_general():
    psd = _psd_array()
    a, b = _hlms(MODES, 31), _hlms(MODES, 32)
    kw = dict(analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.0, verbose=False)
    ref = FL.ComputeModeCrossTermIP(a, b, psd, FMIN, FMAX, FNYQ, DELTAF, batched=False, **kw)
    got = FL.ComputeModeCrossTermIP(a, b, psd, FMIN, FMAX, FNYQ, DELTAF, batched=True, **kw)
    assert set(ref) == set(got)
    scale = max(abs(v) for v in ref.values())
    assert max(abs(got[k] - ref[k]) for k in ref) <= 1e-13 * scale


@pytest.mark.parametrize("prefix", ["U", "V"])
def test_crossterm_batched_preserves_same_waveform_symmetry(prefix):
    """same_waveform_Q fills the lower triangle by mirroring the upper one.  The batched
    path computes the full matrix, so it must re-impose that mirror rather than keep the
    independently-computed element -- otherwise an exact symmetry becomes approximate."""
    psd = _psd_array()
    a = _hlms(MODES, 41)
    kw = dict(analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.0, verbose=False,
              prefix=prefix, same_waveform_Q=True)
    ref = FL.ComputeModeCrossTermIP(a, a, psd, FMIN, FMAX, FNYQ, DELTAF, batched=False, **kw)
    got = FL.ComputeModeCrossTermIP(a, a, psd, FMIN, FMAX, FNYQ, DELTAF, batched=True, **kw)
    assert set(ref) == set(got)
    scale = max(abs(v) for v in ref.values())
    assert max(abs(got[k] - ref[k]) for k in ref) <= 1e-13 * scale
    # Off-diagonal only: the shipped `pairs = combinations(mode_list, 2)` loop never
    # touches the diagonal, so no mirror is claimed there.  The loop result is asserted
    # first, as a control on the criterion itself -- if it did not hold for the shipped
    # path, holding for the batched one would mean nothing.
    for i, m1 in enumerate(MODES):
        for m2 in MODES[i + 1:]:
            for d in (ref, got):
                mirror = d[(m2, m1)] if prefix == "V" else np.conj(d[(m2, m1)])
                assert d[(m1, m2)] == mirror


def test_batched_is_off_unless_asked(monkeypatch):
    """Default must be the shipped path.  A flag nobody can see fire is a silent no-op, so
    the batched path bumps a counter and this pins that it stays put."""
    monkeypatch.delenv("RIFT_PRECOMPUTE_BATCHED_CROSSTERMS", raising=False)
    assert FL._crossterm_batched_default() is False
    psd = _psd_array()
    a, b = _hlms(MODES[:2], 51), _hlms(MODES[:2], 52)
    before = FL._CROSSTERM_BATCH_CALLS[0]
    FL.ComputeModeCrossTermIP(a, b, psd, FMIN, FMAX, FNYQ, DELTAF, verbose=False)
    assert FL._CROSSTERM_BATCH_CALLS[0] == before

    monkeypatch.setenv("RIFT_PRECOMPUTE_BATCHED_CROSSTERMS", "1")
    assert FL._crossterm_batched_default() is True
    FL.ComputeModeCrossTermIP(a, b, psd, FMIN, FMAX, FNYQ, DELTAF, verbose=False)
    assert FL._CROSSTERM_BATCH_CALLS[0] == before + 1
