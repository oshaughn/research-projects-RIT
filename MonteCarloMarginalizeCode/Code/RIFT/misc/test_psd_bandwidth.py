#!/usr/bin/env python3
"""test_psd_bandwidth -- representative-detector choice, and the fallback contract.

Two things are guarded, both of which are about behaviour under imperfect input rather than
about the arithmetic:

  1. VIRGO IS NOT THE REPRESENTATIVE unless Virgo is all there is.  Its noise curve differs
     enough from H/L that characterising an H/L/V network by it would misdescribe the band --
     but a V-only analysis is legitimate and must still get an answer.
  2. EVERY FAILURE RETURNS None WITH A REASON, and none of them raise.  PSDs get copied into a
     run directory late, so "no PSD yet" is an ordinary mid-setup state, not an error.  The
     contract is that callers fall back to their SAFE option on None -- a tool that raised, or
     that quietly guessed, would be worse than no tool.

Self-contained: numpy only, no lal, no data.  Runs instantly.

    python3 test_psd_bandwidth.py        # or: pytest test_psd_bandwidth.py
"""
from __future__ import print_function

import numpy as np

from RIFT.misc.psd_bandwidth import (
    IFO_PREFERENCE,
    bandwidth_from_psd,
    choose_representative_ifo,
    estimate_signal_bandwidth,
    inspiral_amplitude_sq,
)


def test_virgo_is_last_but_not_excluded():
    """The rule RO'S asked for: not Virgo unless V-only."""
    assert choose_representative_ifo(['H1', 'L1', 'V1']) == 'H1'
    assert choose_representative_ifo(['V1', 'L1']) == 'L1'
    assert choose_representative_ifo(['V1', 'K1']) == 'K1'
    # ...but a V-only run must still get an answer, not None
    assert choose_representative_ifo(['V1']) == 'V1'
    print("V1 chosen only when alone; H1/L1/K1 preferred otherwise: OK")


def test_representative_choice_is_order_independent_and_total():
    """The answer must not depend on dict/list ordering, and unknown names must not give None."""
    for order in (['H1', 'L1', 'V1'], ['V1', 'H1', 'L1'], ['L1', 'V1', 'H1']):
        assert choose_representative_ifo(order) == 'H1', order
    # unknown instrument names: deterministic, and never None just because we do not know them
    got = choose_representative_ifo(['X9', 'A3'])
    assert got == 'A3', got
    # ...but a known name still wins over an unknown one
    assert choose_representative_ifo(['X9', 'L1']) == 'L1'
    # empty / degenerate input is the one case that legitimately gives None
    for empty in ([], None, ['', '  ']):
        assert choose_representative_ifo(empty) is None, empty
    print("choice is order-independent, total over unknown names, None only when empty: OK")


def test_every_failure_returns_none_with_a_reason_and_never_raises():
    """The fallback contract.  A caller must be able to tell 'no estimate' from a number."""
    cases = [
        ({}, "empty mapping"),
        (None, "None mapping"),
        ({'H1': '/nonexistent/path/to/H1-psd.xml.gz'}, "missing file"),
        ({'H1': None}, "None path"),
        ({'': ''}, "blank names"),
    ]
    for psd_names, label in cases:
        bw, ifo, reason = estimate_signal_bandwidth(psd_names, 20.0, 1700.0, m_total_msun=30.0)
        assert bw is None, "%s must give no estimate, got %r" % (label, bw)
        assert isinstance(reason, str) and reason, "%s must give a reason for the log" % label
    print("all failure modes return None with a reason, none raise: OK")


def _flat_psd(f_lo=5.0, f_hi=4096.0, df=0.25):
    freqs = np.arange(f_lo, f_hi + df, df)
    return freqs, np.ones_like(freqs) * 1e-46


def test_bandwidth_is_bounded_by_the_band_and_by_the_mass():
    """Sanity that the estimate means what it says, on a flat PSD where the answer is analytic."""
    freqs, psd = _flat_psd()
    bw = bandwidth_from_psd(freqs, psd, 20.0, 1700.0, m_total_msun=2.6)
    assert bw is not None and 20.0 <= bw <= 1700.0, bw

    # A heavier binary must give a LOWER bandwidth: f_ISCO falls as 1/M.
    bw_light = bandwidth_from_psd(freqs, psd, 20.0, 1700.0, m_total_msun=2.6)
    bw_heavy = bandwidth_from_psd(freqs, psd, 20.0, 1700.0, m_total_msun=80.0)
    print("flat PSD, fmin 20, fmax 1700: M=2.6 -> %.1f Hz, M=80 -> %.1f Hz" % (bw_light, bw_heavy))
    assert bw_heavy < bw_light, "a heavier binary must occupy a narrower band (%g vs %g)" % (
        bw_heavy, bw_light)

    # Raising fmin must not lower the bandwidth -- the band only loses low-frequency content.
    bw_lo = bandwidth_from_psd(freqs, psd, 20.0, 1700.0, m_total_msun=5.0)
    bw_hi = bandwidth_from_psd(freqs, psd, 150.0, 1700.0, m_total_msun=5.0)
    print("M=5: fmin 20 -> %.1f Hz, fmin 150 -> %.1f Hz" % (bw_lo, bw_hi))
    assert bw_hi >= bw_lo, "raising fmin must not reduce the estimated bandwidth"


def test_binary_too_heavy_for_the_band_gives_no_estimate():
    """f_ISCO below fmin means the system does not radiate in band at all.

    Returning a number here would be worse than returning None: it would be a bandwidth for a
    signal that is not there.
    """
    freqs, psd = _flat_psd()
    bw = bandwidth_from_psd(freqs, psd, 100.0, 1700.0, m_total_msun=1000.0)  # f_ISCO ~ 4.4 Hz
    assert bw is None, "a binary with f_ISCO below fmin must give no estimate, got %r" % bw
    print("binary too heavy to radiate in band -> no estimate: OK")


def test_malformed_psd_inputs_return_none():
    freqs, psd = _flat_psd()
    assert bandwidth_from_psd(None, psd, 20, 1700) is None
    assert bandwidth_from_psd(freqs, None, 20, 1700) is None
    assert bandwidth_from_psd(freqs, psd[:-5], 20, 1700) is None          # length mismatch
    assert bandwidth_from_psd(freqs, psd, 1700, 20) is None               # inverted band
    assert bandwidth_from_psd(freqs, psd, 'x', 1700) is None              # unparseable
    assert bandwidth_from_psd(freqs, np.zeros_like(psd), 20, 1700) is None  # PSD all zero
    assert bandwidth_from_psd(freqs, psd, 20, 1700, quantile=1.5) is None   # bad quantile
    print("malformed PSD inputs return None: OK")


def test_inspiral_amplitude_truncates_at_isco():
    f = np.array([10.0, 100.0, 1000.0])
    a_untrunc = inspiral_amplitude_sq(f)
    assert np.all(a_untrunc > 0)
    # M=55 -> f_ISCO ~ 80 Hz, so only the 10 Hz bin survives
    a_trunc = inspiral_amplitude_sq(f, m_total_msun=55.0)
    assert a_trunc[0] > 0 and a_trunc[1] == 0 and a_trunc[2] == 0
    # power-law shape, f^(-7/3)
    ratio = a_untrunc[0] / a_untrunc[1]
    assert abs(ratio - 10.0 ** (7.0 / 3.0)) < 1e-6 * ratio
    print("inspiral amplitude is f^(-7/3), truncated at f_ISCO: OK")


def test_estimator_does_not_degenerate_into_f_isco():
    """THE STRUCTURAL GUARD.  The PSD must actually influence the answer.

    At a very high power quantile the f_ISCO truncation dominates and this tool returns f_ISCO to
    within a percent, contributing nothing over a formula that needs no PSD at all -- and
    inheriting f_ISCO's measured 7.4x drift against true Q bandwidth, which is exactly what made
    an earlier f_ISCO-based stencil rule unusable.  The default quantile must sit where the PSD's
    high-frequency roll-off is doing real work.

    Uses a synthetic PSD that rises steeply above a knee, which is the feature of a real detector
    curve that makes this work.  No lal, no data.
    """
    df = 0.25
    freqs = np.arange(df, 2048.0 + df, df)
    knee = 300.0
    psd = 1e-46 * (1.0 + (freqs / knee) ** 4)          # flat, then steeply rising

    for m_total in (5.0, 20.0, 55.0):
        f_isco = 4397.0 / m_total
        bw = bandwidth_from_psd(freqs, psd, 30.0, 1700.0, m_total_msun=m_total)
        assert bw is not None
        # must be strictly inside the truncation, not sitting on it
        assert bw < 0.98 * f_isco, (
            "at M=%g the estimate (%.1f Hz) is within 2%% of f_ISCO (%.1f Hz): the PSD is not "
            "influencing the answer, so this tool has degenerated into a formula that needs no "
            "PSD -- and inherits f_ISCO's 7.4x drift. Lower DEFAULT_POWER_QUANTILE."
            % (m_total, bw, f_isco))
        print("M=%5.1f: estimate %6.1f Hz vs f_ISCO %6.1f Hz (%.0f%% of it)"
              % (m_total, bw, f_isco, 100.0 * bw / f_isco))

    # ...and the PSD must demonstrably matter: extra high-frequency noise must narrow the band.
    #
    # Compare a flat PSD against the same PSD with a wall of extra noise above 200 Hz.  (Do NOT
    # compare (1+(f/knee)^4) against (1+(f/knee)^8) expecting the latter to be "steeper" -- below
    # the knee x^8 < x^4, so that curve is actually the QUIETER one exactly where the quantile
    # lands, and the comparison measures the opposite of what it looks like.)
    flat = np.ones_like(freqs) * 1e-46
    walled = flat.copy()
    walled[freqs > 200.0] *= 1000.0
    bw_flat = bandwidth_from_psd(freqs, flat, 30.0, 1700.0, m_total_msun=5.0)
    bw_walled = bandwidth_from_psd(freqs, walled, 30.0, 1700.0, m_total_msun=5.0)
    print("M=5, flat PSD %.1f Hz -> with a noise wall above 200 Hz %.1f Hz"
          % (bw_flat, bw_walled))
    assert bw_walled < bw_flat, (
        "adding high-frequency noise must reduce the estimated bandwidth (%g vs %g); if it does "
        "not, the PSD is being ignored" % (bw_walled, bw_flat))


def test_preference_list_is_sane():
    assert IFO_PREFERENCE[-1] == 'V1', "V1 must be last in the preference order"
    assert IFO_PREFERENCE[0] in ('H1', 'L1')
    assert len(set(IFO_PREFERENCE)) == len(IFO_PREFERENCE), "no duplicates"


if __name__ == "__main__":
    test_virgo_is_last_but_not_excluded()
    test_representative_choice_is_order_independent_and_total()
    test_every_failure_returns_none_with_a_reason_and_never_raises()
    test_bandwidth_is_bounded_by_the_band_and_by_the_mass()
    test_binary_too_heavy_for_the_band_gives_no_estimate()
    test_malformed_psd_inputs_return_none()
    test_inspiral_amplitude_truncates_at_isco()
    test_estimator_does_not_degenerate_into_f_isco()
    test_preference_list_is_sane()
    print("\nPASS")
