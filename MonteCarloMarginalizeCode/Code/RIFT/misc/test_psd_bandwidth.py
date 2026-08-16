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
    imr_amplitude_sq,
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


def test_amplitude_is_a_power_law_in_the_inspiral():
    f = np.array([10.0, 100.0, 1000.0])
    a_untrunc = imr_amplitude_sq(f)
    assert np.all(a_untrunc > 0)
    # power-law shape in the inspiral, f^(-7/3)
    ratio = a_untrunc[0] / a_untrunc[1]
    assert abs(ratio - 10.0 ** (7.0 / 3.0)) < 1e-6 * ratio
    print("inspiral amplitude is f^(-7/3): OK")


def test_signal_has_power_above_f_isco():
    """f_ISCO is the TERMINATION POINT OF AN APPROXIMANT, not where a binary stops radiating.

    An earlier version of this module truncated |h|^2 at f_ISCO. That hard-coded TaylorT4's
    behaviour (it terminates at ISCO by construction) as if it were physics, and it made the
    whole estimator degenerate into f_ISCO -- inheriting the 7.4x drift that made an f_ISCO-based
    stencil rule unusable. A real IMR signal keeps radiating through merger and ringdown, to
    ~4x f_ISCO.
    """
    for m_total in (5.0, 20.0, 55.0):
        f_isco = 4397.0 / m_total
        probe = np.array([0.5, 1.5, 3.0, 8.0]) * f_isco
        amp = imr_amplitude_sq(probe, m_total_msun=m_total)
        assert amp[0] > 0 and amp[1] > 0, "inspiral and merger must carry power"
        assert amp[2] > 0, (
            "M=%g: no power at 3x f_ISCO -- the spectrum is being truncated at the approximant's "
            "termination point rather than modelling merger-ringdown" % m_total)
        assert amp[3] == 0, "power must eventually cut off well above ringdown"
    print("IMR spectrum carries power to ~4x f_ISCO, not truncated at it: OK")


def test_quieter_high_frequency_noise_widens_the_band():
    """THE STRUCTURAL GUARD, and the forward-looking one.

    Real detector high-frequency walls are NOT steep -- aLIGO ZDHP is only ~3.7x its minimum at
    1500 Hz -- and future detectors are flatter still. So the estimate must respond to the
    high-frequency noise level in the right direction: making the detector quieter up there must
    WIDEN the occupied band, because more high-frequency signal becomes measurable.

    A tool that failed this would be reporting the waveform's scale while ignoring the detector,
    which is the failure mode that motivated writing it.
    """
    df = 0.25
    freqs = np.arange(df, 2048.0 + df, df)
    # a realistic shape: flat bucket, GENTLE high-frequency rise (not the steep wall it is
    # tempting to write -- see the module docstring)
    base = 1e-46 * (1.0 + (freqs / 800.0) ** 2)

    prev = None
    for factor in (1.0, 3.0, 10.0, 100.0):
        psd = base.copy()
        psd[freqs > 300.0] /= factor
        bw = bandwidth_from_psd(freqs, psd, 30.0, 1700.0, m_total_msun=20.0)
        assert bw is not None
        print("high-f noise divided by %5.0f -> bandwidth %6.1f Hz" % (factor, bw))
        if prev is not None:
            assert bw > prev, (
                "reducing high-frequency noise by %gx did not widen the band (%.1f -> %.1f Hz); "
                "the estimator is ignoring the detector" % (factor, prev, bw))
        prev = bw


def test_fallback_after_unreadable_psd_returns_the_preferred_READABLE_detector():
    """The requested scenario: H1 MALFORMED while both L1 and V1 are READABLE -> must return L1.

    An earlier version of this test made every _read_psd call fail and only checked the order of
    attempts.  That is not the same claim: it never established which detector is actually USED,
    which is the invariant ("not Virgo unless V-only") the fallback was violating.  Here the
    siblings really do return usable data, so the assertion is on the RESULT.

    Asserted across insertion orders, because the original bug was that the fallback followed
    dict order -- a dict that happens to be ordered favourably would hide it.
    """
    import RIFT.misc.psd_bandwidth as mod

    df = 0.25
    freqs = np.arange(df, 2048.0 + df, df)
    # distinguishable curves, so a wrong pick would also change the number
    curves = {'L1': 1e-46 * (1.0 + (freqs / 800.0) ** 2),
              'V1': 1e-45 * (1.0 + (freqs / 200.0) ** 2)}   # noisier, and rolls off sooner

    orig = mod._read_psd
    try:
        def fake_read(path, ifo):
            if ifo == 'H1':
                return None          # malformed / half-copied, the realistic mid-setup state
            return (freqs, curves[ifo])
        mod._read_psd = fake_read

        results = {}
        for order in (['H1', 'V1', 'L1'], ['H1', 'L1', 'V1'], ['V1', 'L1', 'H1']):
            psd_names = dict((k, '/wherever/%s-psd.xml.gz' % k) for k in order)
            bw, ifo, reason = mod.estimate_signal_bandwidth(
                psd_names, 30.0, 1700.0, m_total_msun=20.0)
            print("insertion %-18s -> used %s, bandwidth %s Hz"
                  % (order, ifo, ("%.1f" % bw) if bw else None))
            assert ifo == 'L1', (
                "insertion order %s selected %r; with H1 unreadable and BOTH L1 and V1 readable "
                "the representative must be L1. Selecting V1 violates the module's stated "
                "invariant, and it happens precisely when a PSD file is bad." % (order, ifo))
            assert bw is not None, "a readable sibling must still yield an estimate"
            assert 'L1' in reason, "the reason line must name the detector actually used: %r" % reason
            results[tuple(order)] = bw

        # the answer must not depend on insertion order either
        assert len(set(results.values())) == 1, \
            "bandwidth varied with dict insertion order: %r" % results

        # ...and the V1 curve really is distinguishable, so the assertion above has teeth:
        # if V1 had been chosen the number would differ.
        bw_v_only, ifo_v, _ = mod.estimate_signal_bandwidth(
            {'V1': '/wherever/V1-psd.xml.gz'}, 30.0, 1700.0, m_total_msun=20.0)
        assert ifo_v == 'V1', "a V-only network must still be answered"
        assert abs(bw_v_only - list(results.values())[0]) > 1.0, (
            "the L1 and V1 curves give the same bandwidth (%.1f), so 'it returned L1' is not "
            "actually distinguishable from 'it returned V1' -- strengthen the fixture"
            % bw_v_only)
        print("V-only network answered with V1 (%.1f Hz), distinct from L1: OK" % bw_v_only)
    finally:
        mod._read_psd = orig


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
    test_amplitude_is_a_power_law_in_the_inspiral()
    test_signal_has_power_above_f_isco()
    test_quieter_high_frequency_noise_widens_the_band()
    test_fallback_after_unreadable_psd_returns_the_preferred_READABLE_detector()
    test_preference_list_is_sane()
    print("\nPASS")
