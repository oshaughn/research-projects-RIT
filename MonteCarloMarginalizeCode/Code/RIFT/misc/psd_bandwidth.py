"""Estimate the frequency band a signal actually occupies, from a PSD, at workflow-build time.

WHAT THIS IS FOR.  Several build-time decisions depend on where a signal's power really sits in
[fmin, fmax] rather than on fmax itself -- most immediately the choice of sub-sample Q_lm
interpolation stencil (see RIFT.likelihood.time_interp_choice), where using fmax was measured to
pick the worse stencil by up to 330x.  The operative quantity is the bandwidth of the
matched-filter integrand, which depends on the MASSES and on fmin as well as on the PSD.

DESIGN CONSTRAINTS, both learned the hard way:

  * IT MUST NOT REQUIRE A PSD.  PSDs are routinely copied into a run directory late, so any
    build-time tool that assumes they are present will fail exactly when a human is mid-setup.
    Every entry point here returns None rather than raising or guessing, and callers are expected
    to fall back to their SAFE option -- not to their preferred one.
  * IT MUST NOT PICK VIRGO AS THE REPRESENTATIVE unless Virgo is all there is.  Virgo's noise
    curve differs enough from the LIGO detectors that using it to characterise the band would
    misrepresent a network that is mostly H/L.  See choose_representative_ifo.

numpy only at import time; lalsimutils is imported lazily inside the reader, so importing this
module costs nothing in a pipeline script.
"""
from __future__ import division

import os

import numpy as np

# Preference order for the "representative" detector.  H and L first because they dominate
# network sensitivity and share a noise curve shape; K and I ahead of V for the same reason V is
# last.  V1 is chosen ONLY when nothing else is present -- a V-only analysis is legitimate and
# must still get an answer.
IFO_PREFERENCE = ('H1', 'L1', 'K1', 'I1', 'V1')

# Fraction of the matched-filter SNR^2 that must accumulate below the reported bandwidth.
#
# CALIBRATED, and the value matters more than it looks.  Compared against Q bandwidths measured
# directly from the likelihood's own Q_lm spectra (ZDHP analytic PSD, fmin 30, fmax 1700), the
# ratio estimate/measured behaves like this:
#
#     M/Msun         2.6    5    10    20    35    55    80   120     spread
#     q = 0.9999    3.23  1.86  1.33  1.10  0.94  0.81  0.68  0.45     7.2x
#     q = 0.99      1.35  1.25  1.16  1.05  0.92  0.81  0.67  0.45     3.0x
#     q = 0.95      0.67  0.68  0.80  0.88  0.85  0.77  0.66  0.45     1.5x
#
# At a very high quantile the f_ISCO truncation dominates and the PSD contributes essentially
# nothing -- the estimator degenerates into f_ISCO and inherits its 7x drift, which is precisely
# the failure that made an earlier f_ISCO-based stencil rule unusable.  Only at a lower quantile
# does the PSD's high-frequency roll-off actually do the work, and the drift collapses.
#
# 0.95 systematically UNDER-reads the true bandwidth by ~25%, roughly uniformly (0.66-0.88
# excluding M=120, which is a degenerate 6.6 Hz-wide band).  Under-reading is the safe direction
# for the stencil decision: it inflates fNyq/bandwidth and so favours the cheaper, more forgiving
# stencil.  Do not raise this without re-checking that the estimator has not collapsed back onto
# f_ISCO -- test_psd_bandwidth guards exactly that.
#
# CALIBRATION IS PROVISIONAL: the reference bandwidths above were measured with TaylorT4, which
# terminates at ISCO and has no merger-ringdown, so the high-mass columns are not trustworthy.
# An IMR re-measurement is in progress; expect the true high-mass bandwidths to be HIGHER than
# these, which would make the current under-read larger at high mass (still the safe direction).
DEFAULT_POWER_QUANTILE = 0.95


def choose_representative_ifo(ifos):
    """Pick the detector whose PSD should characterise the band, or None if there are none.

    Prefers H1/L1, then K1/I1, and falls back to V1 only when Virgo is the ONLY detector present
    -- a V-only run still needs an answer, but a network containing H or L should never be
    characterised by Virgo's noise curve.  Unrecognised detector names are accepted after the
    known ones, so a new instrument does not silently produce None.
    """
    if not ifos:
        return None
    present = [str(x).strip() for x in ifos if str(x).strip()]
    if not present:
        return None
    for want in IFO_PREFERENCE:
        for got in present:
            if got.upper() == want:
                return got
    # unknown naming: deterministic, but do not pretend to a preference we have not reasoned about
    return sorted(present)[0]


def _read_psd(psd_path, ifo):
    """Return (freqs, psd_values) from a RIFT PSD XML, or None if it cannot be read.

    Deliberately forgiving: a missing, unreadable, or malformed PSD is a normal mid-setup state,
    not an error worth stopping a workflow build for.
    """
    if not psd_path or not os.path.isfile(psd_path):
        return None
    try:
        import RIFT.lalsimutils as lalsimutils
        psd = lalsimutils.get_psd_series_from_xmldoc(psd_path, ifo)
        if psd is None:
            return None
        values = np.asarray(psd.data.data, dtype=float)
        freqs = float(psd.f0) + float(psd.deltaF) * np.arange(len(values))
        return freqs, values
    except Exception:
        return None


def inspiral_amplitude_sq(freqs, m_total_msun=None):
    """|h(f)|^2 for a stationary-phase inspiral, up to an arbitrary constant.

    The SPA amplitude goes as f^(-7/6), so the power goes as f^(-7/3).  If a total mass is given
    the spectrum is truncated at the (2,2) GW frequency at ISCO, 4397/M Hz, which is where an
    inspiral-only description stops being meaningful.

    NOTE this is an INSPIRAL model: it has no merger-ringdown, so it UNDERSTATES the band for
    high-mass systems where merger power matters.  That is the safe direction for the stencil
    decision (it inflates fNyq/bandwidth and so favours the cheaper, more forgiving stencil), but
    it is a real limitation -- do not use this to make a claim about high-mass merger content.
    """
    freqs = np.asarray(freqs, dtype=float)
    amp_sq = np.zeros_like(freqs)
    good = freqs > 0
    amp_sq[good] = freqs[good] ** (-7.0 / 3.0)
    if m_total_msun:
        try:
            m_total = float(m_total_msun)
        except (TypeError, ValueError):
            m_total = 0.0
        if np.isfinite(m_total) and m_total > 0:
            amp_sq[freqs > (4397.0 / m_total)] = 0.0
    return amp_sq


def bandwidth_from_psd(freqs, psd_values, fmin, fmax, m_total_msun=None,
                       quantile=DEFAULT_POWER_QUANTILE):
    """Frequency below which `quantile` of the matched-filter SNR^2 accumulates, or None.

    The integrand is |h(f)|^2 / S(f) over [fmin, fmax] -- the same thing the likelihood
    integrates -- so this reports where the analysis actually has sensitivity, not merely where
    the band edges were set.

    Returns None on any unusable input, so a caller can distinguish "no estimate" from a number.
    """
    if freqs is None or psd_values is None:
        return None
    freqs = np.asarray(freqs, dtype=float)
    psd_values = np.asarray(psd_values, dtype=float)
    if freqs.size < 2 or freqs.size != psd_values.size:
        return None
    try:
        fmin = float(fmin)
        fmax = float(fmax)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(fmin) and np.isfinite(fmax)) or fmax <= fmin:
        return None
    if not (0.0 < float(quantile) < 1.0):
        return None

    band = (freqs >= fmin) & (freqs <= fmax) & np.isfinite(psd_values) & (psd_values > 0)
    if band.sum() < 2:
        return None
    f = freqs[band]
    s = psd_values[band]
    integrand = inspiral_amplitude_sq(f, m_total_msun) / s
    if not np.any(integrand > 0):
        # the whole in-band integrand was killed, e.g. f_ISCO below fmin (a binary too heavy to
        # radiate in this band at all).  No meaningful bandwidth; say so.
        return None
    cumulative = np.cumsum(integrand)
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 0:
        return None
    idx = int(np.searchsorted(cumulative, quantile * total))
    idx = min(idx, len(f) - 1)
    return float(f[idx])


def estimate_signal_bandwidth(psd_names, fmin, fmax, m_total_msun=None,
                              quantile=DEFAULT_POWER_QUANTILE):
    """Top-level: estimate the occupied bandwidth in Hz from a {ifo: psd_path} mapping.

    Returns (bandwidth_hz, ifo_used, reason).  bandwidth_hz is None whenever no estimate could be
    made, and `reason` then says why in a form fit for a log line -- callers should report it
    rather than silently substituting a default.

    NOTHING HERE RAISES.  A missing or half-copied PSD set is an ordinary mid-setup state; the
    contract is that the caller falls back to its SAFE choice on None.
    """
    if not psd_names:
        return None, None, "no PSDs available"
    ifo = choose_representative_ifo(list(psd_names.keys()))
    if ifo is None:
        return None, None, "no usable detector names in the PSD set"
    data = _read_psd(psd_names.get(ifo), ifo)
    if data is None:
        # one bad file should not sink the estimate if a sibling is readable
        for alt in [x for x in psd_names if x != ifo]:
            data = _read_psd(psd_names.get(alt), alt)
            if data is not None:
                ifo = alt
                break
    if data is None:
        return None, ifo, "PSD for %s not readable (missing or malformed)" % (ifo,)
    freqs, values = data
    bw = bandwidth_from_psd(freqs, values, fmin, fmax, m_total_msun, quantile)
    if bw is None:
        return None, ifo, "PSD for %s read, but no bandwidth could be computed in [%s, %s]" % (
            ifo, fmin, fmax)
    return bw, ifo, "from %s PSD, %.4g%% SNR^2 quantile" % (ifo, 100.0 * quantile)
