"""Estimate the frequency band a signal actually occupies, from a PSD, at workflow-build time.

WHAT THIS IS FOR.  Several build-time decisions depend on where a signal's power really sits in
[fmin, fmax] rather than on fmax itself -- most immediately the choice of sub-sample Q_lm
interpolation stencil (see RIFT.likelihood.time_interp_choice), where using fmax alone was
measured to pick the worse stencil.  The operative quantity is the bandwidth of the
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
# CHOSEN FOR SEPARATING POWER, NOT FOR RATIO ACCURACY -- those are different objectives and they
# disagree here.  Validated against 9 SEOBNRv4 (IMR) stencil measurements: rank each configuration
# by fNyq/estimate and ask whether the sinc winners and the cubic winners separate.
#
#     ranked by                     sinc wins up to   cubic wins from   separates?   gap
#     fNyq / measured 99.99%             4.628             4.233        NO (overlap)  --
#     fNyq / estimate, q = 0.95          6.059             6.113        yes          1.009x
#     fNyq / estimate, q = 0.99          2.990             4.330        yes          1.45x
#
# q = 0.95 gives the most uniform estimate/measured RATIO (spread 1.36x against IMR) but leaves a
# 1% window to place a threshold in, which is not usable.  q = 0.99 has a worse ratio spread
# (1.76x) and a 45%-wide window.  For a decision, separation is what matters.
#
# Note the raw measured 99.99%-power bandwidth does NOT separate them at all -- an IMR spectrum
# has a ringdown bump rather than a smooth roll-off, so a very high quantile chases the bump.
# This estimator works precisely because it integrates against the PSD instead.
#
# If a stencil selector is ever built on this: threshold ~ 3.6 (the geometric mean of the
# 2.99-4.33 bracket).  NOT wired in yet -- the fmin dependence has not been re-checked with an
# IMR model, and with TaylorT4 fmin alone flipped the winner at M = 5.
DEFAULT_POWER_QUANTILE = 0.99


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


# Characteristic frequencies of an IMR signal, as multiples of the GW frequency at ISCO.
# f_ISCO is NOT where the signal stops -- it is where the inspiral description stops and merger
# begins.  A real binary keeps radiating through merger and ringdown, and for a remnant spin
# a ~ 0.7 the (2,2) ringdown sits at ~3.9 f_ISCO.  Truncating at f_ISCO models the TERMINATION OF
# AN APPROXIMANT (TaylorT4 stops there by construction), not the physics.
MERGER_OVER_ISCO = 2.0        # inspiral -> merger transition
RINGDOWN_OVER_ISCO = 3.9      # (2,2) ringdown of an a~0.7 remnant
RINGDOWN_Q = 3.0              # QNM quality factor; the Lorentzian width is f_ring / (2 Q)
CUTOFF_OVER_RINGDOWN = 3.0    # where the ringdown Lorentzian has fallen far enough to drop


def imr_amplitude_sq(freqs, m_total_msun=None):
    """|h(f)|^2 for an inspiral-merger-ringdown signal, up to an arbitrary constant.

    Piecewise, in the standard IMRPhenom shape:

        f <  f_merg    inspiral   |h| ~ f^(-7/6)   ->  |h|^2 ~ f^(-7/3)
        f <  f_ring    merger     |h| ~ f^(-2/3)   ->  |h|^2 ~ f^(-4/3)
        f >= f_ring    ringdown   Lorentzian of width f_ring / (2 Q)

    WHY NOT SIMPLY TRUNCATE AT f_ISCO.  An earlier version of this function did, and it was
    wrong in a way that mattered: f_ISCO is where an inspiral-only APPROXIMANT terminates, not
    where a binary stops radiating.  Truncating there hard-codes the artifact -- it also made the
    whole estimator degenerate into f_ISCO, reproducing the 7.4x drift that made an f_ISCO-based
    stencil rule unusable in the first place.  Here f_ISCO only sets the SCALE of the merger and
    ringdown features; real power continues to ~4x it.

    With no mass supplied this falls back to the pure inspiral power law, because the merger
    scale is unknown -- that is the one case where the caller genuinely has nothing better.
    """
    freqs = np.asarray(freqs, dtype=float)
    amp_sq = np.zeros_like(freqs)
    good = freqs > 0
    amp_sq[good] = freqs[good] ** (-7.0 / 3.0)

    m_total = None
    if m_total_msun:
        try:
            m_total = float(m_total_msun)
        except (TypeError, ValueError):
            m_total = None
        if m_total is not None and not (np.isfinite(m_total) and m_total > 0):
            m_total = None
    if m_total is None:
        return amp_sq

    f_isco = 4397.0 / m_total
    f_merg = MERGER_OVER_ISCO * f_isco
    f_ring = RINGDOWN_OVER_ISCO * f_isco
    sigma = f_ring / (2.0 * RINGDOWN_Q)
    f_cut = f_ring + CUTOFF_OVER_RINGDOWN * sigma

    # merger: |h|^2 ~ f^(-4/3), matched to the inspiral value at f_merg so the spectrum is
    # continuous (the absolute normalisation is irrelevant -- only the SHAPE sets the quantile).
    merger = good & (freqs >= f_merg) & (freqs < f_ring)
    if np.any(merger):
        scale = f_merg ** (-7.0 / 3.0) / (f_merg ** (-4.0 / 3.0))
        amp_sq[merger] = scale * freqs[merger] ** (-4.0 / 3.0)

    # ringdown: Lorentzian in |h|, so |h|^2 is the square, matched at f_ring
    ring = good & (freqs >= f_ring) & (freqs <= f_cut)
    if np.any(ring):
        amp_ring = f_merg ** (-7.0 / 6.0) / (f_merg ** (-2.0 / 3.0)) * f_ring ** (-2.0 / 3.0)
        lorentz = 1.0 / (1.0 + ((freqs[ring] - f_ring) / (0.5 * sigma)) ** 2)
        amp_sq[ring] = (amp_ring * lorentz) ** 2

    amp_sq[freqs > f_cut] = 0.0
    return amp_sq


# Backwards-compatible alias.  The old name promised inspiral-only behaviour, which is no longer
# what this does; keep it working but point callers at the accurate name.
inspiral_amplitude_sq = imr_amplitude_sq


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
    integrand = imr_amplitude_sq(f, m_total_msun) / s
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
    if choose_representative_ifo(list(psd_names.keys())) is None:
        return None, None, "no usable detector names in the PSD set"
    # One bad file must not sink the estimate if a sibling is readable -- but the fallback has to
    # keep obeying the PREFERENCE order, not dict insertion order.  Re-running the chooser over
    # the remaining candidates is what makes {'H1': malformed, 'V1': ok, 'L1': ok} pick L1; a
    # plain iteration over the mapping picks whichever happens to come first, which for that
    # example is Virgo, silently violating this module's stated representative-detector invariant.
    remaining = list(psd_names.keys())
    data = None
    while remaining:
        ifo = choose_representative_ifo(remaining)
        if ifo is None:
            break
        data = _read_psd(psd_names.get(ifo), ifo)
        if data is not None:
            break
        remaining = [x for x in remaining if x != ifo]
    if data is None:
        return None, ifo, "PSD for %s not readable (missing or malformed)" % (ifo,)
    freqs, values = data
    bw = bandwidth_from_psd(freqs, values, fmin, fmax, m_total_msun, quantile)
    if bw is None:
        return None, ifo, "PSD for %s read, but no bandwidth could be computed in [%s, %s]" % (
            ifo, fmin, fmax)
    return bw, ifo, "from %s PSD, %.4g%% SNR^2 quantile" % (ifo, 100.0 * quantile)
