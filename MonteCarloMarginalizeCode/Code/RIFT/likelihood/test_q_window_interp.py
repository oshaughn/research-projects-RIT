#!/usr/bin/env python3
"""test_q_window_interp.py -- accuracy of the Q(t) sub-sample interpolation stencils.

Q^a_lm(t) is the inverse transform of something supported on [fmin, fmax], so it is BAND-LIMITED,
and it is sampled at 1/deltaT -- usually far above the Nyquist rate its own band requires.  This
test builds a signal with exactly that property, samples it, asks each stencil for values at
random sub-sample offsets, and compares against the exact band-limited signal.

What this pins down is the CROSSOVER, because there isn't a uniformly better stencil:

  * 'nearest' is a rounding, not an interpolation: O(1) error everywhere.
  * 'cubic' (4-point Lagrange) has O(h^4) error, so it improves FAST with oversampling and is
    poor near Nyquist.
  * 'sinc' (Lanczos, 2a taps) has window-limited error that is independent of oversampling, so
    it PLATEAUS -- much better than cubic near Nyquist, worse than cubic once heavily
    oversampled.

Asserted: sinc beats cubic by >10x at fNyq/fmax <= 2 (the production regime -- srate 4096 with
fmax ~1700 is ~1.2), cubic beats sinc by the top of the range, and both beat nearest throughout.
A regression that "improved" sinc into winning everywhere would mean the window had been widened
until it was no longer a local stencil, so the crossover is asserted in BOTH directions.

Self-contained: numpy only, no LAL, no data.  Runs in about a second.

    python3 test_q_window_interp.py
"""
from __future__ import print_function

import numpy as np

from RIFT.likelihood.factored_likelihood import (
    _cubic_Q_window_numpy,
    _nearest_Q_window_numpy,
    _sinc_Q_window_numpy,
)


def band_limited_signal(n_time, n_lm, oversample, seed=1234):
    """A complex signal whose spectrum is zero above n_time/(2*oversample) bins.

    Returned as (samples, evaluate) where evaluate(t) gives the exact continuum value at
    arbitrary real sample coordinate t, by direct evaluation of the Fourier sum -- so the
    comparison is against truth, not against another interpolant.
    """
    rng = np.random.RandomState(seed)
    kmax = int(n_time // (2 * oversample))
    ks = np.arange(-kmax, kmax + 1)
    amps = (rng.randn(len(ks), n_lm) + 1j * rng.randn(len(ks), n_lm)) / np.sqrt(len(ks))

    def evaluate(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        phase = np.exp(2j * np.pi * np.outer(t, ks) / float(n_time))
        return phase.dot(amps)

    return evaluate(np.arange(n_time)), evaluate


def max_rel_error(kind, samples, evaluate, starts, fracs, npts, n_time):
    if kind == "nearest":
        got = _nearest_Q_window_numpy(samples, (np.round(starts + fracs)).astype(int), npts)
    elif kind == "cubic":
        got = _cubic_Q_window_numpy(samples, starts, fracs, npts)
    elif kind == "sinc":
        got = _sinc_Q_window_numpy(samples, starts, fracs, npts)
    else:
        raise ValueError(kind)
    err = 0.0
    scale = np.max(np.abs(samples))
    for i in range(len(starts)):
        t = starts[i] + fracs[i] + np.arange(npts)
        # stay clear of the ends, where every stencil zero-extends
        keep = (t > 32) & (t < n_time - 32)
        if not np.any(keep):
            continue
        err = max(err, np.max(np.abs(got[i][keep] - evaluate(t[keep]))) / scale)
    return err


def main():
    n_time, n_lm, npts = 4096, 2, 24
    rng = np.random.RandomState(7)
    starts = rng.randint(200, n_time - 300, size=6)
    fracs = rng.rand(6)

    print("%-12s %14s %14s %14s" % ("fNyq/fmax", "nearest", "cubic", "sinc(a=8)"))
    err = {}
    for oversample in (1.5, 2, 4, 8, 16):
        samples, evaluate = band_limited_signal(n_time, n_lm, oversample)
        e = {k: max_rel_error(k, samples, evaluate, starts, fracs, npts, n_time)
             for k in ("nearest", "cubic", "sinc")}
        err[oversample] = e
        print("%-12s %14.3e %14.3e %14.3e"
              % (oversample, e["nearest"], e["cubic"], e["sinc"]))
        assert e["cubic"] < e["nearest"], "cubic must beat nearest at fNyq/fmax=%s" % oversample
        assert e["sinc"] < e["nearest"], "sinc must beat nearest at fNyq/fmax=%s" % oversample

    # Near Nyquist -- the production regime -- sinc must win, and by a lot.
    for oversample in (1.5, 2):
        gain = err[oversample]["cubic"] / err[oversample]["sinc"]
        print("  fNyq/fmax=%s: sinc is %.0fx better than cubic" % (oversample, gain))
        assert gain > 10, "sinc must beat cubic by >10x at fNyq/fmax=%s (got %.1fx)" % (
            oversample, gain)

    # Heavily oversampled, cubic's h^4 wins: assert that too, so nobody "fixes" sinc into
    # winning everywhere by quietly widening the window past a local stencil.
    assert err[16]["cubic"] < err[16]["sinc"], (
        "cubic should win at fNyq/fmax=16 (%g vs %g) -- if this fails the stencil is no longer "
        "local" % (err[16]["cubic"], err[16]["sinc"]))
    print("  fNyq/fmax=16: cubic is %.0fx better than sinc, as expected"
          % (err[16]["sinc"] / err[16]["cubic"]))

    # At integer offsets every stencil must reproduce the samples exactly.
    samples, _ = band_limited_signal(n_time, n_lm, 8)
    exact = _sinc_Q_window_numpy(samples, starts, np.zeros(len(starts)), npts)
    for i, s0 in enumerate(starts):
        assert np.allclose(exact[i], samples[s0:s0 + npts], atol=1e-12), \
            "sinc must be the identity at zero fractional offset"
    print("zero-offset identity: OK")

    # Weights must sum to one for any offset, so a constant is interpolated exactly.
    from RIFT.likelihood.factored_likelihood import _sinc_lanczos_weights
    for u in (0.0, 0.1, 0.5, 0.9, 0.999):
        _, w = _sinc_lanczos_weights(u)
        assert abs(w.sum() - 1.0) < 1e-12, "weights must sum to 1 at u=%g" % u
    print("partition of unity: OK")
    print("\nPASS")


if __name__ == "__main__":
    main()
