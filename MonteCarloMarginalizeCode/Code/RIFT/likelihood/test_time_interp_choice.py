#!/usr/bin/env python3
"""test_time_interp_choice -- the pipeline's automatic Q_lm stencil selection.

Guards three things that a run's accuracy depends on and that nothing else would catch:

  1. The threshold is on the right side of the MEASURED crossover, and the two regimes that
     actually occur in this tree land where they should -- production (srate 4096, fmax 1700)
     on 'sinc', the heavily-oversampled slow-rotation brute-force configuration on 'cubic'.
  2. Bad inputs fall back to 'cubic', never to the more expensive stencil.
  3. The legacy '--internal-ile-interpolate-time True' spelling still means "choose for me",
     so existing invocations keep working.

Self-contained: numpy only, runs instantly.

    python3 test_time_interp_choice.py
"""
from __future__ import print_function

from RIFT.likelihood.time_interp_choice import (
    INTERP_TIME_OVERSAMPLING_THRESHOLD,
    choose_time_interp_stencil,
    is_auto_request,
)


def test_threshold_matches_measured_crossover():
    """The measured crossover is fNyq/fmax ~= 5.3 (12 seeds, spread bracketing 1.0 over 5-6).

    The threshold must sit inside that band: below 5 sinc wins by >=1.4x at every seed, above 6
    cubic wins by >=1.6x at every seed, so a threshold outside [5, 6] would pick the measurably
    worse stencil in a regime where the answer is not ambiguous.
    """
    assert 5.0 <= INTERP_TIME_OVERSAMPLING_THRESHOLD <= 6.0, (
        "threshold %g is outside the measured ambiguous band [5, 6]; if the stencils or their "
        "accuracy changed, re-measure with test_q_window_interp.py and update the table in "
        "time_interp_choice.py rather than moving this bound"
        % INTERP_TIME_OVERSAMPLING_THRESHOLD)
    print("threshold %g inside measured ambiguous band [5,6]: OK"
          % INTERP_TIME_OVERSAMPLING_THRESHOLD)


def test_real_configurations():
    """The two configurations that actually occur in this tree."""
    # production: fNyq/fmax ~ 1.2, where sinc is 35-50x more accurate
    stencil, ov = choose_time_interp_stencil(4096, 1700)
    print("srate 4096, fmax 1700 -> fNyq/fmax=%.2f -> %s" % (ov, stencil))
    assert stencil == 'sinc', "near-Nyquist production must get sinc, got %r" % stencil
    assert abs(ov - 4096 / 2.0 / 1700) < 1e-12

    # slow-rotation brute-force tests: fmax 512 at srate 16384, i.e. 16 -- cubic's regime
    stencil, ov = choose_time_interp_stencil(16384, 512)
    print("srate 16384, fmax 512 -> fNyq/fmax=%.2f -> %s" % (ov, stencil))
    assert stencil == 'cubic', "heavily oversampled must get cubic, got %r" % stencil

    # a run right at the threshold takes cubic (the cheaper incumbent)
    stencil, _ = choose_time_interp_stencil(4096, 2048 / INTERP_TIME_OVERSAMPLING_THRESHOLD)
    assert stencil == 'cubic', "at the threshold exactly, the cheaper stencil must win"
    print("exactly at threshold -> cubic: OK")


def test_bad_inputs_fall_back_to_cubic():
    """Nothing malformed may select the expensive stencil by accident."""
    for srate, fmax in ((None, 1700), (4096, None), (4096, 0), (0, 1700),
                        ('nonsense', 1700), (4096, -100), (float('nan'), 1700),
                        (float('inf'), 1700)):
        stencil, ov = choose_time_interp_stencil(srate, fmax)
        assert stencil == 'cubic', \
            "srate=%r fmax=%r must fall back to cubic, got %r" % (srate, fmax, stencil)
    print("malformed srate/fmax fall back to cubic: OK")

    # ...but a valid pair must NOT report None for the factor, or the log line lies
    _, ov = choose_time_interp_stencil(4096, 1700)
    assert ov is not None


def test_legacy_true_still_means_auto():
    """Backward compatibility: existing invocations pass a bare flag or the literal 'True'."""
    for v in ('True', 'true', 'TRUE', '1', 'yes', 'auto', ' True '):
        assert is_auto_request(v), "%r must request automatic selection" % v
    for v in ('nearest', 'cubic', 'sinc', 'False'):
        assert not is_auto_request(v), "%r must be passed through, not auto-selected" % v
    print("legacy 'True' means auto; explicit stencil names pass through: OK")


if __name__ == "__main__":
    test_threshold_matches_measured_crossover()
    test_real_configurations()
    test_bad_inputs_fall_back_to_cubic()
    test_legacy_true_still_means_auto()
    print("\nPASS")
