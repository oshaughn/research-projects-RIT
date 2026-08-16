#!/usr/bin/env python3
"""test_time_interp_choice -- the pipeline's automatic Q_lm stencil selection.

Guards four things that a run's accuracy depends on and that nothing else would catch:

  1. Both thresholds sit inside the MEASURED ambiguous band, and the two regimes that actually
     occur in this tree land where they should -- production (srate 4096, fmax 1700) on 'sinc',
     the heavily-oversampled slow-rotation brute-force configuration on 'cubic', on BOTH
     backends (the threshold split must not reach the regime production runs in).
  2. The GPU threshold is the looser one, which follows from sinc costing ~2x cubic there
     against ~4.5x on CPU -- and there is a regime where the backend really changes the answer,
     so the distinction is load-bearing rather than decorative.
  3. Bad inputs fall back to 'cubic', never to the more expensive stencil.
  4. The legacy '--internal-ile-interpolate-time True' spelling still means "choose for me",
     so existing invocations keep working.

Self-contained: numpy only, runs instantly.

    python3 test_time_interp_choice.py
"""
from __future__ import print_function

from RIFT.likelihood.time_interp_choice import (
    INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU,
    INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU,
    choose_time_interp_stencil,
    interp_time_threshold,
    is_auto_request,
)


def test_thresholds_match_measured_crossover():
    """Measured (24 seeds x 8 targets): median crossover fNyq/fmax ~= 5.4; sinc wins in EVERY
    realization up to 4.5 and essentially never above 5.75.

    Both thresholds must sit in [4.5, 5.75]: below 4.5 we would drop sinc while it still wins
    every seed, above 5.75 we would keep it where it has already lost.  Re-measure and update
    the table in time_interp_choice.py rather than widening this bound.
    """
    for name, thr in (("CPU", INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU),
                      ("GPU", INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU)):
        assert 4.5 <= thr <= 5.75, (
            "%s threshold %g is outside the measured ambiguous band [4.5, 5.75]" % (name, thr))
        print("%s threshold %g inside measured ambiguous band [4.5, 5.75]: OK" % (name, thr))


def test_gpu_threshold_is_the_looser_one():
    """The GPU tolerates sinc further out, because there it costs ~2x rather than ~4.5x.

    The ORDERING is the claim, and it follows from the measured cost ratio: cost only breaks the
    near-crossover tie, so the backend where sinc is cheaper should keep it longer.  A change
    that inverted this would mean the cost measurement had been misread.
    """
    assert INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU >= INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU, (
        "GPU threshold (%g) must not be below the CPU one (%g): sinc is ~2x cubic on GPU against "
        "~4.5x on CPU, so cost should break the tie LATER on GPU, not earlier"
        % (INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU, INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU))
    assert interp_time_threshold(on_gpu=True) == INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU
    assert interp_time_threshold(on_gpu=False) == INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU

    # There must be a regime where the backend actually changes the answer, or the whole
    # distinction is decorative and should be removed rather than maintained.
    if INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU > INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU:
        mid = 0.5 * (INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU
                     + INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU)
        srate = 4096
        fmax = (srate / 2.0) / mid
        s_cpu, _, _ = choose_time_interp_stencil(srate, fmax, on_gpu=False)
        s_gpu, _, _ = choose_time_interp_stencil(srate, fmax, on_gpu=True)
        assert (s_cpu, s_gpu) == ('cubic', 'sinc'), \
            "at fNyq/fmax=%.2f expected CPU->cubic, GPU->sinc, got %r/%r" % (mid, s_cpu, s_gpu)
        print("at fNyq/fmax=%.2f: CPU->%s, GPU->%s (backend changes the answer): OK"
              % (mid, s_cpu, s_gpu))


def test_real_configurations():
    """The configurations that actually occur in this tree -- on BOTH backends."""
    for on_gpu in (False, True):
        tag = "GPU" if on_gpu else "CPU"
        # production: fNyq/fmax ~ 1.2, where sinc is 35-50x more accurate.  Both backends must
        # agree here: the threshold split must not reach the regime production actually runs in.
        stencil, ov, thr = choose_time_interp_stencil(4096, 1700, on_gpu=on_gpu)
        print("[%s] srate 4096, fmax 1700 -> fNyq/fmax=%.2f (thr %g) -> %s"
              % (tag, ov, thr, stencil))
        assert stencil == 'sinc', "near-Nyquist production must get sinc on %s, got %r" % (
            tag, stencil)
        assert abs(ov - 4096 / 2.0 / 1700) < 1e-12

        # slow-rotation brute-force tests: fmax 512 at srate 16384, i.e. 16 -- cubic's regime
        stencil, ov, _ = choose_time_interp_stencil(16384, 512, on_gpu=on_gpu)
        print("[%s] srate 16384, fmax 512 -> fNyq/fmax=%.2f -> %s" % (tag, ov, stencil))
        assert stencil == 'cubic', "heavily oversampled must get cubic on %s, got %r" % (
            tag, stencil)

        # a run right at the backend's own threshold takes cubic (the cheaper stencil)
        stencil, _, _ = choose_time_interp_stencil(
            4096, 2048 / interp_time_threshold(on_gpu), on_gpu=on_gpu)
        assert stencil == 'cubic', "at the threshold exactly, the cheaper stencil must win"
    print("exactly at threshold -> cubic on both backends: OK")


def test_bad_inputs_fall_back_to_cubic():
    """Nothing malformed may select the expensive stencil by accident, on either backend."""
    for on_gpu in (False, True):
        for srate, fmax in ((None, 1700), (4096, None), (4096, 0), (0, 1700),
                            ('nonsense', 1700), (4096, -100), (float('nan'), 1700),
                            (float('inf'), 1700)):
            stencil, ov, thr = choose_time_interp_stencil(srate, fmax, on_gpu=on_gpu)
            assert stencil == 'cubic', \
                "srate=%r fmax=%r must fall back to cubic, got %r" % (srate, fmax, stencil)
            # the threshold must still be reported, or the caller's log line cannot be written
            assert thr == interp_time_threshold(on_gpu)
    print("malformed srate/fmax fall back to cubic on both backends: OK")

    # ...but a valid pair must NOT report None for the factor, or the log line lies
    _, ov, _ = choose_time_interp_stencil(4096, 1700)
    assert ov is not None


def test_legacy_true_still_means_auto():
    """Backward compatibility: existing invocations pass a bare flag or the literal 'True'."""
    for v in ('True', 'true', 'TRUE', '1', 'yes', 'auto', ' True '):
        assert is_auto_request(v), "%r must request automatic selection" % v
    for v in ('nearest', 'cubic', 'sinc', 'False'):
        assert not is_auto_request(v), "%r must be passed through, not auto-selected" % v
    print("legacy 'True' means auto; explicit stencil names pass through: OK")


if __name__ == "__main__":
    test_thresholds_match_measured_crossover()
    test_gpu_threshold_is_the_looser_one()
    test_real_configurations()
    test_bad_inputs_fall_back_to_cubic()
    test_legacy_true_still_means_auto()
    print("\nPASS")
