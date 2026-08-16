#!/usr/bin/env python3
"""test_time_interp_choice -- the pipeline's automatic Q_lm stencil selection.

Guards the things a run's accuracy depends on that nothing else would catch:

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
  5. The decision uses the sampling rate the run is ACTUALLY on -- --srate-internal overrides
     deltaT inside ILE and reaches the command line without passing through the helper, and an
     absent --srate means the ILE's own (4x larger) default applies.  Both silently select a
     stencil for a configuration the run never has.
  6. ILE_DEFAULT_SRATE still matches the driver, which is a script and cannot be imported, so
     the duplicated constant is read back out of its source rather than trusted.
  7. An explicit stencil name is validated while the workflow is BUILT, not once per job after
     submission.

Self-contained: numpy only, runs instantly.

    python3 test_time_interp_choice.py
"""
from __future__ import print_function

import os
import re

from RIFT.likelihood.time_interp_choice import (
    ILE_DEFAULT_SRATE,
    INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU,
    INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU,
    TIME_INTERP_CHOICES,
    choose_time_interp_stencil,
    effective_srate_for_stencil,
    interp_time_threshold,
    is_auto_request,
    is_off_request,
    validate_stencil_name,
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


def test_off_spellings_disable_rather_than_raise():
    """'--internal-ile-interpolate-time False' must mean OFF, not "unknown stencil".

    The flag now takes a value, so 'False' arrives as the STRING 'False' -- which is truthy in
    Python.  Without an explicit off-check it sails past the pipeline's `if opts...:` guard and
    is then rejected as a bad stencil name, i.e. the most natural way to spell "turn this off"
    becomes a hard error.  Every value must fall into exactly one of off / auto / stencil.
    """
    for v in ('False', 'false', 'FALSE', '0', 'no', 'off', 'none', ' False '):
        assert is_off_request(v), "%r must mean disabled" % v
        assert not is_auto_request(v), "%r must not also mean auto" % v
    for v in ('True', '1', 'yes', 'auto'):
        assert not is_off_request(v), "%r must not mean disabled" % v
    for v in TIME_INTERP_CHOICES:
        assert not is_off_request(v) and not is_auto_request(v), \
            "%r is a stencil name, neither off nor auto" % v
    print("off / auto / stencil-name are disjoint and exhaustive: OK")


def test_effective_srate_tracks_what_the_run_actually_uses():
    """The decision must use the grid the likelihood is ON, which is not always `srate`.

    Two ways it diverges, both live:
      * --srate-internal overrides deltaT inside ILE and is appended to the ILE command line by
        util_RIFT_pseudo_pipe.py WITHOUT passing through the helper.
      * if the helper emits no --srate, the ILE uses its own default, which is 4x the pipeline's
        usual 4096.
    Getting this wrong silently selects a stencil for a configuration the run never has.
    """
    # plain case: helper emits --srate, no internal override
    assert effective_srate_for_stencil(4096, None, True) == 4096

    # --srate-internal wins, and it must flip the answer in the case that motivated this
    assert effective_srate_for_stencil(4096, 32768, True) == 32768
    s_naive, ov_naive, _ = choose_time_interp_stencil(4096, 1700, on_gpu=True)
    s_true, ov_true, _ = choose_time_interp_stencil(
        effective_srate_for_stencil(4096, 32768, True), 1700, on_gpu=True)
    print("srate 4096 + --srate-internal 32768, fmax 1700: naive fNyq/fmax=%.2f -> %s ; "
          "true fNyq/fmax=%.2f -> %s" % (ov_naive, s_naive, ov_true, s_true))
    assert (s_naive, s_true) == ('sinc', 'cubic'), (
        "the --srate-internal case must change the chosen stencil, or this guard is not "
        "testing the bug it exists for (got %r then %r)" % (s_naive, s_true))

    # no --srate emitted -> ILE's own default, not the pipeline's srate
    assert effective_srate_for_stencil(4096, None, False) == float(ILE_DEFAULT_SRATE)


def test_ile_default_srate_has_not_drifted():
    """ILE_DEFAULT_SRATE duplicates a value in a script that cannot be imported.

    Read it back out of the driver source so the duplication cannot rot silently.  Skipped only
    if the driver is not on disk next to this checkout.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    driver = os.path.normpath(os.path.join(here, '..', '..', 'bin',
                                           'integrate_likelihood_extrinsic_batchmode'))
    if not os.path.isfile(driver):
        print("driver not found at %s, skipping drift check" % driver)
        return
    with open(driver) as f:
        src = f.read()
    m = re.search(r'optp\.add_option\(\s*"--srate"\s*,\s*default\s*=\s*(\d+)', src)
    assert m, "could not find the --srate default in %s; update this test with the driver" % driver
    found = int(m.group(1))
    print("driver --srate default = %d, ILE_DEFAULT_SRATE = %d" % (found, ILE_DEFAULT_SRATE))
    assert found == ILE_DEFAULT_SRATE, (
        "ILE_DEFAULT_SRATE (%d) no longer matches the driver's --srate default (%d); the "
        "pipeline would choose the stencil from the wrong sampling rate whenever the helper "
        "emits no --srate" % (ILE_DEFAULT_SRATE, found))


def test_explicit_stencil_names_are_validated_at_build_time():
    """A typo must fail while the workflow is BUILT, not once per job after submission."""
    for good in ('nearest', 'cubic', 'sinc', ' SINC ', 'Cubic'):
        assert validate_stencil_name(good) in TIME_INTERP_CHOICES
    for bad in ('sinK', 'lanczos', 'Sinc8', '', 'true', 'nearest,cubic'):
        try:
            validate_stencil_name(bad)
        except ValueError:
            continue
        raise AssertionError("validate_stencil_name(%r) must raise" % bad)
    print("explicit stencil names validated, typos rejected: OK")


def test_choices_agree_with_the_likelihood_module():
    """This leaf module duplicates TIME_INTERP_CHOICES to stay import-cheap; keep them in step."""
    from RIFT.likelihood.factored_likelihood import TIME_INTERP_CHOICES as FL_CHOICES
    assert tuple(TIME_INTERP_CHOICES) == tuple(FL_CHOICES), (
        "time_interp_choice.TIME_INTERP_CHOICES %r disagrees with factored_likelihood's %r"
        % (TIME_INTERP_CHOICES, FL_CHOICES))
    print("stencil name lists agree with factored_likelihood: OK")


if __name__ == "__main__":
    test_thresholds_match_measured_crossover()
    test_gpu_threshold_is_the_looser_one()
    test_real_configurations()
    test_bad_inputs_fall_back_to_cubic()
    test_legacy_true_still_means_auto()
    test_off_spellings_disable_rather_than_raise()
    test_effective_srate_tracks_what_the_run_actually_uses()
    test_ile_default_srate_has_not_drifted()
    test_explicit_stencil_names_are_validated_at_build_time()
    test_choices_agree_with_the_likelihood_module()
    print("\nPASS")
