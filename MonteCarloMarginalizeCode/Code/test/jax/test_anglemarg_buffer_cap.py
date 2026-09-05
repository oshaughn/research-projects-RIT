#!/usr/bin/env python3
# Registered by NAME in .travis/test-jax.sh's FILES array -- that job selects by an
# explicit list, not by a marker.  A '# RIFT-CI-GATE:' line here would name a gate
# that does not exist and the roster census refuses it, correctly.
"""The anglemarg eval-chunk cap: still bounds the buffer, no longer assumes 4 GiB.

The cap exists because on 2026-08-28 the laplace path asked XLA for a single 36.41 GiB
buffer at chunk 4000 / npts 1193 and died RESOURCE_EXHAUSTED against a 25 GiB cgroup.
Making the target device-aware must not weaken that: these tests pin the bound itself,
not the constant that used to express it.  Device-aware means AVAILABLE memory, not the
allocator's capacity ceiling -- these GPUs are shared, and a ceiling-sized allowance on a
card someone else is already holding is the same OOM with a nicer derivation.
"""
from __future__ import print_function
import pytest

sam = pytest.importorskip("RIFT.likelihood.jax_ile.samplers")


class _Data(object):
    def __init__(self, npts): self.npts = npts


class _Like(object):
    def __init__(self, scheme, npts):
        self.angle_marg_scheme = scheme
        self.data = _Data(npts)


def _target(monkeypatch, byts):
    monkeypatch.setattr(sam, "_angle_marg_buffer_target", lambda: byts)


def test_the_original_blowup_is_still_refused(monkeypatch):
    """chunk 4000 at npts 1193 must not survive at the historical 4 GiB target."""
    _target(monkeypatch, 4 << 30)
    got = sam.angle_marg_eval_chunk(_Like("laplace", 1193), 4000)
    assert got < 4000
    # the buffer the returned chunk implies must fit the target
    assert got * sam._ANGLE_MARG_BYTES_PER_SAMPLE_PT * 1193 <= (4 << 30)


@pytest.mark.parametrize("target", [4 << 30, 12 << 30, 24 << 30])
def test_the_bound_holds_at_every_target(monkeypatch, target):
    """Whatever the device reports, the implied buffer never exceeds it."""
    _target(monkeypatch, target)
    for npts in (614, 1193, 4915, 32769):
        got = sam.angle_marg_eval_chunk(_Like("laplace", npts), 4000)
        assert got >= 1
        assert got * sam._ANGLE_MARG_BYTES_PER_SAMPLE_PT * npts <= target


def test_a_bigger_device_lifts_the_throttle(monkeypatch):
    """The point of the change: 4 GiB caps production npts below the nominal chunk."""
    npts = 1230
    _target(monkeypatch, 4 << 30)
    small = sam.angle_marg_eval_chunk(_Like("laplace", npts), 1000)
    _target(monkeypatch, 16 << 30)
    big = sam.angle_marg_eval_chunk(_Like("laplace", npts), 1000)
    assert small < 1000, "4 GiB should still throttle at production npts"
    assert big == 1000, "a 16 GiB device should not throttle at all"


def test_grid_is_never_capped(monkeypatch):
    """`grid` is a sentinel for 'no dense angle scheme' and must pass through."""
    _target(monkeypatch, 4 << 30)
    assert sam.angle_marg_eval_chunk(_Like("grid", 32769), 4000) == 4000


# ---------------------------------------------------------------------------
# Everything above stubs `_angle_marg_buffer_target` via `_target()`, which is right
# for testing the BOUND but means none of it touches the probe itself.  An earlier
# revision of this file "covered" the probe with
#     monkeypatch.setattr(s, "_angle_marg_buffer_target", lambda: FALLBACK)
#     assert s._angle_marg_buffer_target() == (4 << 30)
# which replaces the function under test with a lambda and then asserts the lambda
# returns what it was written to return.  It passes against ANY implementation,
# including no implementation.  What follows drives the real function by faking the
# device, so the probe fails when the probe is wrong.
# ---------------------------------------------------------------------------


class _Dev(object):
    """Minimal stand-in for a jax Device.

    `limit` is the allocator's CAPACITY CEILING and, deliberately, is not enough on its
    own for the probe to size anything.  The earlier version of this class modelled only
    a limit, which is why it could not see the review finding below: every fake device it
    built was an idle one, so a ceiling and free memory were the same number and treating
    one as the other looked correct.  `free` (largest servable block) and `pool`/`in_use`
    are what say how much of the ceiling is actually obtainable.
    """
    def __init__(self, platform, limit=None, key="bytes_limit",
                 free=None, pool=None, in_use=None):
        self.platform = platform
        self._limit = limit
        self._key = key
        self._free = free
        self._pool = pool
        self._in_use = in_use

    def memory_stats(self):
        stats = {}
        if self._limit is not None:
            stats[self._key] = self._limit
        if self._free is not None:
            stats["largest_free_block_bytes"] = self._free
        if self._pool is not None:
            stats["pool_bytes"] = self._pool
        if self._in_use is not None:
            stats["bytes_in_use"] = self._in_use
        return stats


def _idle_gpu(total):
    """A card of `total` bytes with nobody else on it: ceiling AND free both `total`."""
    return _Dev("gpu", total, free=total)


def _fake_jax(monkeypatch, devices=None, raises=None):
    """Install a fake `jax` module that the probe's local `import jax` will find."""
    import sys
    import types
    mod = types.ModuleType("jax")
    if raises is not None:
        def devs():
            raise raises
    else:
        def devs():
            return list(devices)
    mod.devices = devs
    monkeypatch.setitem(sys.modules, "jax", mod)
    return mod


GIB = 1 << 30


def test_probe_failure_falls_back_to_four_gib(monkeypatch):
    """A device we cannot interrogate must behave exactly as before -- never larger."""
    _fake_jax(monkeypatch, raises=RuntimeError("no device"))
    assert sam._angle_marg_buffer_target() == 4 * GIB


def test_no_gpu_falls_back_to_four_gib(monkeypatch):
    """CPU-only: nothing to be device-aware about."""
    _fake_jax(monkeypatch, devices=[_Dev("cpu", 999 * GIB, free=999 * GIB)])
    assert sam._angle_marg_buffer_target() == 4 * GIB


def test_empty_memory_stats_falls_back_to_four_gib(monkeypatch):
    """A GPU whose runtime reports nothing is a probe failure, not a zero limit."""
    _fake_jax(monkeypatch, devices=[_Dev("gpu", None)])
    assert sam._angle_marg_buffer_target() == 4 * GIB


def test_the_gpu_is_picked_out_of_a_mixed_device_list(monkeypatch):
    """The platform filter must actually select, not just happen to be index 0."""
    _fake_jax(monkeypatch, devices=[_Dev("cpu", 999 * GIB, free=999 * GIB),
                                    _idle_gpu(24 * GIB)])
    assert sam._angle_marg_buffer_target() == 12 * GIB


def test_the_reservable_limit_still_clamps_when_bytes_limit_is_absent(monkeypatch):
    """The alternate ceiling spelling is still read -- but only downward.

    A runtime reporting a free block larger than its own allocator limit is misreporting;
    the ceiling may shrink the allowance, never license one.
    """
    _fake_jax(monkeypatch,
              devices=[_Dev("gpu", 8 * GIB, key="bytes_reservable_limit",
                            free=999 * GIB)])
    assert sam._angle_marg_buffer_target() == 4 * GIB


def test_the_fraction_is_applied_to_available_memory(monkeypatch):
    monkeypatch.setattr(sam, "_ANGLE_MARG_BUFFER_FRACTION", 0.25)
    _fake_jax(monkeypatch, devices=[_idle_gpu(24 * GIB)])
    assert sam._angle_marg_buffer_target() == 6 * GIB


@pytest.mark.parametrize("free_gib", [1, 2, 4, 6, 8, 16, 24, 80])
def test_the_allowance_never_exceeds_what_is_actually_free(monkeypatch, free_gib):
    """THE regression this file exists for after review.

    An earlier revision returned max(4 GiB, limit * fraction).  On a 6 GiB card that is
    4 GiB -- two thirds of the whole device for ONE buffer -- and on anything under 4 GiB
    it hands out more memory than exists.  4 GiB is the answer for a device we cannot
    SEE; it is not a safe minimum for a device we can.
    """
    _fake_jax(monkeypatch, devices=[_idle_gpu(free_gib * GIB)])
    got = sam._angle_marg_buffer_target()
    assert got <= free_gib * GIB, "allowance exceeds what the device reports free"
    assert got == int(free_gib * GIB * sam._ANGLE_MARG_BUFFER_FRACTION)


def test_a_small_device_is_not_floored_at_four_gib(monkeypatch):
    """Stated separately from the sweep so the failure names the defect."""
    _fake_jax(monkeypatch, devices=[_idle_gpu(6 * GIB)])
    assert sam._angle_marg_buffer_target() == 3 * GIB


# --- the ceiling is not free memory (review P1, second round) ----------------
# Every fake device above this line was IDLE, so its ceiling and its free memory were
# the same number and a probe that read either looked correct.  The cards this runs on
# are shared: a survey of the interactive hosts found 24 GiB GPUs with 18-22 GiB already
# held by other processes.  `bytes_limit` does not move when that happens.


def test_a_busy_shared_card_is_not_sized_from_its_ceiling(monkeypatch):
    """24 GiB ceiling, 22 GiB held by someone else, 2 GiB actually free.

    Sizing off the ceiling returns a 12 GiB allowance here -- six times what the card
    has left -- and walks straight back into the RESOURCE_EXHAUSTED this cap exists to
    prevent.  The allowance must come from the 2 GiB, not the 24.
    """
    _fake_jax(monkeypatch, devices=[_Dev("gpu", 24 * GIB, free=2 * GIB)])
    got = sam._angle_marg_buffer_target()
    assert got < 12 * GIB, "allowance still derived from the capacity ceiling"
    assert got <= 2 * GIB, "allowance exceeds the memory that is actually free"
    assert got == 1 * GIB


def test_a_zero_largest_free_block_is_a_known_full_device(monkeypatch):
    """A reported zero is availability data, not a missing probe result."""
    _fake_jax(monkeypatch, devices=[_Dev("gpu", 24 * GIB, free=0)])
    assert sam._angle_marg_buffer_target() == 0
    with pytest.raises(MemoryError):
        sam.angle_marg_eval_chunk(_Like("laplace", 1193), 4000)


def test_a_ceiling_with_no_free_report_falls_back_rather_than_guessing_up(monkeypatch):
    """The device is visible but says nothing about occupancy.

    This is the shape the old fake device had, and the answer is NOT half the ceiling:
    a limit alone cannot distinguish an idle card from a full one.  Fall back to the
    conservative 4 GiB and let the operator assert otherwise with the absolute override.
    """
    _fake_jax(monkeypatch, devices=[_Dev("gpu", 24 * GIB)])
    assert sam._angle_marg_buffer_target() == 4 * GIB


def test_the_reserved_pool_minus_what_we_hold_is_used_when_no_block_is_reported(
        monkeypatch):
    """Second-choice availability signal: memory already reserved to us is genuinely
    ours, unlike the ceiling, so pool - in_use is a real free figure."""
    _fake_jax(monkeypatch,
              devices=[_Dev("gpu", 24 * GIB, pool=16 * GIB, in_use=4 * GIB)])
    assert sam._angle_marg_buffer_target() == 6 * GIB


def test_a_full_pool_yields_no_allowance_and_the_eval_refuses(monkeypatch):
    """Nothing free is a READING, not a failure to read.

    Falling back to the 4 GiB guess here would hand out memory the runtime has just said
    does not exist, so the target goes to zero and the eval refuses with the message that
    names the knobs -- an outage the operator can act on, not a silent OOM later.
    """
    _fake_jax(monkeypatch,
              devices=[_Dev("gpu", 24 * GIB, pool=24 * GIB, in_use=24 * GIB)])
    assert sam._angle_marg_buffer_target() == 0
    with pytest.raises(MemoryError):
        sam.angle_marg_eval_chunk(_Like("laplace", 1193), 4000)
    # and the sentinel still short-circuits before any of it
    assert sam.angle_marg_eval_chunk(_Like("grid", 1193), 4000) == 4000


# --- the advertised override ------------------------------------------------

def test_the_default_fraction_applies_when_unset():
    assert sam._read_buffer_fraction({}) == 0.5


@pytest.mark.parametrize("raw,expect", [("0.8", 0.8), ("1.0", 1.0), ("0.25", 0.25)])
def test_a_usable_override_is_honoured(raw, expect):
    assert sam._read_buffer_fraction(
        {"RIFT_ANGLEMARG_BUFFER_FRACTION": raw}) == expect


@pytest.mark.parametrize("raw", ["", "half", "0.5x", "1.5", "2", "0", "-0.5", "nan"])
def test_an_unusable_override_is_refused_loudly(raw):
    """Refused, NOT silently replaced by the default.

    A value above 1 asks for a buffer bigger than the device reports, i.e. asks this
    code to cause the OOM it exists to prevent.  A value at or below 0 bounds nothing.
    Either way the caller believes a bound is in force, so failing quietly is worse
    than failing.
    """
    with pytest.raises(ValueError):
        sam._read_buffer_fraction({"RIFT_ANGLEMARG_BUFFER_FRACTION": raw})


# --- the constant the whole bound rests on ----------------------------------

def test_bytes_per_sample_point_still_reproduces_the_observed_allocation():
    """Pin _ANGLE_MARG_BYTES_PER_SAMPLE_PT against a number the code does not own.

    FOUND BY MUTATION, and it is why this test exists: halving the constant
    8192 -> 4096 left all 33 other tests in this file passing.  Every one of them
    computes the expected buffer as `got * sam._ANGLE_MARG_BYTES_PER_SAMPLE_PT * npts`
    -- reading the same constant the production code reads -- so the assertion is
    self-consistent for ANY value of it.  The bound would silently permit a buffer
    twice the intended size and the suite would stay green.

    The independent reference is XLA's own report from 2026-08-28: at chunk 4000
    and npts 1193 the laplace path asked for a single buffer of 36.41 GiB.  8192
    reproduces that to 0.01%.  This is an EXTERNAL measurement, not a restatement
    of the constant, so it fails when the constant moves.
    """
    observed_gib = 36.41          # from the RESOURCE_EXHAUSTED message itself
    chunk, npts = 4000, 1193      # the configuration that produced it
    implied = chunk * npts * sam._ANGLE_MARG_BYTES_PER_SAMPLE_PT / float(GIB)
    assert abs(implied / observed_gib - 1.0) < 0.01, (
        "%d bytes/sample-point implies a %.2f GiB buffer at chunk %d / npts %d, but "
        "the allocation this cap was built from was %.2f GiB.  If the per-point size "
        "genuinely changed, re-measure it and update BOTH the constant and this "
        "reference." % (sam._ANGLE_MARG_BYTES_PER_SAMPLE_PT, implied, chunk, npts,
                        observed_gib))


# ---------------------------------------------------------------------------
# THE BOUND WAS NOT ACTUALLY A BOUND: max(1, target // per_sample)
#
# Review P1 on #250.  Every assertion above stubs a target that is comfortably larger
# than one sample, so none of them can reach the floor.  Once ONE sample costs more than
# the target, `max(1, ...)` returns a chunk of 1 and the buffer that chunk implies is
# `bytes_per * npts` -- over the target, by construction.  The floor turned "we cannot
# meet the bound" into "here is a chunk", silently.
#
# Two rules for the tests below, both learned on this file:
#   * do NOT express the expected buffer as `got * sam._ANGLE_MARG_BYTES_PER_SAMPLE_PT`.
#     That reads the same constant production reads and is self-consistent for any value
#     of it -- the mistake the last section of this file documents.  Targets here are
#     explicit literals and the peak-local slab is written out as an explicit literal.
#   * the peak-local dimensions are the REVIEWER'S worked example, checked against the
#     kernel rather than taken on faith: PHI_CHUNK_DEFAULT=16, n_x=256, 4 cells,
#     U_NODE_STREAM_CHUNK=8 live nodes, 8 bytes -> 1048576 bytes per sample-time-point.
# ---------------------------------------------------------------------------

import numpy as np

#: Streamed body plus the stacked phi-scan output at the production floor:
#: 16*256*4*8*8 + 352*256*8.
PEAKLOCAL_BYTES_PER_PT = 1769472
#: ... and one sample of it at production npts=1230.  2.027 GiB.
PEAKLOCAL_ONE_SAMPLE = 2176450560


class _PeakLocalLike(object):
    """peak-local at the production dimensions of the review's worked example."""
    def __init__(self, npts=1230, n_x=256, amp_sizing=None):
        self.angle_marg_scheme = "peak-local"
        self.data = _Data(npts)
        self.x_grid = np.zeros(n_x)
        self.angle_marg_info = {"amp_sizing": amp_sizing}


def test_the_peak_local_slab_really_is_that_big():
    """Pin the reviewer's dimension model against the kernel's own constants.

    This is the number the P1 finding rests on, and it is NOT the module's
    8192 bytes/sample-point: that constant models the DENSE (exact/laplace) path and
    peak-local overrides it upward with max().  Both are right, for different schemes;
    the tension in the review was between a peak-local figure and a laplace constant.
    """
    from RIFT.likelihood.jax_ile import joint_anglemarg_peaklocal as jp
    from RIFT.likelihood.jax_ile import anglemarg
    streamed = jp.PHI_CHUNK_DEFAULT * 256 * 4 * jp.U_NODE_STREAM_CHUNK * 8
    n_phi = jp.required_n_phi(anglemarg.ANGLE_MARG_CROSSOVER_AMPLITUDE,
                              m_max=2)
    modeled = streamed + n_phi * 256 * 8
    assert modeled == PEAKLOCAL_BYTES_PER_PT, (
        "the peak-local live-slab model moved: kernel constants now imply %d bytes per "
        "sample-time-point, the P1 review example assumed %d" % (modeled,
                                                                 PEAKLOCAL_BYTES_PER_PT))
    assert PEAKLOCAL_BYTES_PER_PT * 1230 == PEAKLOCAL_ONE_SAMPLE


def test_one_sample_over_the_target_is_refused_not_floored(monkeypatch):
    """THE P1 REGRESSION.  A 2 GiB card at the default fraction 0.5 gives a 1 GiB
    allowance; one peak-local sample at production dimensions is 1.20 GiB.  The old
    code returned chunk 1 and therefore a 1.20 GiB buffer -- over a bound it claimed to
    enforce.  It must refuse."""
    _target(monkeypatch, 1 << 30)              # explicit literal, not a code constant
    assert PEAKLOCAL_ONE_SAMPLE > (1 << 30)    # the premise, stated in literals
    with pytest.raises(MemoryError):
        sam.angle_marg_eval_chunk(_PeakLocalLike(), 4000)


def test_the_refusal_names_what_the_user_can_change(monkeypatch):
    """A bound that fails closed with a bare assertion is a different outage from one
    that says which knob to turn.  Pin the actionable content, not the wording."""
    _target(monkeypatch, 1 << 30)
    with pytest.raises(MemoryError) as ei:
        sam.angle_marg_eval_chunk(_PeakLocalLike(), 4000)
    msg = str(ei.value)
    for token in ("peak-local", "npts=1230",
                  str(PEAKLOCAL_ONE_SAMPLE), str(PEAKLOCAL_BYTES_PER_PT),
                  str(1 << 30),
                  "RIFT_ANGLEMARG_BUFFER_FRACTION", "RIFT_ANGLEMARG_BUFFER_BYTES"):
        assert token in msg, "refusal does not mention %r:\n%s" % (token, msg)


@pytest.mark.parametrize("npts", [1193, 1230, 4915, 32769])
def test_no_returned_chunk_ever_exceeds_the_target(monkeypatch, npts):
    """The invariant, measured against a per-point size the module does NOT own.

    36.41 GiB at chunk 4000 / npts 1193 is XLA's own report from 2026-08-28, so this
    checks the returned chunk against an EXTERNAL measurement rather than against
    _ANGLE_MARG_BYTES_PER_SAMPLE_PT.  Either the call refuses, or the chunk it returns
    implies a buffer inside the target -- there is no third outcome, and the old floor
    produced exactly that third outcome.
    """
    xla_bytes_per_pt = 36.41 * GIB / (4000 * 1193)
    for target in (1 << 20, 8 << 20, 1 << 30, 4 << 30, 24 << 30):
        _target(monkeypatch, target)
        try:
            got = sam.angle_marg_eval_chunk(_Like("laplace", npts), 4000)
        except MemoryError:
            # refusing is allowed ONLY when one sample genuinely does not fit
            assert xla_bytes_per_pt * npts > target * 1.01, (
                "refused at target %d although one sample is only ~%.0f bytes"
                % (target, xla_bytes_per_pt * npts))
            continue
        assert got >= 1
        implied = got * xla_bytes_per_pt * npts
        assert implied <= target * 1.01, (
            "chunk %d at npts %d implies ~%.2f GiB against a %.2f GiB target"
            % (got, npts, implied / GIB, target / float(GIB)))


def test_a_sample_that_exactly_fills_the_target_is_allowed(monkeypatch):
    """The boundary, so `>` cannot quietly become `>=`.

    Exactly at the allowance the bound IS met, at a chunk of one.  A refusal here would
    be over-tight and would take out a configuration that fits.
    """
    _target(monkeypatch, PEAKLOCAL_ONE_SAMPLE)
    assert sam.angle_marg_eval_chunk(_PeakLocalLike(), 4000) == 1
    _target(monkeypatch, PEAKLOCAL_ONE_SAMPLE - 1)
    with pytest.raises(MemoryError):
        sam.angle_marg_eval_chunk(_PeakLocalLike(), 4000)


def test_the_dense_schemes_reach_the_refusal_too(monkeypatch):
    """Not a peak-local special case: any scheme whose sample outgrows the allowance."""
    _target(monkeypatch, 1 << 20)
    for scheme in ("exact", "laplace"):
        with pytest.raises(MemoryError):
            sam.angle_marg_eval_chunk(_Like(scheme, 32769), 4000)
    # and the sentinel still short-circuits before any of this
    assert sam.angle_marg_eval_chunk(_Like("grid", 32769), 4000) == 4000


# --- the absolute allowance override, which is what makes the refusal actionable ----
# Failing closed against _ANGLE_MARG_BUFFER_TARGET_FALLBACK would be failing closed
# against a number the file itself calls a guess with no guarantee, on exactly the
# machines whose device we could not read.  RIFT_ANGLEMARG_BUFFER_FRACTION cannot help
# there -- it is a fraction of a limit that path never obtained.

def test_no_bytes_override_means_none():
    assert sam._read_buffer_bytes({}) is None


@pytest.mark.parametrize("raw,expect", [("1073741824", 1 << 30),
                                        ("2e9", 2000000000),
                                        ("12884901888", 12 << 30)])
def test_a_usable_bytes_override_is_honoured(raw, expect):
    assert sam._read_buffer_bytes({"RIFT_ANGLEMARG_BUFFER_BYTES": raw}) == expect


@pytest.mark.parametrize("raw", ["", "lots", "4GiB", "0", "-1", "nan", "inf"])
def test_an_unusable_bytes_override_is_refused_loudly(raw):
    with pytest.raises(ValueError):
        sam._read_buffer_bytes({"RIFT_ANGLEMARG_BUFFER_BYTES": raw})


def test_the_bytes_override_beats_the_device_probe(monkeypatch):
    """It has to win over the probe, or it cannot rescue a machine the probe misreads.

    The fake card is idle, so the probe would otherwise answer 12 GiB: the 3 GiB below
    is the override winning, not the fallback coinciding with it.
    """
    _fake_jax(monkeypatch, devices=[_idle_gpu(24 * GIB)])
    monkeypatch.setenv("RIFT_ANGLEMARG_BUFFER_BYTES", str(3 * GIB))
    assert sam._angle_marg_buffer_target() == 3 * GIB


def test_the_bytes_override_beats_the_fallback_and_lifts_a_refusal(monkeypatch):
    """The case the knob exists for: no readable device, and the 4 GiB guess refuses a
    configuration the operator knows their machine can hold."""
    _fake_jax(monkeypatch, raises=RuntimeError("no device"))
    big = _PeakLocalLike(npts=8192)         # 13.5 GiB per sample, over the 4 GiB guess
    with pytest.raises(MemoryError):
        sam.angle_marg_eval_chunk(big, 4000)
    monkeypatch.setenv("RIFT_ANGLEMARG_BUFFER_BYTES", str(32 * GIB))
    assert sam.angle_marg_eval_chunk(big, 4000) == 2


# ---------------------------------------------------------------------------
# MUTATION SWEEP of the section above (2026-09-05, 57 collected).  Each mutation was
# applied to a pristine copy of samplers.py, verified present in the FILE ON DISK
# before running -- a replacement that changes no bytes reports as a surviving guard
# and is a harness bug, not a result -- and reverted afterwards.
#
#   restore the pre-fix `cap = max(1, target // per_sample)`   9 failed  KILLED
#   `>` -> `>=` in the refusal                                 1 failed  KILLED
#   drop the peak-local slab model (use the 8192 constant)     4 failed  KILLED
#   make RIFT_ANGLEMARG_BUFFER_BYTES inert                    13 failed  KILLED
#   read that override inside the probe's blanket except       1 failed  KILLED
#   strip the override names out of the refusal message        1 failed  KILLED
#   accept a zero/negative absolute allowance                  2 failed  KILLED
#   put max(1, ...) back AROUND the surviving division         0 failed  SURVIVED
#
# The survivor is an EQUIVALENT mutant, and it is recorded rather than chased: the
# refusal above guarantees `per_sample <= target` on every path that reaches the
# division, so `target // per_sample` is already >= 1 and the floor cannot change any
# value.  It is the floor REPLACING the refusal (the first row) that was the defect,
# not the floor as such.  No test can distinguish an unreachable branch, and writing
# one that appeared to would mean the refusal had a hole.
# ---------------------------------------------------------------------------


def test_a_malformed_bytes_override_is_not_swallowed_by_the_probe(monkeypatch):
    """It is read OUTSIDE the probe's blanket `except Exception` on purpose: inside it,
    a typo would be silently replaced by the 4 GiB fallback and the operator would never
    learn their override did nothing."""
    _fake_jax(monkeypatch, devices=[_idle_gpu(24 * GIB)])
    monkeypatch.setenv("RIFT_ANGLEMARG_BUFFER_BYTES", "24GiB")
    with pytest.raises(ValueError):
        sam._angle_marg_buffer_target()
