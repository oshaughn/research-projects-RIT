#!/usr/bin/env python3
# Registered by NAME in .travis/test-jax.sh's FILES array -- that job selects by an
# explicit list, not by a marker.  A '# RIFT-CI-GATE:' line here would name a gate
# that does not exist and the roster census refuses it, correctly.
"""The anglemarg eval-chunk cap: still bounds the buffer, no longer assumes 4 GiB.

The cap exists because on 2026-08-28 the laplace path asked XLA for a single 36.41 GiB
buffer at chunk 4000 / npts 1193 and died RESOURCE_EXHAUSTED against a 25 GiB cgroup.
Making the target device-aware must not weaken that: these tests pin the bound itself,
not the constant that used to express it.
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
    """Minimal stand-in for a jax Device."""
    def __init__(self, platform, limit=None, key="bytes_limit"):
        self.platform = platform
        self._limit = limit
        self._key = key

    def memory_stats(self):
        if self._limit is None:
            return {}
        return {self._key: self._limit}


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
    _fake_jax(monkeypatch, devices=[_Dev("cpu", 999 * GIB)])
    assert sam._angle_marg_buffer_target() == 4 * GIB


def test_empty_memory_stats_falls_back_to_four_gib(monkeypatch):
    """A GPU whose runtime reports no limit is a probe failure, not a zero limit."""
    _fake_jax(monkeypatch, devices=[_Dev("gpu", None)])
    assert sam._angle_marg_buffer_target() == 4 * GIB


def test_the_gpu_is_picked_out_of_a_mixed_device_list(monkeypatch):
    """The platform filter must actually select, not just happen to be index 0."""
    _fake_jax(monkeypatch, devices=[_Dev("cpu", 999 * GIB), _Dev("gpu", 24 * GIB)])
    assert sam._angle_marg_buffer_target() == 12 * GIB


def test_the_reservable_limit_is_used_when_bytes_limit_is_absent(monkeypatch):
    _fake_jax(monkeypatch,
              devices=[_Dev("gpu", 24 * GIB, key="bytes_reservable_limit")])
    assert sam._angle_marg_buffer_target() == 12 * GIB


def test_the_fraction_is_applied_to_the_reported_limit(monkeypatch):
    monkeypatch.setattr(sam, "_ANGLE_MARG_BUFFER_FRACTION", 0.25)
    _fake_jax(monkeypatch, devices=[_Dev("gpu", 24 * GIB)])
    assert sam._angle_marg_buffer_target() == 6 * GIB


@pytest.mark.parametrize("limit_gib", [1, 2, 4, 6, 8, 16, 24, 80])
def test_the_allowance_never_exceeds_what_the_device_reports(monkeypatch, limit_gib):
    """THE regression this file exists for after review.

    The reviewed revision returned max(4 GiB, limit * fraction).  On a 6 GiB card that
    is 4 GiB -- two thirds of the whole device for ONE buffer -- and on anything under
    4 GiB it hands out more memory than exists.  4 GiB is the answer for a device we
    cannot SEE; it is not a safe minimum for a device we can.
    """
    _fake_jax(monkeypatch, devices=[_Dev("gpu", limit_gib * GIB)])
    got = sam._angle_marg_buffer_target()
    assert got <= limit_gib * GIB, "allowance exceeds the device's own reported limit"
    assert got == int(limit_gib * GIB * sam._ANGLE_MARG_BUFFER_FRACTION)


def test_a_small_device_is_not_floored_at_four_gib(monkeypatch):
    """Stated separately from the sweep so the failure names the defect."""
    _fake_jax(monkeypatch, devices=[_Dev("gpu", 6 * GIB)])
    assert sam._angle_marg_buffer_target() == 3 * GIB


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
