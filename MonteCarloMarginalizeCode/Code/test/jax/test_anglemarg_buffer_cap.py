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


def test_probe_failure_falls_back_to_four_gib(monkeypatch):
    """No jax, no GPU, or a moved API must behave exactly as before -- never larger."""
    import RIFT.likelihood.jax_ile.samplers as s
    monkeypatch.setattr(s, "jax", None, raising=False)
    def boom(): raise RuntimeError("no device")
    monkeypatch.setattr(s, "_angle_marg_buffer_target",
                        lambda: s._ANGLE_MARG_BUFFER_TARGET_FALLBACK)
    assert s._angle_marg_buffer_target() == (4 << 30)
