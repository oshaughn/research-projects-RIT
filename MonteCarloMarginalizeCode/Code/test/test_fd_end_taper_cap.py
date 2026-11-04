#!/usr/bin/env python
"""hlmoft's ChooseFDModes end taper must not reach the merger in long buffers.

The end taper is 1% of the buffer.  With fd_alignment_postevent_time=2 (as ILE
passes) the merger sits 2 s before the buffer end, so an uncapped taper covers it
once the buffer exceeds ~185 s.  fd_end_taper_max_time caps it.  The defect depends
on buffer length, not signal length, so a short BBH in a long buffer exercises it.
"""

import lal
import lalsimulation as lalsim
import numpy as np
import pytest

import RIFT.lalsimutils as lsu


DELTA_T = 1. / 1024


def _params(T):
    P = lsu.ChooseWaveformParams()
    P.m1 = 30 * lal.MSUN_SI
    P.m2 = 25 * lal.MSUN_SI
    P.fmin = 20.
    P.fref = 20.
    P.dist = 400e6 * lal.PC_SI
    P.incl = 0.4
    P.deltaT = DELTA_T
    P.deltaF = 1. / T
    P.approx = lalsim.IMRPhenomXHM
    return P


def _h22(T, **kw):
    hlms = lsu.hlmoft(_params(T), Lmax=2, fd_alignment_postevent_time=2, **kw)
    return hlms[(2, 2)].data.data.copy()


def test_short_buffer_unchanged():
    # 1% of 64 s is below the 1 s cap: output must be identical to the uncapped rule
    np.testing.assert_array_equal(_h22(64), _h22(64, fd_end_taper_max_time=None))


def _energy_vs_short_buffer(T, **kw):
    # Merger sits 2 s before the end of every buffer; the 64 s buffer is unaffected,
    # so compare the last 16 s, where this BBH lives.
    n = int(16 / DELTA_T)
    ref = _h22(64)[-n:]
    h = _h22(T, **kw)[-n:]
    return np.sum(np.abs(h)**2) / np.sum(np.abs(ref)**2)


@pytest.mark.parametrize("T", [512, 2048])
def test_long_buffer_keeps_merger(T):
    assert abs(_energy_vs_short_buffer(T) - 1) < 1e-4
    # the defect the cap removes: the uncapped 1% taper eats the merger
    assert _energy_vs_short_buffer(T, fd_end_taper_max_time=None) < 0.5
