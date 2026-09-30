"""ComplexOverlap.ip(interpolate_max=True): sub-sample peak of |overlap(t)|.

Two identical unit-norm complex frequency series, one shifted in time by a
non-integer number of samples, have a true match of exactly 1.  The sampled
|overlap(t)| peak falls short of 1; the interpolated peak must recover most of
that deficit and must not overshoot 1.  Shifts include peaks at index 0 and
index N-1, where the stencil wraps around the periodic series.
"""
import numpy as np
import pytest

lal = pytest.importorskip("lal")
lalsimutils = pytest.importorskip("RIFT.lalsimutils")

F_NYQ = 2048.
DELTA_F = 1. / 8.
F_LOW = 20.


def _overlap():
    return lalsimutils.ComplexOverlap(fLow=F_LOW, fNyq=F_NYQ, deltaF=DELTA_F,
                                      interpolate_max=True)


def _series(IP, shift_samples):
    """Unit-norm chirp-like series and a copy delayed by shift_samples*deltaT."""
    n = IP.len2side
    f = (np.arange(n) - n // 2) * DELTA_F
    # One-sided support, as for h+ - i hx: |overlap(t)| is then the envelope.
    af = np.abs(f)
    amp = np.zeros(n)
    band = (f < 0) & (af >= F_LOW)
    amp[band] = af[band] ** (-7. / 6.)
    phase = 2e3 * np.where(band, af, F_LOW) ** (-5. / 3.)
    base = amp * np.exp(1j * phase)
    tau = shift_samples * IP.deltaT
    out = []
    for data in (base, base * np.exp(-2j * np.pi * f * tau)):
        h = lal.CreateCOMPLEX16FrequencySeries("h", lal.LIGOTimeGPS(0.), -F_NYQ,
                                               DELTA_F, lalsimutils.lsu_HertzUnit, n)
        h.data.data[:] = data
        h.data.data[:] /= IP.norm(h)
        out.append(h)
    return out


@pytest.mark.parametrize("shift_samples", [0.5, 0.3, -0.3, -0.7, 17.5, -40.25, 123.7])
def test_interpolated_match_recovers_unity(shift_samples):
    IP = _overlap()
    h1, h2 = _series(IP, shift_samples)
    assert IP.norm(h1) == pytest.approx(1., abs=1e-12)
    assert IP.norm(h2) == pytest.approx(1., abs=1e-12)

    match = IP.ip(h1, h2)
    sampled = np.abs(IP.ovlp.data.data).max()

    # Measured for this signal: the 3-point parabola leaves 0.22-0.30 of the
    # sampled deficit; a 20% error in the vertex correction leaves ~0.4.
    assert match <= 1. + 1e-6
    assert 1. - match < 0.35 * (1. - sampled)


def test_integer_shift_returns_sample():
    IP = _overlap()
    h1, h2 = _series(IP, 0.)
    assert IP.ip(h1, h2) == pytest.approx(1., abs=1e-10)
