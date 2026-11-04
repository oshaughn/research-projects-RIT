"""util_TransverseSeedClimb.load_composite: reads ILE-scored lines for the composite-start mode."""
import importlib.util
import os

import numpy as np

from RIFT import lalsimutils

_BIN = os.path.join(os.path.dirname(__file__), "..", "..", "bin", "util_TransverseSeedClimb.py")


def _module():
    spec = importlib.util.spec_from_file_location("util_TransverseSeedClimb", _BIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_composite(tmp_path):
    mod = _module()
    rows = np.array([
        # indx m1 m2 s1x s1y s1z s2x s2y s2z lnL sigma ntot neff
        [0, 30.0, 20.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 50.0, 0.1, 1e5, 10],
        [1, 30.0, 20.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 50.2, 0.1, 1e5, 10],    # duplicate point
        [2, 15.0, 25.0, 0.0, 0.3, -0.1, 0.0, 0.0, 0.4, 60.0, 0.1, 1e5, 10],   # m2 > m1
        [3, 31.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.1, 1e5, 10],  # failed line
    ])
    fname = tmp_path / "all.net"
    np.savetxt(fname, rows)
    Ps, lnL = mod.load_composite(str(fname), lalsimutils.ChooseWaveformParams())
    assert len(Ps) == 2 and np.allclose(lnL, [50.0, 60.0])
    msun = mod.MSUN
    assert np.isclose(Ps[0].m1 / msun, 30.0) and np.isclose(Ps[0].s1x, 0.1)
    # components swapped so m1 >= m2, spins follow their bodies
    assert np.isclose(Ps[1].m1 / msun, 25.0) and np.isclose(Ps[1].s1z, 0.4)
    assert np.isclose(Ps[1].s2y, 0.3)
