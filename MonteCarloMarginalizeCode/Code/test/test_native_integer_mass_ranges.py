"""Actual grid CLI preserves valid integer solar-mass bounds in both layouts."""
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from RIFT import lalsimutils
import lal

CODE = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("layout", ["random-only", "fixed-plus-random"])
def test_integer_and_float_mass_ranges_generate_identical_physical_grids(tmp_path, layout):
    grids=[]
    for label, bound in (("integer", "[14,20]"), ("float", "[14.0,20.0]")):
        stem = tmp_path / label
        flags=["--skip-overlap", "--fname", str(stem), "--grid-cartesian-npts", "4",
               "--random-parameter", "mc", "--random-parameter-range", bound]
        if layout == "random-only":
            flags += ["--random-parameter", "delta_mc", "--random-parameter-range", "[0.1,0.2]"]
        else:
            flags += ["--parameter", "delta_mc", "--parameter-range", "[0.1,0.2]"]
        code="import numpy as np, runpy, sys; np.random.seed(1729); script=sys.argv.pop(1); sys.argv[0]=script; runpy.run_path(script, run_name='__main__')"
        proc=subprocess.run([sys.executable,"-c",code,str(CODE/"bin/util_ManualOverlapGrid.py"),*flags],
                            capture_output=True,text=True,timeout=120)
        assert proc.returncode == 0, proc.stdout[-5000:]+proc.stderr[-5000:]
        points=lalsimutils.xml_to_ChooseWaveformParams_array(str(stem)+".xml.gz")
        values=np.array([[p.m1/lal.MSUN_SI,p.m2/lal.MSUN_SI] for p in points])
        assert len(values)>0 and np.isfinite(values).all()
        mc=(values[:,0]*values[:,1])**.6 / values.sum(axis=1)**.2
        assert np.all(mc>=14-1e-5) and np.all(mc<=20+1e-5)
        grids.append(values)
    np.testing.assert_array_equal(grids[0],grids[1])
