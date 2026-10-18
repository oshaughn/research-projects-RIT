"""Reproducibility harness: O4c's --seed sets NumPy, but not CuPy's RNG."""
import runpy
import sys
import cupy

cupy.random.seed(19)
script = sys.argv.pop(1)
runpy.run_path(script, run_name="__main__")
