#!/bin/bash
# Scheduler-free development environment. Does not qualify frozen PyPI rc12.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_root=$(CDPATH= cd -- "$here/../.." && pwd)
prefix=${1:?Usage: setup-cit.sh /absolute/new/development-prefix}
case "$prefix" in /*) ;; *) echo 'Use an absolute new prefix' >&2; exit 2 ;; esac
test ! -e "$prefix"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/cvmfs/software.igwn.org/conda/envs/igwn-py311-20261013/bin/python3.11
"$base" -m venv "$prefix"
"$prefix/bin/python" -m pip install pip==26.2.2
"$prefix/bin/python" -m pip install -r "$here/dependencies-py311.lock.txt"
"$prefix/bin/python" -m pip install --no-deps --no-build-isolation "$source_root"
"$prefix/bin/python" -m pip check
"$prefix/bin/python" "$here/verify-development.py" "$here/dependencies-py311.lock.txt" > "$prefix/development-validation.json"
