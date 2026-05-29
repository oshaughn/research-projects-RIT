#! /bin/bash
# Pipeline build test. The coinc file is from a synthetic event.
#
# Builds (does not submit) RIFT DAGs from a reference ini + coinc using fake
# data, and verifies that the per-distance likelihood export flags (Plan A
# density grid, Plan B fixed-distance slices) thread through
# util_RIFT_pseudo_pipe.py -> create_event_parameter_pipeline_* and land in the
# correct condor submit file (ILE_extr.sub, the extrinsic stage).

set -e

export RIFT_LOWLATENCY=True
export SINGULARITY_RIFT_IMAGE=foo
# SINGULARITY_RIFT_IMAGE=/cvmfs/singularity.opensciencegrid.org/james-clark/research-projects-rit/rift:test
export SINGULARITY_BASE_EXE_DIR=/usr/bin/
alias gw_data_find=/bin/true  # don't want to reall do the datafind job
touch foo.cache

REF_INI=`pwd`/.travis/ref_ini/GW150914.ini
COINC=`pwd`/.travis/ref_ini/coinc.xml

# require a flag to be present in a file
assert_has() {  # file pattern
    if ! grep -q -- "$2" "$1"; then
        echo "FAIL: expected '$2' in $1"; exit 1
    fi
}
# require a flag to be absent from a file
assert_absent() {  # file pattern
    if grep -q -- "$2" "$1"; then
        echo "FAIL: did not expect '$2' in $1"; exit 1
    fi
}

# --- 1. baseline build (original smoke test) ---
util_RIFT_pseudo_pipe.py --use-ini $REF_INI --use-coinc $COINC --use-rundir `pwd`/test_build_pipe --fake-data-cache `pwd`/foo.cache

# --- 2. Plan-A distance-grid export, threaded onto the extrinsic stage ---
util_RIFT_pseudo_pipe.py --use-ini $REF_INI --use-coinc $COINC --use-rundir `pwd`/test_build_grid --fake-data-cache `pwd`/foo.cache --add-extrinsic --export-marginal-distance-grid
assert_has    `pwd`/test_build_grid/ILE_extr.sub "--export-marginal-distance-grid"
assert_has    `pwd`/test_build_grid/ILE_extr.sub "--internal-use-lnL"
assert_absent `pwd`/test_build_grid/ILE_extr.sub "--distance-marginalization"
assert_absent `pwd`/test_build_grid/args_ile.txt "--distance-marginalization"
assert_absent `pwd`/test_build_grid/ILE.sub      "--export-marginal-distance-grid"
echo "OK: Plan-A distance-grid export threaded into ILE_extr.sub"

# --- 3. Plan-B distance-slice export, threaded onto the extrinsic stage ---
util_RIFT_pseudo_pipe.py --use-ini $REF_INI --use-coinc $COINC --use-rundir `pwd`/test_build_slices --fake-data-cache `pwd`/foo.cache --add-extrinsic --export-distance-slices 10 --export-distance-slices-wing-delta-lnL 7.0 --export-distance-slices-skip-threshold 1.0
assert_has    `pwd`/test_build_slices/ILE_extr.sub "--export-distance-slices 10"
assert_has    `pwd`/test_build_slices/ILE_extr.sub "--distance-slice-wing-delta-lnL 7.0"
assert_has    `pwd`/test_build_slices/ILE_extr.sub "--distance-slice-skip-threshold 1.0"
assert_has    `pwd`/test_build_slices/ILE_extr.sub "--internal-use-lnL"
assert_absent `pwd`/test_build_slices/ILE_extr.sub "--distance-marginalization"
assert_absent `pwd`/test_build_slices/args_ile.txt "--distance-marginalization"
assert_absent `pwd`/test_build_slices/ILE.sub      "--export-distance-slices"
echo "OK: Plan-B distance-slice export threaded into ILE_extr.sub"

echo "test-build.sh: all pipeline-build checks passed"
