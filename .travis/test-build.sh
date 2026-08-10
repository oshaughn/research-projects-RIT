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
# Distance marginalization must stay ON for the intrinsic ILE jobs (speedup)
# and be disabled ONLY at the extrinsic export stage. The trailing space in
# the pattern matches the standalone --distance-marginalization flag but not
# --distance-marginalization-lookup-table.
util_RIFT_pseudo_pipe.py --use-ini $REF_INI --use-coinc $COINC --use-rundir `pwd`/test_build_grid --fake-data-cache `pwd`/foo.cache --add-extrinsic --export-marginal-distance-grid
assert_has    `pwd`/test_build_grid/ILE_extr.sub "--export-marginal-distance-grid"
assert_has    `pwd`/test_build_grid/ILE_extr.sub "--internal-use-lnL"
assert_absent `pwd`/test_build_grid/ILE.sub      "--export-marginal-distance-grid"
assert_has    `pwd`/test_build_grid/args_ile.txt "--distance-marginalization "
assert_has    `pwd`/test_build_grid/ILE.sub      "--distance-marginalization "
assert_absent `pwd`/test_build_grid/ILE_extr.sub "--distance-marginalization "
echo "OK: Plan-A grid export only on ILE_extr.sub; distance marginalization disabled only at the extrinsic stage"

# --- 3. Plan-B distance-slice export, threaded onto the extrinsic stage ---
util_RIFT_pseudo_pipe.py --use-ini $REF_INI --use-coinc $COINC --use-rundir `pwd`/test_build_slices --fake-data-cache `pwd`/foo.cache --add-extrinsic --export-distance-slices 10 --export-distance-slices-wing-delta-lnL 7.0 --export-distance-slices-skip-threshold 1.0
assert_has    `pwd`/test_build_slices/ILE_extr.sub "--export-distance-slices 10"
assert_has    `pwd`/test_build_slices/ILE_extr.sub "--distance-slice-wing-delta-lnL 7.0"
assert_has    `pwd`/test_build_slices/ILE_extr.sub "--distance-slice-skip-threshold 1.0"
assert_has    `pwd`/test_build_slices/ILE_extr.sub "--internal-use-lnL"
assert_absent `pwd`/test_build_slices/ILE.sub      "--export-distance-slices"
assert_has    `pwd`/test_build_slices/args_ile.txt "--distance-marginalization "
assert_has    `pwd`/test_build_slices/ILE.sub      "--distance-marginalization "
assert_absent `pwd`/test_build_slices/ILE_extr.sub "--distance-marginalization "
echo "OK: Plan-B slice export only on ILE_extr.sub; distance marginalization disabled only at the extrinsic stage"

# --- 4. Gauss-early ('G') CIP groups use the alternate exe in BOTH sub files ---
# --use-gauss-early makes the first cip-args-list entry a G group.  The G exe
# must land in the master sub (CIP_0.sub, the only CIP without
# --cip-explode-jobs) and not just the worker sub; non-G groups keep the
# standard CIP.
util_RIFT_pseudo_pipe.py --use-ini $REF_INI --use-coinc $COINC --use-rundir `pwd`/test_build_gauss --fake-data-cache `pwd`/foo.cache --use-gauss-early
assert_has    `pwd`/test_build_gauss/CIP_0.sub        "GaussianResampling"
assert_absent `pwd`/test_build_gauss/CIP_0.sub        "GenericCoordinates"
assert_has    `pwd`/test_build_gauss/CIP_worker0.sub  "GaussianResampling"
assert_has    `pwd`/test_build_gauss/CIP_1.sub        "GenericCoordinates"
assert_absent `pwd`/test_build_gauss/CIP_1.sub        "GaussianResampling"
echo "OK: gauss-early G group uses GaussianResampling in master and worker subs"

echo "test-build.sh: all pipeline-build checks passed"
