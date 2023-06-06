#! /bin/bash
# This is just a pipeline build test. The coinc file is from a synthetic event.

export RIFT_LOWLATENCY=True
export SINGULARITY_RIFT_IMAGE=foo
# SINGULARITY_RIFT_IMAGE=/cvmfs/singularity.opensciencegrid.org/james-clark/research-projects-rit/rift:test
export SINGULARITY_BASE_EXE_DIR=/usr/bin/
alias gw_data_find=/bin/true  # don't want to reall do the datafind job
touch foo.cache
util_RIFT_pseudo_pipe.py --use-ini  `pwd`/.travis/ref_ini/GW150914.ini --use-coinc `pwd`/.travis/ref_ini/coinc.xml --use-rundir `pwd`/test_build_pipe --fake-data-cache `pwd`/foo.cache
