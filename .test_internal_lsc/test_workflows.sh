#! /bin/bash


GID_FIDUCIAL=G347305
APPROX_FIDUCIAL=IMRPhenomD
EXTRA_ARGS=" --cip-fit-method rf --cip-sampler-method gmm --calibration C01 --choose-data-LI-seglen --fmin 20 --spin-magnitude-prior zprior_aligned --make-bw-psds "

# standard workflow
util_RIFT_pseudo_pipe.py --approx ${APPROX_FIDUCIAL} --gracedb-id ${GID_FIDUCIAL} ${EXTRA_ARGS} --manual-postfix _StandardTest0

# OSG workflow
util_RIFT_pseudo_pipe.py --approx ${APPROX_FIDUCIAL} --gracedb-id ${GID_FIDUCIAL} ${EXTRA_ARGS} --use-osg --condor-local-nonworker --manual-postfix _StandardTest1
