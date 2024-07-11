GRID=$(shell which util_NRExtrudeOverlapGrid.py)
ILE=$(shell which integrate_likelihood_extrinsic_batchmode)
PUFF=$(shell which util_ParameterPuffball.py)
CIP=$(shell which util_ConstructIntrinsicPosterior_GenericCoordinates.py)
REFINE=$(shell which util_TestSpokesIO.py)
PWD=$(shell pwd)

GID=G333631
SID=S190521g
EVENT_TIME=1242442967.4472656
LMAX=4
CHANNEL_NAME=DCS-CALIB_STRAIN_CLEAN_C01
CHANNEL_NAME2=Hrec_hoft_16384Hz
FREF = 10.0
FMAX = 224.0

NR_GROUP=Sequence-RIT-All
MTOT_RANGE='[120,180]'
MC_RANGE='[50,80]'
ETA_RANGE='[0.2,0.2499999]'
NPTS=20
SRATE=512

#ILE_NR=--save-P 0.1 --fmax ${FMAX} --cache ${PWD}/local.cache --event-time ${EVENT_TIME} --channel-name H1=${CHANNEL_NAME} --psd-file H1=${PWD}/H1-psd.xml.gz --fmin-ifo H1=20 --channel-name L1=${CHANNEL_NAME} --psd-file L1=${PWD}/L1-psd.xml.gz --fmin-ifo L1=20 --channel-name V1=${CHANNEL_NAME2} --psd-file V1=${PWD}/V1-psd.xml.gz --fmin-ifo V1=20 --fmin-template 20.0 --reference-freq ${FREF}  --inv-spec-trunc-time 0 --window-shape 0.2 --time-marginalization --inclination-cosine-sampler --declination-cosine-sampler --n-max 4000000 --n-eff 10  --vectorized --gpu --srate ${SRATE} --no-adapt-after-first --no-adapt-distance --adapt-weight-exponent 0.1 --l-max ${LMAX} --force-reset-all  --sampler-method GMM  --force-xpy  --n-events-to-analyze 20 --nr-lookup --nr-lookup-group ${NR_GROUP} --nr-use-provided-strain --save-eccentricity

## Cannot find V PSDs for 190521 in pe dir, eliminate V for now
ILE_NR=--save-P 0.1 --fmax ${FMAX} --cache ${PWD}/local.cache --event-time ${EVENT_TIME} --channel-name H1=${CHANNEL_NAME} --psd-file H1=${PWD}/H1-psd.xml.gz --fmin-ifo H1=20 --channel-name L1=${CHANNEL_NAME} --psd-file L1=${PWD}/L1-psd.xml.gz --fmin-ifo L1=20  --fmin-template 20.0 --reference-freq ${FREF}  --inv-spec-trunc-time 0 --window-shape 0.2 --time-marginalization --inclination-cosine-sampler --declination-cosine-sampler --n-max 4000000 --n-eff 10  --vectorized --gpu --srate ${SRATE} --no-adapt-after-first --no-adapt-distance --adapt-weight-exponent 0.1 --l-max ${LMAX} --force-reset-all  --sampler-method GMM  --force-xpy  --n-events-to-analyze 20 --nr-lookup --nr-lookup-group ${NR_GROUP} --nr-use-provided-strain --save-eccentricity --internal-use-lnL

PUFF_ARGS=--parameter mc --fmin 20.0 --fref ${FREF} --puff-factor 5 --force-away 0.1

CIP_ARGS=--no-plots --fit-method rf   --parameter-implied mu1 --parameter-implied mu2 --parameter-nofit mc --parameter delta_mc  --cap-points 12000  --lnL-offset 1000 --parameter-nofit s1z --parameter-nofit s2z --eta-range ${ETA_RANGE} --mc-range ${MC_RANGE}  --chi-max 0.9  --n-output-samples 2000 --n-eff 500 --n-max 10000000   --downselect-parameter m2 --downselect-parameter-range [1,1000]  --sampler-method GMM --internal-correlate-parameters 'mc,delta_mc,s1z,s2z'  --parameter eccentricity  --use-eccentricity

REFINE_ARGS=--test-refinement --is-eccentric

# pipeline writer needs to be fixed so that this does not need to be present for convert to write
TEST=--method lame  --parameter mc --parameter eta  --iteration ${macroiteration} --parameter xi  --iteration-threshold 2  --threshold 0.01

CONVERT=--export-eccentricity

data:
	(mkdir $@; exit 0)
	(cd $@; gracedb get file ${GID} coinc.xml coinc.xml)
	(cd $@; ligolw_no_ilwdchar coinc.xml)
	(cd $@; gracedb get file ${GID} event.log event.log)
	(cd $@; gracedb get file ${GID} psd.xml.gz psd.xml.gz)
	(cd $@; cp /home/pe.o3/o3a_catalog/run_directories/${SID}/C01_offline/Prod1*.ini ./base.ini)
	(cd $@; switcheroo 'approx = .*' 'approx=IMRPhenomD' base.ini)
	(cd $@; util_RIFT_pseudo_pipe.py --use-ini `pwd`/base.ini --use-coinc `pwd`/coinc.xml --use-rundir `pwd`/rundir --calibration C01)
#	(cd $@/rundir; export CHANNEL_H=$$(sed -n 's/^.*--channel-name H1=\([^ ]*\).*$$/\1/p' args_ile.txt))
#	echo ${CHANNEL_H}

grid:
	(cp data/psd.xml.gz .)
	(${GRID} --group ${NR_GROUP}  --fmin 20.0 --verbose --skip-overlap --mtot-range ${MTOT_RANGE} --grid-cartesian-npts ${NPTS} --eta-range ${ETA_RANGE} --lmax 4 --psd-file psd.xml.gz --aligned-only --eccentricity)

rundir:
	(mkdir $@; exit 0)
	(cd $@; echo X   ${ILE_NR} > args_ile.txt)
	(cd $@; echo X   ${PUFF_ARGS} > args_puff.txt)
	(cd $@; echo X   ${CIP_ARGS} >	args_cip.txt)
	(cd $@; echo X   ${REFINE_ARGS} >  args_refine.txt)
	(cd $@; echo X   ${TEST} >  args_test.txt)
	(cd $@; echo X   ${CONVERT} >  args_convert.txt)
	(cd $@;  python create_event_nr_pipeline_with_cip --request-gpu-ILE --ile-n-events-to-analyze 20 --ile-exe ${ILE}  --ile-args args_ile.txt --cip-args args_cip.txt --nr-refine-args args_refine.txt --nr-refine-exe ${REFINE}  --nr-group ${NR_GROUP} --request-memory-CIP 30000 --request-memory-ILE 4096  --input-grid ${PWD}/overlap-grid.xml.gz --n-samples-per-job 5000 --working-directory ${PWD}/$@ --n-iterations 2 --use-eccentricity --convert-args args_convert.txt --test-args args_test.txt )
	(cp data/rundir/local.cache .)
	(cd $@; mkdir logs)
	(cd $@; switcheroo 'overlap' 'cip' CIP.sub)
	(cd $@; switcheroo 'overlap' 'cip' convert.sub)
# FIX THIS	(cd $@; cp ../data/rundir/*psd.xml.gz .)
