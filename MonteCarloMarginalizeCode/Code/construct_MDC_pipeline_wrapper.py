#! /bin/bash
# construct_MDC_pipeline_wrapper <event_id> <approximant>
#     Creates a new directory with the event ID
#     Generates a dag (and a 'test command') for processing that event
#
# ASSUMES (for now)
#     - Just H1, L1 (!).  Needs to implement variable IFO list!
#     - cache file 'local.cache' exists in current working directory, holding needed locations of files
#     - 1_injections.xml file exists in current working directory, holding all injections
#     - H1_psd.xml.gz and L1_psd.xml.gz exist in the current working directory
#     - channel-name file

export ILE=`which integrate_likelihood_extrinsic`
export CME=`which compute_marginalized_likelihood`
export EWD=`which util_EstimateWaveformDuration.py`
echo Using ILE ${ILE} 
echo Using CME ${CME} 
echo Using EWD ${EWD}
mkdir -p $1
cd $1

export event=$1
if [ -n "$2" ]; then
  export approximant=$2
else
  export approximant=TaylorT4
fi
echo " Constructing directory for ", $event, " using approximant ", $approximant

cp ../1_injections.xml inj.xml
cp ../local.cache local.cache
cp ../H1_psd.xml.gz .
cp ../L1_psd.xml.gz .
cp ../channel-name .

export MASS1=`util_PrintInjectionParameters --inj inj.xml --event $1 --indicate-mass1 1 | tail -1`
export MASS2=`util_PrintInjectionParameters --inj inj.xml --event $1 --indicate-mass2 1 | tail -1`
export EVENT_TIME=`util_PrintInjectionParameters --inj inj.xml --event $1 --signal-time 0.0001  | tail -1`
export BETA=`util_AdaptiveExponent.py --Niter 4000 --inj inj.xml --event $1 --channel-name H1= --psd-file-singleifo "H1=H1_psd.xml.gz" --channel-name L1= --psd-file-singleifo "L1=L1_psd.xml.gz" | tail -1`



# CHANNEL NAMES
#    - by default, use the ER channel names -- that is what I will use for now
INJ_CHANNEL_NAME=`cat channel-name`
GDB_V_INJ_CHANNEL_NAME=FAKE_h_16384Hz_4R


## Make DAG

# Write a command that has the same options but just evaluates the likliehood (marginalized) at the trigger masses.  Helpful for debugging setup
#    - Comments left to indicate kind of arguments ROS wants on master
echo  ${ILE}  --cache-file local.cache   --event-time ${EVENT_TIME} --mass1 ${MASS1} --mass2 ${MASS2} --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME}  --psd-file "H1=H1_psd.xml.gz" --psd-file "L1=L1_psd.xml.gz"  --reference-freq 0 --save-samples    --time-marginalization --n-max 200000 --n-eff 1000 --output-file ile-mdc-${event}.xml.gz   --save-P 0.0001 --fmax 2000 --adapt-weight-exponent ${BETA} --adapt-floor-level 0.1 --n-chunk 4000   > testme-command.sh  #  --adapt-parameter right_ascension --adapt-parameter declination --adapt-parameter distance       --approximant $approximant
# add postprocessing
echo make_triplot ile-mdc-${event}.xml.gz -o ${event}-triplot.pdf >> testme-command.sh
echo plot_integral ile-mdc-${event}.xml.gz --output integral.pdf >> testme-command.sh
chmod a+x testme-command.sh



# Write the actual DAG
#   - large n-max chosen for prototyping purposes.  Hopefully we will hit the n-eff limit before reaching it.
${CME} --cache-file local.cache   --event-time ${EVENT_TIME} --mass1 ${MASS1} --mass2 ${MASS2} --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --psd-file "H1=H1_psd.xml.gz" --psd-file "L1=L1_psd.xml.gz"    --save-samples  --time-marginalization --n-max 1000000 --n-eff 1000 --output-file CME-${event}.xml.gz   --save-P 0.0001  --n-copies 2 --fmax 2000 --adapt-weight-exponent ${BETA} --adapt-floor-level 0.1 --n-chunk 4000  # --adapt-parameter right_ascension --adapt-parameter declination --adapt-parameter distance   --fmax 2000

# Write a command to convert the result to a flat ascii grid in m1,m2, lnL.  Ideally part of postprocessing DAG
echo > postprocess-massgrid.sh <<EOF
for i in CME-*; do ligolw_print -t sngl_inspiral -c mass1 -c mass2 -c snr  -d ' ' $i; done > massgrid.txt &
EOF
chmod a+x postprocess-massgrid.sh

echo "===== TO SUBMIT THE DAG ====== "
echo condor_submit_dag `pwd`/marginalize-extrinsic.dag

echo "===== TO TEST THE UNDERLYING INFRASTRUCTURE USING THIS DATA ====== "
echo  `pwd`/testme-command.sh 
