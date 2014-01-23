#! /bin/bash
# construct_GraceDB_pipeline_wrapper <GraceDBID> <approximant>
#     Creates a new directory with the GraceDB event ID
#     Retrieves data
#     Generates a dag (and a 'test command') for processing that event

export ILE=`which integrate_likelihood_extrinsic`
export CME=`which compute_marginalized_likelihood`
export EWD=`which util_EstimateWaveformDuration.py`
echo Using ILE ${ILE} 
echo Using CME ${CME} 
echo Using EWD ${EWD}
mkdir -p $1
cd $1

export gid=$1
if [ -n "$2" ]; then
  export approximant=$2
else
  export approximant=TaylorT4
fi
export ldf_server=ldr.ligo.caltech.edu:443
if [ -n "${LIGO_DATAFIND_SERVER}" ]; then
    ldf_server=${LIGO_DATAFIND_SERVER}
fi
echo " Constructing directory for ", $gid, " using approximant ", $approximant

gracedb download $gid coinc.xml 
if [ ! -e coinc.xml ]; then   # Stop if not in GraceDB
    echo " No event ! "
    exit 1
fi
echo " GDB event ", $gid, " mass pairs: ",`ligolw_print -t sngl_inspiral -c mass1 -c mass2 coinc.xml | head -1`
gracedb download $gid skymap.fits.gz 
gracedb download ${gid} psd.xml.gz 


export MASS1=`ligolw_print -t sngl_inspiral -c mass1 coinc.xml | head -1`
export MASS2=`ligolw_print -t sngl_inspiral -c mass2 coinc.xml  | head -1`
export BETA=`util_AdaptiveExponent.py --Niter 4000 --coinc coinc.xml | tail -1`
if [ ! -n "$MASS1" ];  then  
   echo "FAILURE:  mass not populated"
   exit 1
fi

export tEvent=`ligolw_print -t coinc_inspiral -c end_time coinc.xml`
export tDuration=`${EWD} --coinc coinc.xml | tail -n 1`
export tBefore=`echo   "${tDuration}*1.1 + 8+2 " | bc `    # Buffer for inverse spectrum truncation plus safety. Just used for data retrieval.
export tAfter=${tBefore}   # Buffer for inverse spectrum truncation plus safety. Just used for data retrieval.
export DATA_START=`echo $tEvent - $tBefore | bc`
export DATA_END=`echo $tEvent + $tAfter | bc`
echo " Event ", $tEvent
echo " DATA_START ", $DATA_START
echo " DATA_END ", $DATA_END
echo " Duration ", $tDuration

if [ ! -n "$tDuration" ];  then  
   echo "FAILURE:  duration not populated"
   exit 1
fi



if [ ! -n "${LIGO_DATAFIND_SERVER}" ]; then 
  echo " Assuming user on laptop; retrieving data files ";
  ligo_data_find \
		-u file \
		--gaps \
		-o H \
		-t H1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $DATA_START \
		-e $DATA_END > H_remote.cache;
   ligo_data_find \
		-u file \
		--gaps \
		-o L \
		-t L1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $DATA_START \
		-e $DATA_END >L_remote.cache;
   ligo_data_find \
		-u file \
		--gaps \
		-o V \
		-t V1Online \
		--server ${ldf_server}\
		-s $DATA_START \
		-e $DATA_END > V_remote.cache;
   mkdir -p gdb-data-${gid};
   while read line; do 
			gsiscp ldas-pcdev1.ligo.caltech.edu:${line:16} gdb-data-${gid}/; \
   done < H_remote.cache;
   while read line; do 
        		gsiscp ldas-pcdev1.ligo.caltech.edu:${line:16} gdb-data-${gid}/; \
   done < L_remote.cache;
   while read line; do 
        		gsiscp ldas-pcdev1.ligo.caltech.edu:${line:16} gdb-data-${gid}/; \
   done < V_remote.cache;

   find . -name '*.gwf' | lalapps_path2cache > local.cache;

else
  echo " Assuming user on cluster "
  ligo_data_find \
		-u file \
		--gaps \
		-o H \
		-t H1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $DATA_START \
		-e $DATA_END > H_remote.cache;
   ligo_data_find \
		-u file \
		--gaps \
		-o L \
		-t L1_ER_C00_L1 \
		--server ${ldf_server}\
		-s $DATA_START \
		-e $DATA_END >L_remote.cache;
   ligo_data_find \
		-u file \
		--gaps \
		-o V \
		-t V1Online \
		--server ${ldf_server}\
		-s $DATA_START \
		-e $DATA_END > V_remote.cache;

   cat ?_remote.cache |  util_CacheFileConvert.py > local.cache;

fi

# CHANNEL NAMES
#    - by default, use the ER channel names -- that is what I will use for now
INJ_CHANNEL_NAME=FAKE-STRAIN
GDB_V_INJ_CHANNEL_NAME=FAKE_h_16384Hz_4R


## Make DAG

# Write a command that has the same options but just evaluates the likliehood (marginalized) at the trigger masses.  Helpful for debugging setup
#    - Comments left to indicate kind of arguments ROS wants on master
echo  ${ILE}  --cache-file local.cache  --coinc coinc.xml   --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${GDB_V_INJ_CHANNEL_NAME} --psd-file "H1=psd.xml.gz" --psd-file "L1=psd.xml.gz" --psd-file "V1=psd.xml.gz"   --mass1 ${MASS1} --mass2 ${MASS2} --reference-freq 0 --save-samples    --time-marginalization --n-max 200000 --n-eff 1000 --output-file ile-gracedb-${gid}.xml.gz   --save-P 0.0001 --fmax 2000 --adapt-weight-exponent ${BETA} --adapt-floor-level 0.1 --n-chunk 4000 --approximant $approximant --convergence-tests-on  > testme-command.sh  #  --adapt-parameter right_ascension --adapt-parameter declination --adapt-parameter distance       --approximant $approximant
# add postprocessing
echo make_triplot ile-gracedb-${gid}.xml.gz -o ${gid}-triplot.pdf >> testme-command.sh
echo plot_integral ile-gracedb-${gid}.xml.gz --output integral.pdf >> testme-command.sh
chmod a+x testme-command.sh



# Write the actual DAG
#   - large n-max chosen for prototyping purposes.  Hopefully we will hit the n-eff limit before reaching it.
${CME} --cache-file local.cache  --coinc coinc.xml  --channel-name H1=${INJ_CHANNEL_NAME} --channel-name L1=${INJ_CHANNEL_NAME} --channel-name V1=${GDB_V_INJ_CHANNEL_NAME} --psd-file "H1=psd.xml.gz" --psd-file "L1=psd.xml.gz" --psd-file "V1=psd.xml.gz"   --mass1 ${MASS1} --mass2 ${MASS2}  --save-samples  --time-marginalization --n-max 1000000 --n-eff 1000 --output-file CME-${gid}.xml.gz   --save-P 0.0001  --n-copies 2 --fmax 2000 --adapt-weight-exponent ${BETA} --adapt-floor-level 0.1 --n-chunk 4000 --approximant $approximant --convergence-tests-on # --adapt-parameter right_ascension --adapt-parameter declination --adapt-parameter distance   --fmax 2000

# Write commands to do some useful postprocessing.  Ideally part of the dag.
#   -  convert the result to a flat ascii grid in m1,m2, lnL
#   -  convert the individual XML outputs to a single compressed tabular ascii file
#   -  make some detailed 2d plots.  These *should* be done by the DAG.  (We need to add a variable number of bins)
echo 'for i in CME-*.xml.gz; do ligolw_print -t sngl_inspiral -c mass1 -c mass2 -c snr -c tau0  -d ' ' $i; done > massgrid.txt &' > postprocess-massgrid.sh
cat >> postprocess-massgrid.sh <<EOF
convert_output_format_ile2inference  CME-*.xml.gz > flatfile-points.dat
postprocess_1d_cumulative --save-sampler-file flatfile
gzip flatfile-points.dat
cat ILE_MASS*.cache > net-ile.cache 
ligolw_sqlite CME-*.xml.gz -d net-ile.sqlite
EOF
chmod a+x postprocess-massgrid.sh

echo "===== TO SUBMIT THE DAG ====== "
echo condor_submit_dag `pwd`/dagtest-gracedb-${gid}-marginalize.dag

echo "===== TO TEST THE UNDERLYING INFRASTRUCTURE USING THIS DATA ====== "
echo  `pwd`/testme-command.sh 
