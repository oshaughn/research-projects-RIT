#! /bin/bash
#
# Uses a fiducial event time, start time, stop time
#
#  util_WriteFrameAndCacheFromXML.sh   <inj> <event> <base>  [<approx>
#
#     <approx>    [TaylorT4|MATLAB|EOBNRv2|...]  # approximant used
#
# EXAMPLES
#   util_WriteFrameAndCacheFromXML.sh mdc.xml.gz 0 BASE
#   util_WriteFrameAndCacheFromXML.sh mdc.xml.gz 0 BASE MATLAB
#   util_WriteFrameAndCacheFromXML.sh mdc.xml.gz 0 BASE EOBNRv2

PI=`which util_PrintInjection.py`
ED=`which util_EstimateWaveformDuration.py`
LWF=`which util_LALWriteFrame.py`                # Can switch to EOBWriteFrame if necessary
EWF=`which util_EOBTideWriteFrame.py`                # Can switch to EOBWriteFrame if necessary
echo Using PrintInjection ${PI}
echo Using EstimateDuration ${ED}
echo Using LALWriteFrame ${LWF}

if [ "${PI}" == ''  ]; 
then
 echo " Cannot find executables; failing "
 exit 1
fi

get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

echo " ===== Creating frame cache for event ======="
INJ_RAW=$1
INJ_XML=$(get_abs_filename $1)
EVENT=$2
BASE=$3
${PI} --inj ${INJ_XML} --event ${EVENT} --verbose

# Last argument:
#    - approximant
#    - MATLAB (for Bernuzzi)
#    
APPROX=TaylorT4
if [ "$#" -gt 3 ]; then
  echo " Processing approximant ", $4
  if [ $4 == "MATLAB" ]; then
      echo " -------    USING EOB MODEL   -------  "
      LWF=${EWF}
  else
      APPROX=$4
  fi
fi

if [ "${LWF}" == ''  ]; 
then
 echo " Cannot find executables; failing "
 exit 1
fi




INJ_CHANNEL_NAME=FAKE-STRAIN
export DUR_EST=`${ED} --inj-xml ${INJ_XML} --event ${EVENT} | tail -n 1`
export DUR=`python -c "import math; print( 100+2**(2+int(math.log(${DUR_EST}*2)/math.log(2))))"`  # use a longer window
echo " Estimated duration "   ${DUR_EST}
echo " Target buffer length "  ${DUR}
if [ "${DUR_EST}" == '' ]; 
then
 echo " Cannot estimate duration failing "
 exit 1
fi



# event_time: only used to grab the data. 
#  Beware the number of fields in the 'time of coalescence' may change in the future. Not safe, should read from XML directtly
#EVENT_TIME=`${PI} --inj ${INJ_XML}  --event ${EVENT} --verbose | grep coal` # | awk '{print $6}'`
#EVENT_TIME=${EVENT_TIME##* }   #  http://stackoverflow.com/questions/3162385/how-to-split-a-string-in-shell-and-get-the-last-field
EVENT_TIME=`ligolw_print -c geocent_end_time ${INJ_XML}`
# Modified so the buffer has a power-of-2 length
START=`python -c "print( int(${EVENT_TIME}-${DUR}))"`
STOP=`python -c "print( int(${EVENT_TIME}+${DUR}))"`
DUR_REVISED=`python -c "print( int(${STOP}-${START}))"`

echo " Duration estimates :"  ${START} ${STOP}  ${DUR_EST} ${DUR} ${DUR_REVISED}

function make_cache(){
  local ifo=$1;
  local base=$2;
  local START=$3;
  local STOP=$4;
  local INJ_XML=$5;
  local EVENT=$6;
  local MIN_DUR=$7;
  mkdir ${base}_mdc;
  (cd ${base}_mdc;  ${LWF} --start ${START} --stop ${STOP} --inj ${INJ_XML} --event ${EVENT} --single-ifo --instrument ${ifo}1  --approx ${APPROX})
#  find ${base}_mdc/ -name  ${ifo}"-*.gwf" | lalapps_path2cache > ${ifo}.cache
}


make_cache H ${BASE} ${START} ${STOP} ${INJ_XML} ${EVENT} $DUR
make_cache L ${BASE} ${START} ${STOP} ${INJ_XML} ${EVENT} $DUR
make_cache V ${BASE} ${START} ${STOP} ${INJ_XML} ${EVENT} $DUR

find ${BASE}_mdc -name '*.gwf'  | lalapps_path2cache  > ${BASE}.cache
