#! /bin/bash
#
# Uses a fiducial event time, start time, stop time
#
#  util_WriteFrameAndCacheFromXML.sh   <inj> <event> <base>

PI=`which util_PrintInjection.py`
ED=`which util_EstimateWaveformDuration.py`
LWF=`which util_LALWriteFrame.py`                # Can switch to EOBWriteFrame if necessary
EWF=`which util_EOBWriteFrame.py`                # Can switch to EOBWriteFrame if necessary
echo Using PrintInjection ${PI}
echo Using EstimateDuration ${ED}
echo Using LALWriteFrame ${LWF}


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


INJ_CHANNEL_NAME=FAKE-STRAIN
export DUR_EST=`${ED} --inj-xml ${INJ_XML} --event ${EVENT} | tail -n 1`
export DUR=`python -c "import math; print 2**(2+int(math.log(${DUR_EST}*2)/math.log(2)))"`  # use a longer window
echo " Estimated duration " , ${DUR_EST}
echo " Target buffer length ", ${DUR}

EVENT_TIME=`util_PrintInjection.py --inj mdc.xml.gz  --event 0 --verbose | grep coal | awk '{print $6}'`
START=`python -c "print int(${EVENT_TIME}-${DUR}-64)"`
STOP=`python -c "print int(${EVENT_TIME}+${DUR}+64)"`

echo " Duration estimates :"  ${START} ${STOP} ${DUR} ${DUR_EST} 

function make_cache(){
  local ifo=$1
  local base=$2
  local START=$3
  local STOP=$4
  local INJ_XML=$5
  local EVENT=$6
  local MIN_DUR=$7
  mkdir ${base}_mdc;
  (cd ${base}_mdc;  ${LWF} --start ${START} --stop ${STOP} --inj ${INJ_XML} --event ${EVENT} --single-ifo --instrument ${ifo}1 --seglen ${DUR})
#  find ${base}_mdc/ -name  ${ifo}"-*.gwf" | lalapps_path2cache > ${ifo}.cache
}


make_cache H ${BASE} ${START} ${STOP} ${INJ_XML} ${EVENT} $DUR
make_cache L ${BASE} ${START} ${STOP} ${INJ_XML} ${EVENT} $DUR
make_cache V ${BASE} ${START} ${STOP} ${INJ_XML} ${EVENT} $DUR

find ${BASE}_mdc -name '*.gwf'  | lalapps_path2cache  > ${BASE}.cache
