#! /bin/bash
#
#  util_ForOSG_MakeTruncatedLocalFramesDir.sh <rundir>
#  ASSUMES cache is called 'local.cache'
#   


OUT=frames_dir

cd $1
mkdir ${OUT}
#cat local.cache | awk '{print $NF}'  | tr -d '\r' > my_temp_files
#switcheroo file://localhost ' ' my_temp_files
if [ ! -e ILE.sub ]; then 
   echo " CANNOT IDENTIFY EVENT TIME; failing convert attempt "
fi
grep arguments ILE.sub | sed s/--/\\n/g | grep time > my_time_args
grep arguments ILE.sub | sed s/--/\\n/g  | grep channel  | tr '=' ' '  | awk '{print $2,$3}' > my_channel_pairs
TSTART=`grep data-start-time my_time_args | awk '{print $NF}' | xargs printf '%.*f\n' 0 `
TSTART=`echo ${TSTART} - 1 | bc `
TEND=`grep data-end-time my_time_args | awk '{print $NF}' | xargs printf '%.*f\n' 0 `
TEND=`echo  ${TEND} + 1 | bc `
SEGLEN=`echo ${TEND} - ${TSTART} | bc`
echo ${TSTART} ${TEND} ${SEGLEN}

cat my_channel_pairs | awk '{print $1}' > my_ifo_list
#for i in `cat my_temp_files`; do basename $i; done | tr '-' ' ' | awk '{print $1}' | sort | uniq  > my_ifo_list
#for i in `cat my_temp_files`; do basename $i | tr '-' ' ' ; done | awk '{print $1,$2}' | sort | uniq > my_channel_pairs

# Loop over interferometers.  Join together all frames from that interferometer
for i in `cat my_ifo_list`
do
 echo Frame truncation starting   $i
 # note some horrible old frame names have 'V1' in them, notably LOSC GW710817. So I need a workaround 
 # I also need a workaround if the channel name contains dashes: I can't put that in the filename if I want to run on OSG properly
  ifo_name=`echo $i | tr 1 ' ' | awk '{$1=$1;print}' `
  grep ^${ifo_name} local.cache | awk '{print $NF}' | tr -d '\r' > temp_file_now
  switcheroo file://localhost '' temp_file_now
  INFILES=`cat temp_file_now`
  CHANNEL=`grep $i my_channel_pairs | awk '{print $NF}' `
  CHANNEL_NO_DASH=`echo ${CHANNEL} | tr '-' '_'`
  util_TruncateMergeFrames.py  --start ${TSTART} --end ${TEND} --output ${OUT}/${i}-${CHANNEL_NO_DASH}-${TSTART}-${SEGLEN}.gwf --channel ${i}:${CHANNEL}  ${INFILES}
#  FrCopy -f ${TSTART} -l ${TEND}  -i `grep ${i} my_temp_files`  -o  ${OUT}/${i}-${CHANNEL}-${TSTART}-${SEGLEN}.gwf
done
