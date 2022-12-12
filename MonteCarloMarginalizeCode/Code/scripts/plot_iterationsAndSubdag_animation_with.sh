#! /bin/bash

CMD_LINE=$@

PLOT=plot_posterior_corner.py
#PLOT=echo


# Print preliminary assessments
N_MAIN=`ls posterior_samples*.dat | wc -l`
N_SUBDAG=`ls iter*cip/posterior_samples*.dat | wc -l`
N_TOT=`expr ${N_MAIN} + ${N_SUBDAG}`
IT_SUBDAG=` ls iter*cip/ILE.sub | tr '/' ' ' | awk '{print $1}' | tr '_' ' ' | awk '{print $2}'`
echo " Plots expected: total, main, subdag = ${N_TOT} ${N_MAIN} ${N_SUBDAG} "
echo " Subdag iteration : ${IT_SUBDAG} "



plot_up_to() {
 n_it_max=$1
 n_it_max_before=`expr ${n_it_max}  - 1 `
 USECOUNTER=0
 fnames_here=`ls posterior_samples*.dat | sort -n | head -n ${n_it_max}`
 echo Filenames ${n_it_max} : ${fnames_here}
 echo Composite files 0 ... ${n_it_max_before}
 rm -rf comp_so_far; 
 for id in `seq 0 ${n_it_max_before}`; do cat consolidated_${id}.composite >> comp_so_far; done
 (for name in ${fnames_here}; do echo --posterior-file ${name} --posterior-label ${USECOUNTER}  ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile.txt
 echo ${CMD_LINE}  `cat myfile.txt` > myargs.txt
}

plot_up_to_after() {
 it_subdag=$1
 n_it_max=$2
 size_end=`expr ${n_it_max} - ${it_subdag}`
 n_it_max_before=`expr ${n_it_max}   `  # include last ILE step before subdag
  USECOUNTER=${it_subdag}
 n_post_top=`ls posterior_samples*.dat | wc -l`
 if [ ${n_post_top} \> ${it_subdag} ]; then
   echo ===== FILES POST SUBDAG DETECTED : ${n_post_top} ${it_subdag} ${size_end}
 fnames_before=`ls posterior_samples*.dat | sort -n | head -n ${it_subdag}`
 fnames_subdag=`ls iteration_${it_subdag}_cip/posterior_samples*.dag | sort -n `
 fnames_here=`ls posterior_samples*.dat | sort -n | head -n ${n_it_max} |  tail -n ${size_end}`
  echo Filenames after subdag: ${n_it_max} : ${fnames_here}
  rm -rf comp_so_far;
 for id in `seq 0 ${n_it_max_before}`; do cat consolidated_${id}.composite >> comp_so_far; done
 cat bonus*.composite >> comp_so_far
# (for name in ${fnames_before}; do echo --posterior-file ${name} --posterior-label ${USECOUNTER}  ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile1.txt
# (for name in ${fnames_subdag}; do echo --posterior-file ${name} --posterior-label Z${USECOUNTER}  ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile2.txt
 (for name in ${fnames_here}; do echo --posterior-file ${name} --posterior-label F${USECOUNTER}  ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile3.txt
  echo `cat myfile1.txt myfile2.txt myfile3.txt` >>myargs.txt
 fi
}

plot_subdag_up_to(){
 it_subdag=$1
 n_it_max=$2
  USECOUNTER=${it_subdag}
 n_it_max_before=`expr ${it_subdag}   `
 n_it_max_before_subdag=`expr ${n_it_max} - 1`
# fnames_old=`ls posterior_samples*.dat | sort -n | head -n ${n_it_max}`
 fnames_here=`ls iteration_${it_subdag}_cip/posterior_samples*.dat | sort -n | head -n ${n_it_max}`
 echo Filenames ${n_it_max} : ${fnames_old}  ${fnames_here}
 rm -rf comp_so_far;

 for id in `seq 0 ${n_it_max_before}`; do cat consolidated_${id}.composite >> comp_so_far; done
  for id in `seq 0 ${n_it_max_before_subdag}`; do cat iteration_${it_subdag}_cip/consolidated_${id}.composite >> comp_so_far; done
 # plot just subdag 
 (for name in ${fnames_here}; do echo --posterior-file ${name} --posterior-label Z${USECOUNTER}  ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile.txt
  echo  `cat myfile.txt` >> myargs.txt
}


N_MAX=${IT_SUBDAG} # `ls posterior_samples*.dat | wc -l`

N_MAX_ALL=${N_TOT}
echo " ==== INITIAL STAGE "
for i in `seq 1 ${N_MAX}`
do
  if [ -e posterior_samples-${i}.dat ]; then
  plot_up_to $i
  if [ ! -f "anim_${i}.png" ]; then 
    ${PLOT} `cat myargs.txt` --composite-file comp_so_far --lnL-cut 15 --quantiles None --ci-list [0.9]
    mv corner*.png  anim_${i}.png
  else
    echo  anim_${i}.png  EXISTS in `pwd`
    echo  Existing animation slides: `ls anim*.png`
  fi
  fi
done
IT_PLOT=${N_MAX}

if [ ${N_SUBDAG} == 0 ]; then
  exit 0
fi
echo " == POST INITIAL STAGE: " ${IT_PLOT} plots made

for j in `seq 1 ${N_SUBDAG}`
do
  IT_PLOT=`expr ${IT_PLOT} + 1 `
  plot_up_to ${N_MAX}
  plot_subdag_up_to ${IT_SUBDAG} $j
  if [ ! -f  "anim_${IT_PLOT}.png" ]; then
  ${PLOT} `cat myargs.txt` --composite-file comp_so_far --lnL-cut 15 --quantiles None --ci-list [0.9]
  mv corner*.png anim_${IT_PLOT}.png
  fi
done


# Test if main files exist after subdag
if [ ${N_MAIN} -gt ${IT_SUBDAG} ]; then
  echo " More plots after subdag : ${N_MAIN} ${IT_SUBDAG} "
else
  echo " No more plots after subdag ... "
  exit 0
fi

echo " == POST SUBDAG STAGE "



# Now plot post-subdag stages
IT_SUBDAG_P1=`expr ${IT_SUBDAG} + 1`
echo === FINAL PLOT STAGE ${IT_SUBDAG_P1} ${N_MAIN}
for j in `seq ${IT_SUBDAG_P1} ${N_MAIN}`
do
  IT_PLOT=`expr ${IT_PLOT} + 1 `
  plot_up_to ${IT_SUBDAG}
  plot_subdag_up_to ${IT_SUBDAG}  ${N_SUBDAG}
  plot_up_to_after  ${IT_SUBDAG} $j
  if [ ! -e "anim_${IT_PLOT}.png" ]; then 
    ${PLOT} `cat myargs.txt` --composite-file comp_so_far --lnL-cut 15 --quantiles None --ci-list [0.9]
    mv corner*.png anim_${IT_PLOT}.png
  fi
done
