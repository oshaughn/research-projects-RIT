#! /bin/bash

CMD_LINE=$@

plot_up_to() {
 n_it_max=$1
 n_it_max_before=`expr ${n_it_max}  - 1 `
 USECOUNTER=0
 fnames_here=`ls posterior_samples*.dat | sort -n | head -n ${n_it_max}`
 echo Filenames ${n_it_max} : ${fnames_here}
 rm -rf comp_so_far; 
 for id in `seq 0 ${n_it_max_before}`; do cat consolidated_${id}.composite >> comp_so_far; done
 (for name in ${fnames_here}; do echo --posterior-file ${name} --posterior-label ${USECOUNTER}  ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile.txt
 echo ${CMD_LINE}  `cat myfile.txt` > myargs.txt
 #plot_posterior_corner.py `cat myargs.txt` --composite-file comp_so_far --lnL-cut 15 --quantiles None --ci-list [0.9]
 #mv corner*.png  anim_${n_it_max}.png
}

plot_up_to_after() {
 it_subdag=$1
 n_it_max=$2
 n_it_max_before=`expr ${n_it_max}  - 1 `
  USECOUNTER=${it_subdag}
 fnames_before=`ls posterior_samples*.dat | sort -n | head -n ${it_subdag}`
 fnames_subdag=`ls iteration_${it_subdag}_cip/posterior_samples*.dag | sort -n `
 fnames_here=`ls posterior_samples*.dat | sort -n | head -n ${n_it_max} |  tail -n +${it_subdag}`
  echo Filenames after subdag: ${n_it_max} : ${fnames_here}
  rm -rf comp_so_far;
 for id in `seq 0 ${n_it_max_before}`; do cat consolidated_${id}.composite >> comp_so_far; done
 cat bonus*.composite >> comp_so_far
# (for name in ${fnames_before}; do echo --posterior-file ${name} --posterior-label ${USECOUNTER}  ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile1.txt
# (for name in ${fnames_subdag}; do echo --posterior-file ${name} --posterior-label Z${USECOUNTER}  ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile2.txt
 (for name in ${fnames_here}; do echo --posterior-file ${name} --posterior-label F${USECOUNTER}  ; USECOUNTER=`expr ${USECOUNTER} + 1`;  done) | tr '\n' ' ' > myfile3.txt
  echo `cat myfile1.txt myfile2.txt myfile3.txt` >>myargs.txt
}

plot_subdag_up_to(){
 it_subdag=$1
 n_it_max=$2
  USECOUNTER=${it_subdag}
 n_it_max_before=`expr ${it_subdag}  - 1 `
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


N_MAX=`ls posterior_samples*.dat | wc -l`
if [ -e  iteration_4_cip/all.net  ]; then
  N_MAX=4
elif [ -e iteration_3_cip/all.net ]; then
 N_MAX=3
elif [ -e  iteration_2_cip/all.net  ]; then
 N_MAX=2
fi

N_MAX_ALL=`ls posterior_samples*.dat | wc -l`

for i in `seq 1 ${N_MAX}`
do
  plot_up_to $i
  plot_posterior_corner.py `cat myargs.txt` --composite-file comp_so_far --lnL-cut 15 --quantiles None --ci-list [0.9]
 mv corner*.png  anim_${i}.png
done
IT_PLOT=${N_MAX}

# Plot intermediate steps
if [ -e iteration_4_cip/all.net ]; then
  echo subdag ready
 IT_PLOT=${N_MAX}
  N_SUB_MAX=`ls iter*cip/posterior_samples*.dat | wc -l`
 for j in `seq 1 ${N_SUB_MAX}`
 do
  IT_PLOT=`expr ${IT_PLOT} + 1 `
  plot_up_to ${N_MAX}
  plot_subdag_up_to 4 $j
  plot_posterior_corner.py `cat myargs.txt` --composite-file comp_so_far --lnL-cut 15 --quantiles None --ci-list [0.9]
  mv corner*.png anim_${IT_PLOT}.png
 done
elif [ -e iteration_3_cip/all.net ]; then 
  echo subdag ready
 IT_PLOT=${N_MAX}
  N_SUB_MAX=`ls iter*cip/posterior_samples*.dat | wc -l`
 for j in `seq 1 ${N_SUB_MAX}`
 do
  IT_PLOT=`expr ${IT_PLOT} + 1 `
  plot_up_to ${N_MAX}
  plot_subdag_up_to 3 $j
  plot_posterior_corner.py `cat myargs.txt` --composite-file comp_so_far --lnL-cut 15 --quantiles None --ci-list [0.9]
  mv corner*.png anim_${IT_PLOT}.png
 done
elif [ -e iteration_2_cip/all.net ]; then
  echo subdag ready
 IT_PLOT=${N_MAX}
  N_SUB_MAX=`ls iter*cip/posterior_samples*.dat | wc -l`
 for j in `seq 1 ${N_SUB_MAX}`
 do
  IT_PLOT=`expr ${IT_PLOT} + 1 `
  plot_up_to ${N_MAX}
  plot_subdag_up_to 2 $j
  plot_posterior_corner.py `cat myargs.txt` --composite-file comp_so_far --lnL-cut 15 --quantiles None --ci-list [0.9]
  mv corner*.png anim_${IT_PLOT}.png
 done
fi


# Now plot post-subdag stages
for j in `seq ${N_MAX} ${N_MAX_MAX}`
do
  N_SUB_MAX=`ls iter*cip/posterior_samples*.dat | wc -l`
  IT_PLOT=`expr ${IT_PLOT} + 1 `
  plot_up_to ${N_MAX}
  plot_subdag_up_to ${N_MAX}  ${N_SUB_MAX}
  plot_up_to_after  ${N_MAX} $j
  plot_posterior_corner.py `cat myargs.txt` --composite-file comp_so_far --lnL-cut 15 --quantiles None --ci-list [0.9]
  mv corner*.png anim_${IT_PLOT}.png
done
