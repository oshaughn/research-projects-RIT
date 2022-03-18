HERE=`pwd`
export run=$1
export run_dir=analysis_event_${run}
shift
cd ${run_dir}

TIDES_ARGS=""
head -n 1 all.net | awk '{print NF}' > line_length.txt
LINE_LENGTH=`cat line_length.txt`
if [ ${LINE_LENGTH} -gt 13 ]; then
   TIDES_ARGS="--flag-tides-in-composite"
   MAX_LNL=`sort -rg -k12 all.net | awk '{print $12}' | head -n 1`
else
   MAX_LNL=`sort -rg -k10 all.net | awk '{print $10}' | head -n 1`
fi
#MAX_LNL=`sort -rg -k10 all.net | awk '{print $10}' | head -n 1`
EXTRA_ARGS=" --use-title event=${run},lnL=${MAX_LNL} --use-all-composite-but-grayscale "
${HERE}/plot_last_iterations_with.sh  $@ --use-legend --plot-1d-extra --truth-file ${HERE}/mdc.xml.gz --truth-event ${run} --composite-file all.net --lnL-cut 15 --quantiles None  --ci-list  '[0.9]' ${EXTRA_ARGS} ${TIDES_ARGS}
 ${HERE}/plot_last_iterations_with.sh --parameter m1 --parameter m2 --use-legend --plot-1d-extra --truth-file ${HERE}/mdc.xml.gz --truth-event ${run} --composite-file all.net --lnL-cut 15 --quantiles None  --ci-list  '[0.9]'  ${EXTRA_ARGS} ${TIDES_ARGS}
