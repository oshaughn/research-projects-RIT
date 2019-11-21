
# PP plot pipeline

This driver performs the steps necessary to generate "PP plots": consistency tests of the code, using injections and recovery from a known prior.
Following usual RIFT convention, we adopt uniform priors in (redshifted/detector-frame) m1,m2, bounded by specific ranges in mc, q.
Spin priors are either uniform in magnitude (if aligned or precessing), with ranges controlled by chi_max.
Lambda priors are uniform up to lambda_max.


## Tutorial

First, build and run the analysis of each event.

```
# insure add_frames.py is in your path
export PATH=${PATH}:`pwd`
# Generate workflow
pp_RIFT --ini sample_pp_config.ini
cd test_pp; condor_submit_dag master.dag
```

Next, for each injection j, extract P_{j,\alpha}(<x_{j,alpha}), the empirical CDF evaluated for the jth injection and parameter \alpha at the true value value x_{j,\alpha} of that parameter.   This script uses the last iteration provided by the PP code, and reports data in a format
```
# p(mc) p(q) p(a1z) p(a2z) lnL test_val
```

```
export PATH=${PATH}:path_to_file
HERE=`pwd`
for  i in  `seq 0 5` ; do 
  echo " ++" $i; 
  export HIGHEST_SAMPLE_FILE=`ls  analysis_event_${i}/posterior_samples-*.dat | sort | tail -n 1`
  echo Sample file: ${HIGHEST_SAMPLE_FILE}
  echo `python pp_plot_dataproduct.py --posterior-file "${HIGHEST_SAMPLE_FILE}" --truth-file ${HERE}/mdc.xml.gz --truth-event ${i}  --parameter mc --parameter q --parameter a1z --parameter a2z --composite-file analysis_event_${i}/all.net | tail -n 1`  `cat analysis_event_${i}/iteration*/logs/test*.out | tail -n 1` ; 
done > net_pp.dat

grep -v ++ net_pp.dat | grep -v [a-z] > net_pp.dat_clean

```
We recommend you review this data visually early on in your run, to be sure you haven't accidentally adopted inconsistent settings between input and output (e.g., inconsistent PSDs).

Finally, load in the data and make a cumulative CDF plot for each variable, with your favorite plotting code.  For lightweight tests, we provide 'pp_plot.py'

``
pp_plot.py net_pp.dat_clean 2   # second number indicates number of columns to plot
``
