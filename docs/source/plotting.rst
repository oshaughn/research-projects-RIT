================
Plotting with RIFT
================

Once you have run RIFT on your data, the next step is to plot the results.

Corner Plots
--------------
Corner plots are useful for examining the marginal distributions of key parameters. It is common to use corner plots to compare the results when different waveform models were used, but have also been used to compare changes to RIFT initial settings as well as configurations used in previous searches.

Corner plots can be made using `this script <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/bin/plot_posterior_corner.py>`_. The most basic way to use this script is to make a corner plot of just one iteration, probably the final iteration, to look at the posterior distributions of the parameters you might be interested in. For example:

.. code-block:: console

	plot_posterior_corner.py --posterior-file posterior_samples-5.dat --parameter mc --parameter q

This would make a corner plot that includes chirp mass (mc) and mass ratio (q). This script, ``plot_posterior_corner.py``, currently includes the following parameters as long as they have been saved in the output files ``posterior_samples-*.dat``:

--parameter mtotal        total mass
--parameter q
        mass ratio
--parameter mc
        chirp mass
--parameter eta
--parameter chi_eff
--parameter chi1_perp
--parameter chi2_perp
--parameter LambdaTilde
--parameter DeltaLambdaTilde
--parameter lambdat
--parameter dlambdat

There are also many arguments that can be added to the command line call to customize the corner plots.
Optional arguments:
--truth-file
        file containing the true parameters
--posterior-distance-factor--truth-event
        Sequence of factors used to correct the distances
--truth-event
        file containing the true parameters
--composite-file
        filename of *.dat file [standard ILE intermediate]
--use-all-composite-but-grayscale
        Composite; plot all grid points used in iteration but make them grayscale
--flag-tides-in-composite
        Required if you want to parse files with tidal parameters
--flag-eos-index-in-composite
        Required if you want to parse files with EOS index in composite (and tides)
--posterior-label
        label for posterior file
--posterior-color
        color and linestyle for posterior. PREPENDED onto default list, so defaults exist
--posterior-linestyle
        color and linestyle for posterior. PREPENDED onto default list, so defaults exist
--parameter-log-scale
        Put this parameter in log scale
--change-parameter-label
        format name=string. Will be wrapped in $...$
--use-legend

--use-title

--use-smooth-1d

--plot-1d-extra
        
--pdf
        Export PDF plots
--bind-param
        a parameter to impose a bound on, with corresponding --param-bound arg in respective order
--param-bound
        respective bounds for above params
--ci-list
        List for credible intervals. Default is 0.95,0.9,0.68
--quantiles
        List for 1d quantiles intervals. Default is 0.95,0.05
--chi-max
        
--lambda-plot-max
        
--sigma-cut
        
--eccentricity
        Read sample files in format including eccentricity
 
P-P Plots
--------------
Probability-probability plots 

This driver performs the steps necessary to generate "PP plots": consistency tests of the code using injections and recovery from a known prior. Following usual RIFT convention, we adopt uniform priors in (redshifted/detector-frame) m1,m2, bounded by specific ranges in mc, q. Spin priors are either uniform in magnitude (if aligned or precessing), with ranges controlled by chi_max. Lambda priors are uniform up to lambda_max.

The first step is to build the analysis for each injection event and then perform a standard run. 

.. code-block:: console

   # ensure add_frames.py is in your path
   export PATH=${PATH}:`pwd`
   # Generate workflow
   pp_RIFT --use-ini sample_pp_config.ini
   cd test_pp; condor_submit_dag master.dag

Here, the heavy lifting is done by `pp_RIFT <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/test/pp/pp_RIFT>`_. This script requires inputs from an .ini file (formatted in a particular way, see examples in `this directory <https://github.com/oshaughn/research-projects-RIT/tree/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/test/pp>`_) to create a set of synthetic signals with user specified parameters, inject them into noise frames, and build single event workflows to be run with RIFT. The script creates a directory that contains all the necessary files to run RIFT, such that the user needs only to submit the analysis DAG in the top level of the new directory.

RIFT runs as usual. Then, for each injection j, extract :math:`P_{j,\alpha}(<x_{j,alpha})`, the empirical CDF evaluated for the jth injection and parameter :math:`\alpha` at the true value value :math:`x_{j,\alpha}` of that parameter. This script uses the last iteration provided by the PP code, and reports data in a format

.. code-block:: console

   # p(mc) p(q) p(a1z) p(a2z) lnL test_val

.. code-block:: console

   export PATH=${PATH}:path_to_file
   HERE=`pwd`
   for  i in  `seq 0 5` ; do 
     echo " ++" $i; 
     export HIGHEST_SAMPLE_FILE=`ls  analysis_event_${i}/posterior_samples-*.dat | sort | tail -n 1`
     echo Sample file: ${HIGHEST_SAMPLE_FILE}
     echo `python pp_plot_dataproduct.py --posterior-file "${HIGHEST_SAMPLE_FILE}" --truth-file ${HERE}/mdc.xml.gz --truth-event ${i}  --parameter mc --parameter q --parameter a1z --parameter a2z --composite-file analysis_event_${i}/all.net | tail -n 1`  `cat analysis_event_${i}/iteration*/logs/test*.out | tail -n 1` ; 
   done > net_pp.dat

   # grab number-only entries, don't remove floating point
   grep -v ++ net_pp.dat | grep -v [a-df-z] > net_pp.dat_clean

We recommend you review this data visually early on in your run, to be sure you haven't accidentally adopted inconsistent settings between input and output (e.g., inconsistent PSDs).







			















			  
