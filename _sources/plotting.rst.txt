==================
Plotting with RIFT
==================

Once you have run RIFT on your data, the next step is to plot the results.

Corner Plots
--------------
Corner plots are useful for visualizing multidimensional samples and examining the relationships between key parameters. Corner plots show the marginalized posteriors of the user selected parameters along the left diagonal. The peak of the marginalized posteriors represents RIFT's best approximation of the value of that parameter at the end of the run. Two-dimensional posteriors are shown as scatter/contour plots in the other positions on the corner plot. 

It is common to use corner plots to compare the results when different waveform models were used, but they have also been used to compare changes to RIFT initial settings as well as configurations used in previous searches.

Corner plots can be made using `this script <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/bin/plot_posterior_corner.py>`_. The most basic way to use this script is to make a corner plot of just one iteration, probably the final iteration, to look at the posterior distributions of the parameters you might be interested in. For example:

.. code-block:: console

	plot_posterior_corner.py --posterior-file posterior_samples-5.dat --parameter mc --parameter q

This would make a corner plot that includes chirp mass (mc) and mass ratio (q). The points represent individual samples whose color represents the likelihood of the point. A contour is drawn around the user specified confidence interval for the 2D posteriors. You may choose whether not to display the quantiles on the marginalized posteriors using the options below.

It is common to include the results from multiple ``posterior_samples-*.dat`` on the corner plot. This may be multiple iterations of the same run (which shows convergence over the course the run) or to compare the results of recovery of the same signal using different waveforms.

This script, :code:`plot_posterior_corner.py`, currently includes the following parameters as long as they have been saved in the output files ``posterior_samples-*.dat``:

.. collapse:: **Parameters**
	      :open:
		 
	      --parameter mtotal        total mass
	      --parameter q        mass ratio
	      --parameter mc        chirp mass
	      --parameter eta        symmetric mass ratio
	      --parameter chi_eff        effective spin
	      --parameter chi1_perp        spin on BH 1
	      --parameter chi2_perp        spin on BH 2
	      --parameter LambdaTilde        combined dimensionless tidal deformability
	      --parameter DeltaLambdaTilde        change in tidal deformability
	      --parameter lambdat        lambdat
	      --parameter dlambdat        dlambdat
	      --parameter eccentricity        orbital eccentricity

There are also many arguments that can be added to the command line call to customize the corner plots. Open the pull down "optional arguments" below to see the additional options and their descriptions.

.. collapse:: **Optional arguments**
	      
	      --posterior-file   filename of .dat file [standard LI output]
	      --truth-file    file containing the true parameters
	      --posterior-distance-factor   Sequence of factors used to correct the distances
	      --truth-event        number of event in file containing the true parameters
	      --composite-file        filename of .dat file [standard ILE intermediate]
	      --use-all-composite-but-grayscale        Composite; plot all grid points used in iteration but make them grayscale
	      --flag-tides-in-composite        Required if you want to parse files with tidal parameters
	      --flag-eos-index-in-composite        Required if you want to parse files with EOS index in composite (and tides)
	      --posterior-label        label for posterior file
	      --posterior-color        color and linestyle for posterior. PREPENDED onto default list, so defaults exist
	      --posterior-linestyle        color and linestyle for posterior. PREPENDED onto default list, so defaults exist
	      --parameter-log-scale        Put this parameter in log scale
	      --change-parameter-label        format name=string. Will be wrapped in $...$
	      --use-legend        creates a legend so colors of corr posteriors are labeled with --parameter-label
	      --use-title        User input string containing desired plot title
	      --use-smooth-1d        currently hard set to None
	      --plot-1d-extra        (add descrip here)
	      --pdf        Export PDF plots
	      --bind-param        a parameter to impose a bound on, with corresponding --param-bound arg in respective order
	      --param-bound        respective bounds for above params
	      --ci-list        List for credible intervals. Default is 0.95,0.9,0.68
	      --quantiles        List for 1d quantiles intervals. Default is 0.95,0.05
	      --chi-max        sets limits on spin range ?
	      --lambda-plot-max        set upper limits on matter
	      --sigma-cut        removes samples with sigma lower than this value
	      --eccentricity        Read sample files in format including eccentricity



	      
P-P Plots
--------------
Probability-probability plots 

This driver performs the steps necessary to generate "PP plots": consistency tests of the code using injections and recovery from a known prior. These plots show whether the recovered parameters follow the same distribution they are generated with. The data is plotted against the theoretical distribution and should follow approximately a straight line. Too much variation from a straight line indicates that the data is departed from the intended distribution.

Since these plots are created when testing code updates, the goal is to ensure that the changes do not disrupt the recovered data from the initial intended distribution. Following usual RIFT convention, we adopt uniform priors in (redshifted/detector-frame) m1,m2, bounded by specific ranges in mc, q. Spin priors are either uniform in magnitude (if aligned or precessing), with ranges controlled by chi_max. Lambda priors are uniform up to lambda_max. Eccentricity priors are uniform.

There are two useful scripts for creating pp plots after running RIFT on injections:  `pp_plot_dataproduct.py <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/test/pp/pp_plot_dataproduct.py>`_ and  `pp_plot.py <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-\port_python3_restructure_package/MonteCarloMarginalizeCode/Code/test/pp/pp_plot.py>`_ .

For each injection j, extract :math:`P_{j,\alpha}(<x_{j,alpha})`, the empirical CDF evaluated for the jth injection and parameter :math:`\alpha` at the true value value :math:`x_{j,\alpha}` of that parameter. 

This script ``pp_plot_dataproduct.py`` uses the last iteration output posterior file for the run to calculate a p-value for the user specified parameters and the highest value of log-likelihood for the best point in the final posterior samples file. For example, if the user specifies parameter chirp mass, mass ratio, and spins, the script reports data in a format

.. code-block:: console

   # p(mc) p(q) p(a1z) p(a2z) lnL

PP plots require information from each injection, so it is easiest to make a script to loop over your data files that includes the following:

.. code-block:: bash

		export PATH=${PATH}:path_to_file
		HERE= `pwd`
		for  i in  `seq 0 5`; do 
		    echo " ++" $i; 
		    export HIGHEST_SAMPLE_FILE= `ls  analysis_event_${i}/posterior_samples*.dat | sort | tail -n 1`
		    echo Sample file: ${HIGHEST_SAMPLE_FILE}
		    echo `python pp_plot_dataproduct.py --posterior-file "${HIGHEST_SAMPLE_FILE}" --truth-file ${HERE}/mdc.xml.gz --truth-event ${i}  --parameter mc --parameter q --parameter a1z --parameter a2z --composite-file analysis_event_${i}/all.net | tail -n 1`
		    echo `cat analysis_event_${i}/iteration*/logs/test*.out | tail -n 1`; 
		done > net_pp.dat
		# grab number-only entries, don't remove floating point
		grep -v ++ net_pp.dat | grep -v [a-df-z] > net_pp.dat_clean

This gathers the information from above (the parameter p-values for each injection, the maximum log-likelihood, as well as the convergence statistic) into a single file called ``net_pp.dat_clean``. Load in the data and make a cumulative CDF plot for each variable, with your favorite plotting code. For lightweight tests, we provide ``pp_plot.py``

.. code-block:: console
		
		pp_plot.py net_pp.dat_clean 2 ['mc', 'q']

This script orders and plots the p-values for all the injections for each parameter. In the above example, the p-values for chirp mass and mass ratio will be plotted together on a PP plot. The points are displayed with an ellipse representing the :math:`90%` confidence interval. Ideally, the points for each parameter should for a diagonal line on the pp-plot, indicating that the distribution of the recovered parameters matches the injected distribution. If your PP plot is *not* diagonal, there was likely some issue with your run. 




			















			  
