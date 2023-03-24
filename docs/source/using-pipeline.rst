==================
Using the pipeline
==================

Basics
------

The primary user-interface for this code is a command line tool
:code:`util_RIFT_pseudo_pipe.py` which is available after following the `installation
instructions <installation.txt>`_. To see the help for this tool, run

.. code-block:: console

   $ util_RIFT_pseudo_pipe.py --help

If you leave options unspecified, the pipeline will often try to make guesses about the options you want to use, based
on previous experience performing real GW inference.  We strongly recommend most users fully specify the options
involved via an ini file. To compare the options for ini file use versus command line specified:

.. code-block:: console

   $ util_RIFT_pseudo_pipe.py --use-ini my.ini --use-coinc coinc.xml
   # or, for a freestanding example
   $ util_RIFT_pseudo_pipe.py --gracedb-id G329473 --approx IMRPhenomD --calibration C01 --make-bw-psds --l-max 2 --choose-data-LI-seglen

Note that the code will use selected environment variables to identify optional external dependencies necessary
for various features.  The most important feature for most users is their accounting access information 

.. code-block:: console
		
  export LIGO_USER_NAME=albert.einstein
  export LIGO_ACCOUNTING=ligo.dev.o4.rift

When you run :code:`util_RIFT_psuedo_pipe.py`, the pipeline will create a directory structure as follows:

.. code-block:: console
		
   long_directory_name_here/
      -> local.cache
      -> iteration_0_ile/
      -> iteration_0_cip/
      -> iteration_0_con/
      -> iteration_0_test/
      -> iteration_1_ile/
      -> ...
      -> ILE.sub
      -> CIP*.sub
      -> marginalize_intrinsic_parameters_BasicIterationWorkflow.dag 

Inside each iteration file is a `logs` subdirectory.  The various iterations will be initially be empty, except for the
logfile locations.  The top level directory contains several `*.sub` submission scripts, along with the top-level dag
submission script.  

The standard RIFT pipeline only works within an HTCondor scheduling environment. To submit the workflow, use

.. code-block:: console

    $ condor_submit_dag marginalize_intrinsic_parameters_BasicIterationWorkflow.dag 

Before you submit a workflow, however, we recommend you first confirm you've set it up correctly by running one of the worker jobs interactively from the command line. This is a great way to catch common configuration errors. Within the directory, there should be a script called :code:`command-single.sh`. This contains a single worker job, so you can simply run this script to confirm that your worker jobs will proceed smoothly.

.. code-block:: console

    $ ./command-single.sh

This command will run anywhere; however, it will only test the GPU configuration if you run it on a machine with a GPU, like pcdev13 or pcdev11 at CIT. You'll see a lot of output about reading in data, defining parameters, et cetera.  Wait until you start seeing large arrays of numbers interspersed with the words ``Weight entropy (after histogram)``. At this point you may kill the script and submit the DAG as described above. Feel free to watch the job progress by:

.. code-block:: console

    $ watch condor_q

The workflow loosely consists of two parts: worker ILE jobs, which evaluate the marginalized likelihood; and fitting/posterior jobs, which fit the marginalized likelihood and estimate the posterior distribution.  Other nodes help group the output of individual jobs and iterations together.

As your run proceeds, files will begin to appear in your directory. A description of some of the files is as follows:


- ``overlap-grid-0.xml.gz``: The initial grid used in the iterative analysis. You're free to use any grid you want (e.g., the output of some previous analysis), and the workflow can also do the initial grid creation.

- ``ILE.sub``: The submit file for the individual worker ILE jobs. This contains the command line arguments passed to :code:`integrate_likelihood_extrinsic`. If something is going wrong when your ILE jobs run, this file is a good place to check to ensure the code is using the settings you intended.

- ``CIP.sub``: The submit file for the individual fitting jobs.

- ``iteration_*``: Directories holding the output of each iteration, including log files.

As the workflow progresses, you'll see the following additional files

- ``consolidated_*``: These files (particularly those ending in .composite) are the output of each iteration's ILE jobs. Each file is a list of intrinsic parameters and the value of the marginalized likelihood at those parameters.  (The remaining files provide provenance for  how the .composite file was produced.)

- ``output-grid-?.xml.gz``: These files are inferred intrinsic, detector-frame posterior distributions from that iteration, expressed as an XML file.

- ``posterior-samples-*.dat``: These files are reformatted versions of the corresponding XML file, using the command convert_output_format_ile2inference.  This data format should be compatible with LALInference and related postprocessing tools. The final output posterior samples are used to create PP plots. Corner plots for a user specified number of iterations are also created using these files.


Understanding ILE and CIP
-------------------------

ILE.sub
^^^^^^^^^^^
The ``ILE.sub`` file contains the call to and arguments for `integrate_likelihood_extrinsic_batchmode <https://git.ligo.org/rapidpe-rift/rift/-/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/bin/integrate_likelihood_extrinsic_batchmode>`__. This step integrates the extrinsic parameters of the prefactored likelihood function. The marginalized likelihood is calculated during individual parallel worker jobs to be passed to the next (CIP) step.


CIP.sub
^^^^^^^^^^^
The file called ``CIP.sub`` contains the call to and arguments for `util_ConstructIntrinsicPosterior_GenericCoordinates.py <https://git.ligo.org/rapidpe-rift/rift/-/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py>`__. During this step, the log-likelihoof data is loaded in and the peak is fitted using some particular coordinate system. This is passed as an input to the Monte Carlo sampler where samples are drawn from the posterior distribution. These samples become the inputs for the successive iteration. 

Initialization: PSDs and grids
------------------------------
The RIFT pipeline by default can attempt to generate a PSD using existing tools.  However, most users will want to supply a contemporary PSD for analysis on real GW events. RIFT uses an XML format, requiring some conversion for the input PSDs.


Strongly recommended dependencies
---------------------------------
We strongly recommend you install `cuda` and `cupy`, and properly define your environment variables for such an install

.. code-block:: console
		
  # should be provided by igwn
  export CUDA_DIR=/usr/local/cuda  # only needed for GPU code
  export PATH=${PATH}:${CUDA_DIR}/bin  # only needed for GPU code


Additional environment variables are needed if you want to use waveforms through a non-lalsimulation interface.   Such
waveforms include the python implementation of surrogate waveforms;  NR waveforms; or the C++ implementation of
TEOBResumS.   While we provide the necessary environment variables below, please contact one of the developers for
appropriate settings, and keep in mind some surrogates and/or simulations and/or waveforms may be provided in advance of
publication or release to the broader community.

.. code-block:: console
		
   export NR_BASE=/home/oshaughn/unixhome/PersonalNRArchive/Archives/
   export GW_SURROGATE= # your installation of gwsurrogate goes here
   export PYTHONPATH=${PYTHONPATH}:${GW_SURROGATE}


util_RIFT_pseudo_pipe.py help
-----------------------------

For reference, here is the full output of

.. code-block:: console

   $ util_RIFT_pseudo_pipe.py --help

.. highlight:: none

.. code-block::
   
   usage: util_RIFT_pseudo_pipe.py [-h] [--use-production-defaults] [--use-subdags] [--use-ini USE_INI] [--use-rundir USE_RUNDIR]
                                [--use-online-psd-file USE_ONLINE_PSD_FILE] [--use-coinc USE_COINC] [--manual-ifo-list MANUAL_IFO_LIST] [--online]
                                [--extra-args-helper EXTRA_ARGS_HELPER] [--manual-postfix MANUAL_POSTFIX] [--gracedb-id GRACEDB_ID] [--gracedb-exe GRACEDB_EXE]
                                [--use-legacy-gracedb] [--internal-use-gracedb-bayestar] [--event-time EVENT_TIME] [--calibration CALIBRATION] [--playground-data]
				[--approx APPROX] [--use-gwsurrogate] [--l-max L_MAX] [--no-matter] [--assume-nospin] [--assume-precessing]
                                [--assume-nonprecessing] [--assume-matter] [--assume-lowlatency-tradeoffs] [--assume-highq] [--assume-well-placed]
                                [--internal-marginalize-distance] [--internal-marginalize-distance-file INTERNAL_MARGINALIZE_DISTANCE_FILE]
                                [--internal-distance-max INTERNAL_DISTANCE_MAX] [--internal-correlate-default]
				[--internal-force-iterations INTERNAL_FORCE_ITERATIONS] [--internal-flat-strategy] [--internal-use-amr]
                                [--internal-use-amr-bank INTERNAL_USE_AMR_BANK] [--internal-use-amr-puff] [--internal-use-aligned-phase-coordinates]
                                [--external-fetch-native-from EXTERNAL_FETCH_NATIVE_FROM] [--internal-propose-converge-last-stage] [--add-extrinsic]
                                [--batch-extrinsic] [--fmin FMIN] [--fmin-template FMIN_TEMPLATE] [--data-LI-seglen DATA_LI_SEGLEN] [--choose-data-LI-seglen]
                                [--fix-bns-sky] [--ile-sampler-method ILE_SAMPLER_METHOD] [--ile-n-eff ILE_N_EFF] [--cip-sampler-method CIP_SAMPLER_METHOD]
				[--cip-fit-method CIP_FIT_METHOD] [--cip-internal-use-eta-in-sampler] [--ile-jobs-per-worker ILE_JOBS_PER_WORKER] [--ile-no-gpu]
                                [--ile-force-gpu] [--fake-data-cache FAKE_DATA_CACHE] [--spin-magnitude-prior SPIN_MAGNITUDE_PRIOR]
                                [--force-chi-max FORCE_CHI_MAX] [--force-mc-range FORCE_MC_RANGE] [--force-eta-range FORCE_ETA_RANGE]
                                [--force-hint-snr FORCE_HINT_SNR] [--force-initial-grid-size FORCE_INITIAL_GRID_SIZE] [--hierarchical-merger-prior-1g]
                                [--hierarchical-merger-prior-2g] [--link-reference-pe] [--link-reference-psds] [--make-bw-psds] [--link-bw-psds]
                                [--use-online-psd] [--ile-retries ILE_RETRIES] [--general-retries GENERAL_RETRIES]
				[--ile-runtime-max-minutes ILE_RUNTIME_MAX_MINUTES] [--fit-save-gp] [--cip-explode-jobs CIP_EXPLODE_JOBS]
                                [--cip-explode-jobs-last CIP_EXPLODE_JOBS_LAST] [--cip-quadratic-first] [--n-output-samples N_OUTPUT_SAMPLES]
                                [--internal-cip-cap-neff INTERNAL_CIP_CAP_NEFF] [--internal-cip-temper-log] [--internal-ile-sky-network-coordinates]
                                [--internal-ile-freezeadapt] [--internal-ile-adapt-log] [--manual-initial-grid MANUAL_INITIAL_GRID]
				[--manual-extra-ile-args MANUAL_EXTRA_ILE_ARGS] [--verbose] [--use-quadratic-early] [--use-gp-early] [--use-cov-early] [--use-osg]
				[--use-osg-file-transfer] [--condor-local-nonworker] [--condor-nogrid-nonworker] [--use-osg-simple-requirements]
                                [--archive-pesummary-label ARCHIVE_PESUMMARY_LABEL] [--archive-pesummary-event-label ARCHIVE_PESUMMARY_EVENT_LABEL]

Expand below for a description of each of the optional arguments:

.. collapse:: Args

	      --h, --help        show this help message and exit
	      --use-production-defaults        Use production defaults. Intended for use with tools like asimov or by nonexperts who just want something to run on a real event. Will require manual setting of other arguments!
	      --use-subdags         Use CEPP_Alternate instead of CEPP_BasicIteration
	      --use-ini USE_INI     Pass ini file for parsing. Intended to reproduce lalinference_pipe functionality. Overrides most other arguments. Full path recommended
	      --use-rundir USE_RUNDIR	   Intended to reproduce lalinference_pipe functionality. Must be absolute path.
	      --use-online-psd-file USE_ONLINE_PSD_FILE	 Provides specific online PSD file, so no downloads are needed
	      --use-coinc USE_COINC 		Intended to reproduce lalinference_pipe functionality
	      --manual-ifo-list MANUAL_IFO_LIST     Overrides IFO list normally retrieve by event ID. Use with care (e.g., glitch studies) or for events specified with --event-time.
	      --online       online
	      --extra-args-helper EXTRA_ARGS_HELPER        Filename with arguments for the helper. Use to provide alternative channel names and other advanced configuration (--channel-name, data type)!
	      --manual-postfix MANUAL_POSTFIX        manual postfix      
	      --gracedb-id GRACEDB_ID        event id from gracebd      
	      --gracedb-exe GRACEDB_EXE        exe from gracedb
	      --use-legacy-gracedb       gracedb legacy       
	      --internal-use-gracedb-bayestar        Retrieve BS skymap from gracedb (bayestar.fits), and use it internally in integration with --use-skymap bayestar.fits.
	      --event-time EVENT_TIME        Event time. Intended to override use of GracedbID. MUST provide --manual-initial-grid
	      --calibration CALIBRATION        calibration
	      --playground-data     Passed through to helper_LDG_events, and changes name prefix
	      --approx APPROX       Approximant. REQUIRED
	      --use-gwsurrogate     Attempt to use gwsurrogate instead of lalsuite.
	      --l-max L_MAX        lmax
	      --no-matter           Force analysis without matter. Really only matters for BNS
	      --assume-nospin       Force analysis with zero spin
	      --assume-precessing   Force analysis *with* transverse spins
	      --assume-nonprecessing        Force analysis *without* transverse spins
	      --assume-matter       Force analysis *with* matter. Really only matters for BNS
	      --assume-lowlatency-tradeoffs        Force analysis with various low-latency tradeoffs (e.g., drop spin 2, use aligned, etc)
	      --assume-highq        Force analysis with the high-q strategy, neglecting spin2. Passed to 'helper'
	      --assume-well-placed  If present, the code will adopt a strategy that assumes the initial grid is very well placed, and will minimize the number of early iterations performed. Not as extrme as --propose-flat-strategy
	      --internal-marginalize-distance	  If present, the code will marginalize over the distance variable. Passed diretly to helper script. Default will be to generate d_marg script *on the fly*
	      --internal-marginalize-distance-file INTERNAL_MARGINALIZE_DISTANCE_FILE	      Filename for marginalization file. You MUST make sure the max distance is set correctly
	      --internal-distance-max INTERNAL_DISTANCE_MAX        If present, the code will use this as the upper limit on distance (overriding the distance maximum in the ini file, or any other setting). *required* to use internal-marginalize-distance in most circumstances
	      --internal-correlate-default			    Force joint sampling in mc,delta_mc, s1z and possibly s2z
	      --internal-force-iterations INTERNAL_FORCE_ITERATIONS	  If integer provided, overrides internal guidance on number of iterations, attempts to force prolonged run. By default puts convergence tests on
	      --internal-flat-strategy	   Use the same CIP options for every iteration, with convergence tests on. Passes --test-convergence,
	      --internal-use-amr    Changes refinement strategy (and initial grid) to use. PRESENTLY WE CAN'T MIX AND MATCH AMR, CIP ITERATIONS, so this is fixed for the whole run right now; use continuation and 'fetch' to augment
	      --internal-use-amr-bank INTERNAL_USE_AMR_BANK	 Bank used for template
	      --internal-use-amr-puff 				 Use puffball with AMR (as usual). May help with stalling
	      --internal-use-aligned-phase-coordinates		 If present, instead of using mc...chi-eff coordinates for aligned spin, will use SM's phase-based coordinates. Requires spin for now
	      --external-fetch-native-from EXTERNAL_FETCH_NATIVE_FROM	     Directory name of run where grids will be retrieved. Recommend this is for an ACTIVE run, or otherwise producing a large grid so the retrieved grid changes/isn't fixed
	      --internal-propose-converge-last-stage 			     Pass through to helper
	      --add-extrinsic         add extrinsic params
	      --batch-extrinsic         batch
	      --fmin FMIN           Mininum frequency for integration. template minimum frequency (we hope) so all modes resolved at this frequency
	      --fmin-template FMIN_TEMPLATE 	       Mininum frequency for template. If provided, then overrides automated settings for fmin-template = fmin/Lmax
	      --data-LI-seglen DATA_LI_SEGLEN	       If specified, passed to the helper. Uses data selection appropriate to LI. Must specify the specific LI seglen used.
	      --choose-data-LI-seglen        choose data seglen
	      --fix-bns-sky       fix bns
	      --ile-sampler-method ILE_SAMPLER_METHOD        choose ile sampler method
	      --ile-n-eff ILE_N_EFF	 ILE n_eff passed to helper/downstream. Default internally is 50; lower is faster but less accurate, going much below 10 could be dangerous
	      --cip-sampler-method CIP_SAMPLER_METHOD        choose cip sampler method
	      --cip-fit-method CIP_FIT_METHOD        choose cip fit method
	      --cip-internal-use-eta-in-sampler	        Use 'eta' as a sampling parameter. Designed to make GMM sampling behave particularly nicely for objects which could be equal mass
	      --ile-jobs-per-worker ILE_JOBS_PER_WORKER    Default will be 20 per worker usually for moderate-speed approximants, and more for very fast configurations
	      --ile-no-gpu        not using gpu during ile stage
	      --ile-force-gpu        force gpu use for ile jobs
	      --fake-data-cache FAKE_DATA_CACHE        fake data cache
	      --spin-magnitude-prior SPIN_MAGNITUDE_PRIOR	  options are default [volumetric for precessing,uniform for aligned], volumetric, uniform_mag_prec, uniform_mag_aligned, zprior_aligned
	      --force-chi-max FORCE_CHI_MAX       		       Provde this value to override the value of chi-max provided
	      --force-mc-range FORCE_MC_RANGE	    Pass this argument through to the helper to set the mc range
	      --force-eta-range FORCE_ETA_RANGE   Pass this argument through to the helper to set the eta range
	      --force-hint-snr FORCE_HINT_SNR	    Pass this argument through to the helper to control source amplitude effects
	      --force-initial-grid-size FORCE_INITIAL_GRID_SIZE      Only used for automated grids. Passes --force-initial-grid-size down to helper
	      --hierarchical-merger-prior-1g			       As in 1903.06742
	      --hierarchical-merger-prior-2g			       As in 1903.06742
	      --link-reference-pe   If present, creates a directory 'reference_pe' and adds symbolic links to fiducial samples. These can be used by the automated plottingcode. Requires LVC_PE_SAMPLES environment variable defined!
	      --link-reference-psds 	 If present, uses the varialbe LVC_PE_CONFIG to find a 'reference_pe_config_map.dat' file, which provides the location for reference PSDs. Will override PSDs used / setup by default
	      --make-bw-psds        If present, adds nodes to create BW PSDs to the dag. If at all possible, avoid this and re-use existing PSDs
	      --link-bw-psds        If present, uses the script retrieve_bw_psd_for_event.sh to find a precomputed BW psd, and convert it to our format
	      --use-online-psd      If present, will use the online PSD estimates
	      --ile-retries ILE_RETRIES          number retries for ile jobs
	      --general-retries GENERAL_RETRIES        number retries general, for DAG
	      --ile-runtime-max-minutes ILE_RUNTIME_MAX_MINUTES	     If not none, kills ILE jobs that take longer than the specified integer number of minutes. Do not use unless an expert
	      --fit-save-gp         If true, pass this argument to CIP. GP plot for each iteration will be saved. Useful for followup investigations or reweighting. Warning: lots of disk space (1G or so per iteration)
	      --cip-explode-jobs CIP_EXPLODE_JOBS        explode jobs cip
	      --cip-explode-jobs-last CIP_EXPLODE_JOBS_LAST    	Number of jobs to use in last stage. Hopefully in future auto-set
	      --cip-quadratic-first        cip quadratic
	      --n-output-samples N_OUTPUT_SAMPLES       Number of output samples generated in the final iteration
	      --internal-cip-cap-neff INTERNAL_CIP_CAP_NEFF   Largest value for CIP n_eff to use for *non-final* iterations. ALWAYS APPLIED.
	      --internal-cip-temper-log			Use temper_log in CIP. Helps stabilize adaptation for high q for example
	      --internal-ile-sky-network-coordinates		Passthrough to ILE
	      --internal-ile-freezeadapt			Passthrough to ILE
	      --internal-ile-adapt-log			Passthrough to ILE
	      --manual-initial-grid MANUAL_INITIAL_GRID	       Filename (full path) to initial grid. Copied into proposed-grid.xml.gz, overwriting any grid assignment done here
	      --manual-extra-ile-args MANUAL_EXTRA_ILE_ARGS	   Avenue to adjoin extra ILE arguments. Needed for unusual configurations (e.g., if channel names are not being selected, etc)
	      --verbose       verbose print everything
	      --use-quadratic-early	  If provided, use a quadratic fit in the early iterations'
	      --use-gp-early        If provided, use a gp fit in the early iterations'
	      --use-cov-early       If provided, use cov fit in the early iterations'
	      --use-osg             Restructuring for ILE on OSG. The code by default will use CVMFS
	      --use-osg-file-transfer		    Restructuring for ILE on OSG. The code will NOT use CVMFS, and instead will try to transfer the frame files.
	      --condor-local-nonworker 	    Provide this option if job will run in non-NFS space.
	      --condor-nogrid-nonworker	    NOW STANDARD, auto-set if you pass use-osg Causes flock_local for 'internal' jobs
	      --use-osg-simple-requirements	    Provide this option if job should use a more aggressive setting for OSG matching
	      --archive-pesummary-label ARCHIVE_PESUMMARY_LABEL	     If provided, creates a 'pesummary' directory and fills it with this run's final output at the end of the run
	      --archive-pesummary-event-label ARCHIVE_PESUMMARY_EVENT_LABEL	     Label to use on the pesummary page itself














			















			  
