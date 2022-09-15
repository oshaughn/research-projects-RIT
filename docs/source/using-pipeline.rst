================
Using the pipeline
================

Basics
------

The primary user-interface for this code is a command line tool
:code:`util_RIFT_pseudo_pipe.py` which is available after following the `installation
instructions <installation.txt>`_. To see the help for this tool, run

.. code-block:: console

   $ util_RIFT_pseudo_pipe.py --help

If you leave options unspecified, the pipeline will often try to make guesses about the options you want to use, based
on previous experience performing real GW inference.  We strongly recommend most users fully specify the optionsn
involved via an ini file below.

.. code-block:: console

   $ util_RIFT_pseudo_pipe.py --use-ini my.ini --use-coinc coinc.xml
   # or, for a freestanding example
   $ util_RIFT_pseudo_pipe.py --gracedb-id G329473 --approx IMRPhenomD --calibration C01 --make-bw-psds --l-max 2 --choose-data-LI-seglen

Also important, the code will use selected environment variables to identify optional external dependencies necessary
for various features.  The most important feature for most users is their accounting access information 

.. code-block:: console
  export LIGO_USER_NAME=albert.einstein
  export LIGO_ACCOUNTING=ligo.dev.o4.rift

The pipeline will create a directory structure as follows:

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

The standard RIFT pipeline only works within an HTCondor scheduling environment.  To submit the workflow, use

.. code-block:: console

    $ condor_submit_dag marginalize_intrinsic_parameters_BasicIterationWorkflow.dag 

The ini file format corresponds to the `lalinference ini format
<https://github.com/lscsoft/lalsuite-archive/blob/master/lalapps/src/inspiral/posterior/lalinference_pipe_example.ini>_.
with a single named section of options, corresponding to arguments of the pipeline.

Initialization: PSDs and grids
---------------
The RIFT pipeline by default can attempt to generate a PSD using existing tools.  Most users will want to supply a
contemporary PSD (e.g., a semianlytic PSD for synthetic data; an on-source PSD provided by BW).  However, RIFT uses an
XML format, requiring some conversion for the input PSDs.  The user must often convert these non-pipeline PSDs by hand,
then copy them in (with the names

FINISH THIS

Strongly recommended dependencies
---------------
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
2/
   export PYTHONPATH=${PYTHONPATH}:${GW_SURROGATE}


util_RIFT_pseudo_pipe.py help
---------------

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

optional arguments:
