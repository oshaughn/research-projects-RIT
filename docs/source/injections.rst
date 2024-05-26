=================
Signal Injections
=================

There is a useful tool for creating and analyzing CBC injections with RIFT called :code:`pp_RIFT`. This script is located in the `pp directory <https://github.com/oshaughn/research-projects-RIT/tree/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/test/pp>`_. Currently, the script creates fake signals, injects them into noise frames, and performs a RIFT analysis with user specified arguments. These arguments are most easily changed and passed into :code:`pp_RIFT` using an :code:`.ini` file located in the same directory.

pp_RIFT
-------
.. code-block::

   usage: pp_RIFT [-h] [--use-ini USE_INI] [--use-osg] [--add-extrinsic] [--test]

   optional arguments:
   -h, --help         show this help message and exit
   --use-ini USE_INI  Pass ini file for parsing. Intended to reproduce lalinference_pipe functionality. Overrides most other arguments. Full path recommended
   --use-osg          Attempt OSG operation. Command-line level option, not at ini level to make portable results
   --add-extrinsic    Add extrinsic posterior. Corresponds to --add-extrinsic --add-extrinsic-time-resampling --batch-extrinsic for pipeline
   --test             Used to test the pipeline : prints out commands, generates workflows as much as possible without high-cost steps

pp_RIFT builds the analysis directory for a set of injections based on
user-specified inputs from an :code:`ini` file. The user then submits the
analysis using condor. The basic set of commands looks something like:

.. code-block::

   # ensure necessary script add_frames.py is in your path
   export PATH=${PATH}:`pwd`

   # generate the workflow from ini
   pp_RIFT --use-ini sample_pp_config.ini

   # submit the generated workflow
   cd test_pp; condor_submit_dag master.dag

For more information about the :code:`ini` file, see below_. To learn more about the directory structure, see :doc:`using-pipeline`.

Workflow
--------

:code:`pp_RIFT` is a script that take user input options from an ini file and performs the set up required to run a RIFT analysis on fake data with injected signals. The script writes and runs the necessary command lines for standard RIFT scripts.

For a user starting from scratch, :code:`pp_RIFT` first creates signals using `lalsimutils.py <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/RIFT/lalsimutils.py>`_. This python script creates a list of signal waveform parameters based on the user's defined parameter ranges and priors from the `.ini` file. It saves the signal parameters in a waveform table of arrays, which then get written to a RIFT compatible `.xml` file called :code:`mdc.xml.gz`.

Next, :code:`pp_RIFT` writes the signals to frame files using `util_LALWriteFrame.py <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/bin/util_LALWriteFrame.py>`_. This script takes the candidate waveforms and writes them to frame files that can then be injected into noise. The synthetic signals are joined to fiducial noise frames (the path to which is specified in the `.ini` file) using `add_frames.py <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/test/pp/add_frames.py>`_ which is in the :code:`pp_RIFT` directory. The script depends on `pycbc.frame` to join signals and noise into combined frames.

Once the combined frames are created, the full RIFT workflow is built. `helper_LDG_Events.py <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/bin/helper_LDG_Events.py>`_ does the bulk of the work to properly set the arguments for ILE, CIP, and PUFF. :code:`pp_RIFT` takes user options set by the `.ini` file and passes them to the helper. The helper writes the appropriate arguments to text files that get passed to the pipeline constructor. The helper creates the following files:

- helper_ile_args.txt
- helper_cip_args.txt
- helper_cip_arg_list.txt
- helper_test_args.txt
- helper_convert_args.txt
- helper_puff_args.txt

Then, :code:`pp_RIFT` opens and reads the arguments from these files, adds to them if necessary, and creates:

- args_ile.txt
- args_cip_list.txt
- args_puff.txt
- args_test.txt

These "args" files are the ones that finally get passed to `create_event_parameter_pipeline_BasicIteration <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/bin/create_event_parameter_pipeline_BasicIteration>`_. This script creates a directory for each signal event to analyze, and fills it with the proper structure for RIFT. Each iteration has output directories for ILE, CIP, and convergence test tasks. The directory will also contain scripts to join files containing points at the appropriate time. For more information on the directory structure, see :doc:`Using the pipeline <using-pipeline>`. An example command line for :code:`create_event_parameter_pipeline_BasicIteration`, which is automatically run
when the user runs :code:`pp_RIFT`, might look like the following:

.. code-block::

   create_event_parameter_pipeline_BasicIteration --input-grid proposed-grid.xml.gz --start-iteration 0 --ile-n-events-to-analyze 20 --ile-exe  `which integrate_likelihood_extrinsic_batchmode`  --ile-args args_ile.txt --request-memory-ILE 4096 --cip-args-list args_cip_list.txt  --test-args args_test.txt --request-memory-CIP 30000  --n-samples-per-job 1000 --working-directory `pwd` --n-iterations 12 --n-copies 2  --puff-exe `which util_ParameterPuffball.py` --puff-max-it 10 --puff-cadence 1 --puff-args args_puff.txt  --convert-args helper_convert_args.txt  --cip-explode-jobs  10 --cip-explode-jobs-dag  --cip-explode-jobs-flat  --cache-file `pwd`/local.cache

The .ini File
-------------

.. _below:

**[pp]**

Specified in this section are the general settings for the injection study you wish to perform. Some arguments include

- :code:`n_events` - the number of signal injection/recovery events you would like to have
- :code:`working_directory` - what you would like to call the directory containing your event directories. Number of event subdirectories corresponds to :code:`n_events` 
- :code:`test_convergence` - this gets set to False automatically, but can be changed in the .ini file

**[priors]**

The priors for the signal parameters are specified in this section of the :code:`.ini` file. These represent the desired parameter ranges. Prior ranges
**must** match the ranges in your analysis :code:`.SUB` files.

For parameters with supported prior specification in the main version of the
code, see `lines 107-160 <https://git.ligo.org/rapidpe-rift/rift/-/blob/rift_O4b/MonteCarloMarginalizeCode/Code/test/pp/pp_RIFT?ref_type=heads#L107>`_ in :code:`pp_RIFT`. 

**[data]**

The data section contains basic information that :code:`pp_RIFT` needs to call
:code:`util_LALWriteFrame.py` to create the injection frames on which the user will perform the analysis. This requires names for the :code:`ifos`, channel names, frequency range, segment length, and sampling rate. In general you should not need to change these values that much.

**[waveform]**

The waveform section allows the user to specify the waveform used for both injection and recovery. At this time, :code:`pp_RIFT` is only set up to inject and recover with the same waveform (note_). Some waveforms allow or require extra arguments that may be specified in this section. An example is extra parameters (group and param) may be specified for NRSur models. Before using additional paramaters, ensure that :code:`pp_RIFT` is capable of reading them.

- :code:`approx` - the user specified waveform. Using non-lalsimulation waveforms may require extra setup (for example :code:`TEOBResumS`)
- :code:`fmin_template` - minimum frequency for generating waveform
- :code:`lmax` - how many higher order modes

.. _note: For now, to perform recovery on frames with a different waveform, the frames must exist in the directory with the correct structure - meaning that the combined frames reside in a :code:`combine_frames/` directory. If you have previously used :code:`pp_RIFT` to generate the frames, you can copy them to a new directory, change the drectory name and desired recovery waveform in your :code:`.ini` file, and run :code:`pp_RIFT`. Before generating frames, it checks whether frames exist. This is a workaround that will be resolved in future versions of :code:`pp_RIFT`.

**[make_injections]**

Just specifies filename to store injection parameters in, with the structure as required by :code:`lalsimutils` (RIFT) and :code:`ligolw.lsctables` (lalsuite)

**[make_data]**


**[make_psd]**

Must have psds in your working directory with naming convention matching your :code:`.ini` file in order to generate frames.

**[make_workflow]**

These settings are important for generating the settings in the :code:`.SUB` files that are the foundation of your analysis.





