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

These "args" files are the ones that finally get passed to `create_event_parameter_pipeline_BasicIteration <https://github.com/oshaughn/research-projects-RIT/blob/temp-RIT-Tides-port_python3_restructure_package/MonteCarloMarginalizeCode/Code/bin/create_event_parameter_pipeline_BasicIteration>`_. This script creates a directory for each signal event to analyze, and fills it with the proper structure for RIFT. Each iteration has output directories for ILE, CIP, and convergence test tasks. The directory will also contain scripts to join files containing points at the appropriate time. For more information on the directory structure, see :doc:`Using the pipeline <using-pipeline>`. An example command line for :code:`create_event_parameter_pipeline_BasicIteration` might look like the following:

.. code-block::

   create_event_parameter_pipeline_BasicIteration --input-grid proposed-grid.xml.gz --start-iteration 0 --ile-n-events-to-analyze 20 --ile-exe  `which integrate_likelihood_extrinsic_batchmode`  --ile-args args_ile.txt --request-memory-ILE 4096 --cip-args-list args_cip_list.txt  --test-args args_test.txt --request-memory-CIP 30000  --n-samples-per-job 1000 --working-directory `pwd` --n-iterations 12 --n-copies 2  --puff-exe `which util_ParameterPuffball.py` --puff-max-it 10 --puff-cadence 1 --puff-args args_puff.txt  --convert-args helper_convert_args.txt  --cip-explode-jobs  10 --cip-explode-jobs-dag  --cip-explode-jobs-flat  --cache-file `pwd`/local.cache

The .ini File
-------------

**[pp]**

Specified in this section are the general settings for the injection study you wish to perform. Some arguments include

- n_events - the number of signal injection/recovery events you would like to have
- working_directory - what you would like to call the directory containing your event directories. Number of subdirectories corresponds to :code:`n_events` 
- test_convergence - this gets set to False automatically, but can be changed in the .ini file

**[priors]**

The priors for the signal parameters are specified in this section of the :code:`.ini` file.

**[data]**


**[waveform]**

The waveform section allows the user to specify the waveform used for both injection and recovery. At this time, :code:`pp_RIFT` is only set up to inject and recover with the same waveform. Some waveforms allow or require extra arguments that may be specified in this section. An example is extra parameters (group and param) may be specified for NRSur models. Before using additional paramaters, ensure that :code:`pp_RIFT` is capable of reading them.

- approx - the user specified waveform. Using non-lalsimulation waveforms may require extra setup
- fmin_template

**[make_injections]**


**[make_data]**


**[make_psd]**


**[make_workflow]**







B
