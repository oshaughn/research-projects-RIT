In this document, we'll walk you through the bare minimum needed to get RIFT up and running.  We won't explain all the settings, but you will get a realistic analysis working.   You can find a more comprehensive and pedagogical tutorial at
```
https://git.ligo.org/pe/tutorials/blob/master/offline_RIFT.md
```

## Installation and setup

See [INSTALL.md](INSTALL.md) -- this also includes instructions to use pre-built conda and pypi environments.  Remember, if you run on an LDG cluster, you need accounting flags:

```
export LIGO_USER_NAME=albert.einstein  # replace as appropriate
export LIGO_ACCOUNTING=ligo.dev.o3.cbc.pe.lalinferencerapid
```
If you plan on performing production runs in O3, you should be using release code, and a production accounting tag.  On the OSG, you will need to be sure you are using the production release container.  We recommend you create the following script and source it
```
export LIGO_USER_NAME=albert.einstein  # replace as appropriate
export LIGO_ACCOUNTING=ligo.prod.o3.cbc.pe.lalinferencerapid  # production tag
export PS1=${PS1}":(PROD)"   # make sure user knows production accounting is used
SINGULARITY_RIFT_IMAGE=/cvmfs/ligo-containers.opensciencegrid.org/james-clark/research-projects-rit/rift/production
```

### Confirming your installation works

The following short set of commands will confirm your installation is properly configured; please try in your python version of choice.

```
import numpy as np; import lal; import lalsimulation as lalsim; import RIFT.lalsimutils as lalsimutils
P=lalsimutils.ChooseWaveformParams()
P.m1 = 30*lal.MSUN_SI
P.m2 = 0.6*P.m1
P.approx = lalsim.SEOBNRv4
hlmT = lalsimutils.hlmoft(P)  # Make hlm modes. Confirms you have a version of lalsuite with a compatible waveform interface
lalsimutils.ChooseWaveformParams_array_to_xml([P], "outfile") # write P parameters to xml file. Confirms XML i/o working (glue/ligolw, etc)
```


Before getting started, make sure you set 
```
export LIGO_ACCOUNTING=ligo.dev.o3.cbc.pe.lalinferencerapid
export LIGO_USER_NAME=albert.einstein # your own username as appropriate
```

### Possible workarounds (Mar 2019)

The permissions of some executables may not be correctly set. 
If you get an error that a command is not found, please set the execute bit accordingly.
We'll fix this shortly.  For example:

```
chmod a+x research-projects-RIT/MonteCarloMarginalizeCode/Code/util_WriteInjectionFile.py
```

## Walkthrough of an example on synthetic data  (Feb 2019 edition)

The interface to the top-level pipeline is still evolving.  As a concrete example, we'll do an example from the ILE-GPU paper: a massive BBH with no spin.

While it is not necessary, we recommend you log in to a machine with a GPU (e.g., ldas-pcdev13) so you can run one of the individual worker ILE jobs from the command line as a test


### Setting up and understanding the analysis
The following commands set up the workflow. 
```
git  clone https://github.com/oshaughn/ILE-GPU-Paper.git
cd ILE-GPU-Paper/demos/
make test_workflow_batch_gpu_lowlatency
cd test_workflow_batch_gpu_lowlatency; ls
```
To submit the workflow, you can just do
```
condor_submit_dag marginalize_intrinsic_parameters_BasicIterationWorkflow.dag
```
Before you submit a workflow, however, we recommend you first confirm you've set it up correctly, by running one of the worker jobs interactively from the command line.  (This is a great way to catch common configuration errors.)  You'll see a lot of output about reading in data, defining parameters, et cetera.  Wait until you start seeing large arrays of numbers interspersed with the words ``Weight entropy (after histogram) ``
```
./command-single.sh
```
(This command will run anywhere; however, it will only test the GPU configuration if you run it on a machine with a GPU, like pcdev13 or pcdev11 at CIT)

The workflow loosely consists of two parts: worker ILE jobs, which evaluate the marginalized likelihood; and fitting/posterior jobs, which fit the marginalized likelihood and estimate the posterior distribution.  Other nodes help group the output of individual jobs and iterations together.  Unless otherwise noted, all quantities provided are detector-frame;  we haven't added automation for source-frame yet.

In this directory, you'll see
  * ```overlap-grid-0.xml.gz``` : The initial grid used in the iterative analysis. For simplicity, in the Makefile we've set up a cartesian grid in chirp mass and (m1-m2)/M. You're free to use any grid you want (e.g., the output of some previous analysis).  (The workflow can also do the initial grid creation.)
  * *ILE.sub* : The submit file for the individual worker ILE jobs.
  * *CIP.sub* : The submit file for the individual fitting jobs.
  * ```iteration_*```: Directories holding the output of each iteration, including log files.
As the workflow progresses, you'll see the following additional files
  * ```consolidated_*```: These files (particularly those ending in ```.composite```) are the output of each iteration's ILE jobs. Each file is a list of intrinsic parameters, and the value of the marginalized likelihood at those parameters.  (The remaining files provide provenance for  how the .composite file was produced.)
  * ```output-grid-?.xml.gz```: These files are inferred intrinsic, detector-frame posterior distributions from that iteration, expressed as an XML file.
  * ```posterior-samples-*.dat```: These files are reformatted versions of the corresponding XML file, using the command ```convert_output_format_ile2inference```.  This data format should be compatible with LALInference and related postprocessing tools.


The workflow generator command that builds this structure is as follows
```
 create_event_parameter_pipeline_BasicIteration --request-gpu-ILE \
   --ile-n-events-to-analyze ${NPTS_IT} --input-grid overlap-grid-0.xml.gz  
   --ile-exe `which integrate_likelihood_extrinsic_batchmode`  --ile-args args_ile.txt \
   --cip-args args_cip.txt  \
  --test-args args_test.txt --plot-args args_plot.txt --request-memory-CIP ${CIP_MEM} --request-memory-ILE ${ILE_MEM} \
  --input-grid ${PWD}/overlap-grid.xml.gz   --n-samples-per-job ${NPTS_IT}  --n-iterations ${N_IT} \
  --working-directory ${PWD}/$@
```
Key features of this command
  * ```--request-gpu-ILE``` : Self-explanatory. Adds GPU requests to the workflow, and adjusts the ILE command line.
  * ```--cip-args, --ile-args, ...```: Command-line options for the individual progams in the workflow. 
  * ```--n-iterations```: The number of iterations to use
  * ```--n-samples-per-job```: The number of points to analyze in each iteration
  * ```--ile-n-events-to-analyze```: The number of events each worker job analyzes.  For the GPU code, we recommend this should be 20. 

Some other things to note
  * **Exploiting previous work**: Don't waste effort.  If you already have *.composite files from previous analyses with exactly the same settings, join them together and copy them into this directory as ```extra_input.composite```.  The workflow will automatically use this information.  (Alternatively, you can build a workflow that extends a previous workflow, starting at the last iteration from before.)

### Visualizing your results

You can and should use standard PE-compatible tools to manipulate ``posterior-samples-N.dat``.  However, for a quick look, the program ``plot_posterior_corner.py`` can make simple corner plots; the syntax is

```
   plot_posterior_corner.py --posterior-file posterior-samples-5.dat --parameter mc --parameter eta 
```

For the data products made in this example, I  recommend adding the following options

```
 --plot-1d-extra --ci-list  '[0.9]'  --composite-file all.net --quantiles None  --use-legend 
```

Because we produce multiple iterations, you probably want to compare those iterations.  To make the process less laborious, in the ``demos`` directory you will find a script called ``plot_last_iterations_with.sh``.  This will identify all the ``posterior-samples-N.dat`` files and make corner plots of *all* of them, superimposed, with the options you request.  Syntax is the same as plot_posterior_corner.py, except you don't need to specify all the individual files.

### Understanding what just happened
A more thorough discussion of ILE and CIP will be soon found in the main [RIFT LDG tutorial page](https://git.ligo.org/pe/tutorials/blob/master/offline_RIFT.md).

In brief,  the arguments ``--ile-args`` and ``--cip-args`` are most of the arguments to two programs, ```integrate_likelihood_extrinsic_batchmode`` and ``util_ConstructIntrinsicPosterior_GenericCoordinates.py``, called in ILE.sub and CIP.sub.  Let's look at each submit file for this example, and then think about their arguments

#### ILE.sub
After some reformatting for readbility, the submit file for ILE should look something like this
```

arguments = " --output-file CME_out-$(macromassid)-$(cluster)-$(process).xml \
   --event $(macroevent) --n-chunk 10000 \
   --time-marginalization --sim-xml overlap-grid.xml.gz --reference-freq 100.0 --adapt-weight-exponent 0.1 \
  --event-time 1000000014.236547946 --save-P 0.1 
    --cache-file SOME_DIR/zero_noise.cache \
     --fmin-template 10 --n-max 2000000 --fmax 1700.0 
      --save-deltalnL inf --l-max 2 --n-eff 50 --approximant SEOBNRv4 
       --adapt-floor-level 0.1 --maximize-only --d-max 1000 -
        -psd-file H1=SOME_DIR/HLV-ILIGO_PSD.xml.gz --psd-file L1=SOME_DIR/HLV-ILIGO_PSD.xml.gz \
         --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN \
 --inclination-cosine-sampler --declination-cosine-sampler \
--data-start-time 1000000008 --data-end-time 1000000016 --inv-spec-trunc-time 0 \
--no-adapt-after-first --no-adapt-distance --srate 4096 --vectorized --gpu  
 --n-events-to-analyze 20 --sim-xml SOME_DIR/overlap-grid-$(macroiteration).xml.gz  "
request_memory = 2048
request_GPUs = 1

```

The arguments that are most important to understand are as follows:

* `` --fmin-template 10 --fmax 1700.0 ``: The signal starting frequency and the maximum frequency used in the likleihood.  By default, the template starting frequency is equal to the smallest frequency used in integration.
* `` --l-max 2 --approximant SEOBNRv4 ``: Set the modes you are using, and the approximant.
* ``--cache-file ...  --psd-file ... --channel-name ... ``: These arguments specify the specific files identifying the data to analyze and the associated PSD estimates.   
* ``--n-eff 50 --n-max ...``: This sets the termination condition: the code will  stop when it reaches a fixed number of steps, unless the "effective number of samples" n-eff is greater than 50.  (The latter number is roughly 1/(monte carlo error)^2.)

A few arguments are interesting, but you probably don't need to change them.  We have tools that will select this for you automatically:
* ``--data-start-time 1000000008 --data-end-time 1000000016 --inv-spec-trunc-time 0 ``: These determine how much data you will analyze.  You don't have to specify these arguments: the code will pick an amount of data based on the templates being used.  However, using less data  will make the code run a little bit faster.

Finally, we have a bunch of arguments you will almost never change
* ``--save-P 0.1``: Only important with ``--save-samples``, which you should not do yourself.  Only useful for extracting extrinsic parameter information (distance, inclination, etc).  Do not change.
* `` --time-marginalization ``: You must always do this.  It should be a default.
* ``  --inclination-cosine-sampler --declination-cosine-sampler``: You should always do this.  It insures sampling is in cos theta and cos inclination.
* ``   --no-adapt-after-first --no-adapt-distance --srate 4096`` : You should almost always use these arguments.  The last argument sets the sampling rate.  The next two argument insure the adaptive MC integrator  only adjusts its sampling prior for the first point, and only  does so for the two sky location coordinates. 
* `` --adapt-floor-level 0.1 --adapt-weight-exponent 0.1``: You should usually let the expert code choose these for you.  But if you aren't using a very high SNR source, these are good choices.  They change the way the adaptive sampler works.

#### CIP.sub
After some reformatting for readbility, the submit file for CIP should look something like this

```
arguments = "  --fname-output-samples HERE/overlap-grid-$(macroiterationnext) \
 --mc-range  '[23,35]' --eta-range  '[0.20,0.24999]' \
   --parameter mc --parameter-implied eta --parameter-nofit delta_mc \
    --fit-method gp --verbose --lnL-offset 120 --cap-points 12000 \
     --n-output-samples 10000 --no-plots --n-eff 10000 --no-plots  \
      --fname HERE/all.net --fname-output-integral HEREy/overlap-grid-$(macroiterationnext)"
```

The most critical argument 
* ``--fname all.net`` : The filename containing ILE output, one evaluation point per line.  Your workflow will regularly assemble this file, from the output of each iteration.

The next most important options control the coordinate charts in which fitting and MC integration occurs.  Note that because you can perform *dimensional reduction* and use a fitting chart with fewer DOF than your actual problem, you can construct very interesting hierarchical workflows which gradually increase the complexity of your fitting model to address problems with modestly-significant dimensions, or to demonstrate that some DOF explicitly have no impact on results.
* ``--parameter mc --parameter-implied eta`` : The list of ``parameter`` and ``parameter-implied`` describe the coordinate chart used for GP fitting of the evaluation points provided by ILE.  The use of ``parameter-implied'' means that this parameter is derived from another (set of) parameters via a known coordinate chart.



A few options control the Monte Carlo integration, and how many samples you will produce at the end
* ``--parameter mc --parameter-nofit delta_mc``: The list of ``parameter`` and ``parameter-nofit`` describe the coordinate chart used for Monte Carlo integration.  Without exception, the prior must be seperable in this coordinate system (modulo cuts applied at the end to the set of samples).
* ``--n-output-samples``: How many posterior samples the code will try to generate, from the weighted MC samples
* ``--n-eff``: Roughly how many independent samples will exist among the weighted MC samples.  Actually sets the MC error threshold.

A few options control the fitting method
* ``--fit-method gp``: You should almost always do this.
  * ``--lnL-offset``: Remove points with lnL smaller than the maximum value minus lnL-offset, before performing any fitting.  If you have a very, very large number of samples, you can and should adjust this.  If you do, you must insure your result doesn't depend on your choice.
  * ``--cap-points``: If present, and ``fname`` contains more than ``cap-points`` points which satisfy the condition above, then randomly select 


## Walkthrough of an example on a GraceDB event  (Feb 2019 edition, update April 2019)

PLACEHOLDER: Not the best convergence options

If you want the current setup, see the ``LVC-only`` discussion below.
We'll have something working through gwcelery and with a configparser soon.

For now, you can do the following (experimental), which for simplicity uses online PSDs and auto-configured settings to provide a quick estimate using nonprecessing models.   (Warning: will likely modify this soon to use ``--cip-args-list`` rather than ``--cip-args``, to give  flexibility to handle precessing systems with a multi-stage workflow.)

If you want to use this yourself, **please use the ``--observing-run`` ** argument, to prime the auto-selected arguments (channels, etc) to be appropriate to your analysis.

### Generation script
Make the following driver script and call it ``setup_bbh_event.sh``, then do ``chmod a+x setup_bbh_event.sh``
```
mkdir ${1}_analysis_lowlatency
cd ${1}_analysis_lowlatency
helper_LDG_Events.py --use-legacy-gracedb --gracedb-id $1 --use-online-psd --propose-fit-strategy --propose-ile-convergence-options --propose-initial-grid --fmin 20 --fmin-template 20 --working-directory `pwd` --lowlatency-propose-approximant

echo  `cat helper_ile_args.txt`   > args_ile.txt
echo `cat helper_cip_args.txt`  --n-output-samples 5000 --n-eff 5000 --lnL-offset 50 > args_cip.txt
echo "X --always-succeed --method lame  --parameter m1" > args_test.txt

create_event_parameter_pipeline_BasicIteration --request-gpu-ILE --ile-n-events-to-analyze 20 --input-grid proposed-grid.xml.gz --ile-exe  `which integrate_likelihood_extrinsic_batchmode`   --ile-args args_ile.txt --cip-args args_cip.txt --test-args args_test.txt --request-memory-CIP 30000 --request-memory-ILE 4096 --n-samples-per-job 500 --working-directory `pwd` --n-iterations 5 --n-copies 1
```

### Analysis examples

VERIFY OUTPUT CORRECT, adjust number of iterations to be more reasonable.

__A single event__ :Try this, for GW170814 (an example of a triple event, but the coinc.xml/helper is currently only identifying it as an LV double: FIXME)
``
ligo_proxy_init albert.einstein # yes, for some reason you need to do this even on LDG machines
./setup_bbh_event.sh G298172
``

or this, for GW170823 (an example of an HL double event)
``
./setup_bbh_event.sh G298936
``

### More sophisticated demonstrations (LVC-only)
The link
[PSEUDO_ROTA_INSTRUCTIONS.md](https://git.ligo.org/richard-oshaughnessy/rapid_pe_nr_review_o3/blob/master/testing_archival_and_pseudo_online/PseudoOnlineO3/PSEUDO_ROTA_INSTRUCTIONS.md)
provides the location for [``setup_analysis_gracedb_event.py``](https://git.ligo.org/richard-oshaughnessy/rapid_pe_nr_review_o3/blob/master/testing_archival_and_pseudo_online/scripts/setup_analysis_gracedb_event.py) and  instructions on how to use it.   We have developed and tested this script for production use.  Extending the above, it allows you to 
  * efficiently reproduce LI-style settings by choosing the analysis duration time, 
  * tries to adjust some settings (e.g., the starting frequency) based on the trigger time to avoid railing on physics (e.g., you can't always generate a signal at a fixed frequency very close to merger) , and 
   * enables *puffball*, a tool which help randomly explore the parameter space

## Walkthrough of an example which generates synthetic zero-noise nonprecessing BNS (Mar 2019 edition)
Suppose you have a file ``my_injections.xml.gz` of binary neutron star injections.
The following script will generate fake (zero-noise) data and an analysis configuration for it.  The analysis will use SpinTaylorT4 starting at 20 Hz, and assume the binaries don't precess.  Assuming you put the script in the corresponding file, the script will generate one analysis setup with the command
```
 ./setup_synthetic_bns_analysis.sh `pwd`/my_injections.xml.gz 0
```
and similarly for any specific event of interest.  You can loop over several events if you want to perform a large-scale study. 


```
HERE=`pwd`

INJ_XML=$1
EVENT_ID=$2

# Generate PSDs
if [ ! -e ${HERE}/HLV-aLIGO_PSD.xml.gz  ]; then
  ${HERE}/generate_aligo_psd
fi

mkdir synthetic_analysis_bns_event_${EVENT_ID}
cd synthetic_analysis_bns_event_${EVENT_ID}

# Write frame files
START=999999000
STOP=1000000200
EVENT_TIME=1000000000
APPROX=SpinTaylorT4

if [ ! -e local.cache ]; then
 mkdir frames;
 for ifo in H1 L1 V1
 do
 (cd frames;
  util_LALWriteFrame.py --inj  ${INJ_XML} --event ${EVENT_ID} --start ${START} --stop ${STOP} --instrument ${ifo} --approx ${APPROX} #--verbose
 );
 done;
fi
/bin/find frames/ -name '*.gwf' | lalapps_path2cache > local.cache

# Compute SNR (zero noise realization)
if [ ! -e snr_report.txt ]; then 
   util_FrameZeroNoiseSNR.py --cache local.cache --psd-file H1=${HERE}/HLV-aLIGO_PSD.xml.gz --psd-file L1=${HERE}/HLV-aLIGO_PSD.xml.gz  --psd-file V1=${HERE}/HLV-aLIGO_PSD.xml.gz --fmin-snr 20 --fmax-snr 1700 > snr_report.txt
fi 

# extract SNR for file.  Hint used below, so we can be intelligent for very high SNR events
SNR_HERE=`tail -n 1 snr_report.txt | awk '{print $4}' | tr ',' ' '  `

# set up prototype command lines
helper_LDG_Events.py --sim-xml $1 --event $2 --event-time ${EVENT_TIME} --fake-data  --propose-initial-grid --propose-fit-strategy --propose-ile-convergence-options   --fmin 20 --fmin-template 20 --working-directory `pwd` --lowlatency-propose-approximant --psd-file H1=${HERE}/HLV-aLIGO_PSD.xml.gz --psd-file L1=${HERE}/HLV-aLIGO_PSD.xml.gz  --psd-file V1=${HERE}/HLV-aLIGO_PSD.xml.gz  --hint-snr ${SNR_HERE}


echo  `cat helper_ile_args.txt`   > args_ile.txt
echo `cat helper_test_args.txt`   > args_test.txt
# Loop over all helper lines, append this to it
for i in `seq 1 2`  
do 
  echo  --n-output-samples 10000 --n-eff 10000 --n-max 10000000   --downselect-parameter m2 --downselect-parameter-range [1,1000]  --chi-max 0.05
done  > tmp.txt
cat helper_cip_arg_list.txt | paste -d ' ' - tmp.txt  > args_cip_list.txt

create_event_parameter_pipeline_BasicIteration --request-gpu-ILE --ile-n-events-to-analyze 50 --input-grid `pwd`/proposed-grid.xml.gz  --ile-exe  `which integrate_likelihood_extrinsic_batchmode`   --ile-args args_ile.txt --cip-args-list args_cip_list.txt  --test-args args_test.txt --request-memory-CIP 30000 --request-memory-ILE 4096 --n-samples-per-job 1000 --working-directory `pwd` --n-iterations 5 --n-copies 2
```



## Walkthrough of an open-data example (in progress, Mar 2019)

RIFT natively works with standard LIGO frame files.  To analyze open data, we propose you retrieve open data with [gwpy](https://gwpy.github.io/docs/latest/examples/timeseries/public.html), and then [output the data in a frame format](https://gwpy.github.io/docs/stable/timeseries/io.html).  Once you have frame files, you can create ```*.cache``` index files as in the synthetic data example above.

We will post a concrete driver script which performs these steps shortly.   In the meantime, we recommend you propose using FAKE-STRAIN as the channel name, and use --fake-strain as an argument to the helper_LDG_Events script, so the code will correctly match proposed channel names to your frame files.


FINISH WITH CONCRETE DRIVER SCRIPT
