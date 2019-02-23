

In this document, we'll walk you through the bare minimum needed to get RIFT up and running.  We won't explain all the settings, but you will get a realistic analysis working.   You can find a more comprehensive and pedagogical tutorial at
```
https://git.ligo.org/pe/tutorials/blob/master/offline_RIFT.md
```

## Installation

See [INSTALL.md](INSTALL.md)

### Confirming your installation works

The following short set of commands will confirm your installation is properly configured; please try in your python version of choice.

```
import numpy as np; import lal; import lalsimulation as lalsim; import lalsimutils
P=lalsimutils.ChooseWaveformParams()
P.m1 = 30*lal.MSUN_SI
P.m2 = 0.6*P.m1
P.approx = lalsim.SEOBNRv4
hlmT = lalsimutils.hlmoft(P)  # Make hlm modes. Confirms you have a version of lalsuite with a compatible waveform interface
lalsimutils.ChooseWaveformParams_array_to_xml([P], "outfile") # write P parameters to xml file. Confirms XML i/o working (glue/ligolw, etc)
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

## Walkthrough of an example on a GraceDB event  (Feb 2019 edition)

IN PROGRESS
