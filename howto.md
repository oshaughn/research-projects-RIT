# Running ILE

These are notes on running the ILE code, and they have been found to work on the machine Bajor.

## Example workflow

1. Create xml injection file (maybe no noise, mdc.xml.gz)

``` bash
>>> util_WriteInjectionFile.py \
--parameter m1 --parameter-value 100 \
--parameter m2 --parameter-value 30 \
--parameter incl --parameter-value 0.785398 \
--parameter dist --parameter-value 568.3 \
--parameter tref --parameter-value 1000000000 \
--approx EOBNRv2HM \
--parameter fmin --parameter-value 10
>>> util_PrintInjection.py --inj mdc.xml.gz --event 0 --verbose
```

2. Create frame file (frames.cache)

```bash
>>> util_LALWriteFrame.py \
--instrument H1 \
--inj mdc.xml.gz \
--event 0 \
--start 999999900 --stop 1000001000
>>> util_LALWriteFrame.py \
--instrument L1 \
--inj mdc.xml.gz \
--event 0 \
--start 999999900 --stop 1000001000
>>> find . -name '*.gwf' | lalapps_path2cache > frames.cache
```

3. Check zero-noise SNR (note the absolute path to Richard's folder)

```bash
>>> cp /home/scott.field/*psd* .
>>> /home/scott.field/util_FrameZeroNoiseSNR.py \
--cache frames.cache \
--psd-file H1=H1-psd.xml.gz \
--psd-file L1=L1-psd.xml.gz
```

4. lays out grid in (M,q)...

```bash
>>> util_ManualOverlapGrid.py \
--inj mdc.xml.gz \
--event 0 \
--parameter 'mtot' --parameter-range '[110,160]' \
--parameter q --parameter-range '[0.1,1]' \
--skip-overlap --verbose --grid-cartesian-npts 500
```

5. Run create_event_dag_via_grid (from run_standard...), which will generate two files: integrate.sub and command-single.sh. Both run the ile code, where sub uses condor and sh can be executed from the commandline. 

```bash
>>> create_event_dag_via_grid \
--cache-file frames.cache   \
--event-time 1000000000  \
--sim-xml overlap-grid.xml.gz \
--fmin-template 20 \
--reference-freq 0 \
--channel-name H1=FAKE-STRAIN \
--channel-name L1=FAKE-STRAIN  \
--psd-file "H1=H1-psd.xml.gz" \
--psd-file "L1=L1-psd.xml.gz" \
--save-samples \
--time-marginalization \
--n-max 500000 \
--n-eff 200 \
--output-file CME.xml.gz \
--n-copies 2 \
--fmax 2000 \
--adapt-weight-exponent 0.2 \
--adapt-floor-level 0.1 \
--n-chunk 4000 \
--approximant EOBNRv2HM \
--l-max 4 \
--declination-cosine-sampler \
--inclination-cosine-sampler  \
--save-P 0.9
```

6. Either submit the dag to condor, or run on the commandline. Since integrate.sub makes use of variables defined in the dag file, it doesn't make sense to directly run these.

Condor:

```bash
>>> condor_submit_dag marginalize_extrinsic_parameters_grid.dag  # this will make many calls to integrate.sub 
>>> condor_q -dag # monitor progress
```

Commandline

```bash
>>> ./command-single.sh >> ile.log # NOTE: this may not be equivalent to the condor run; could result in errors in next step
```

The result should be the log file, CME.xml.gz, and CME.xml.gz.dat. The log-likelihood evaluations are found in the many files that have the form CME.dat. These will be catted into .composite file (next step).

7. Postprocessing (see Jake's tutorial)

```bash
>>> util_ILEdagPostprocess.sh /path/to/ILE/run/directory/ ''label''
```

8. 

```bash
>>> cat run_standard_allmodes_v?_bajor.composite > tmp_allmodes.composite  # joined !
        mkdir run_standard_allmodes; exit 0
        (cd run_standard_allmodes; python ${HERE}/../../python/util_QuadraticMassPosterior.py --inj-file ../mdc.xml.gz --fname ../tmp_allmodes.composite   --coordinates-M-q --n-max 3e6 --n-eff 3000) #   --fit-method gp )
```

## With makefile

These are the basic steps using makefiles. The steps described in Workflow (above) provide details on the key steps automated by the makefiles. If you're just starting out, please follow the workflow instead. 

```bash
make 
make frames.cache
make run_standard
make run_rom

cd into run_*
condor_submit_dag marginalize_extrinsic_parameters_grid.dag
condor_q
```



# Running iterative workflows (in progress)

These are notes on running the draft iterative workflow.    The current version assumes you have followed steps 1-4 above (i.e., generate an injection; make frames; make or copy a PSD; make a default grid).  The pipeline will perform the iterative steps described in Lange et al 2018: run ILE; generate .composite files; perform a GP fit and generate posterior samples; run ILE; ...

If you have access to bajor, you can find several end-to-end demos in
```
/home/oshaughn/unixhome/Projects/LIGO-ILE-Applications/communications/20180806-Me-PipelineDevelopment
```


 1.  Create a file for arguments of ILE and the fitting code.  Note the input and output arguments should not be provided
   ```
     echo X --cache-file frames.cache   \
--event-time 1000000000  \
--fmin-template 20 \
--reference-freq 0 \
--channel-name H1=FAKE-STRAIN \
--channel-name L1=FAKE-STRAIN  \
--psd-file "H1=${PWD}/H1-psd.xml.gz" \
--psd-file "L1=${PWD}/L1-psd.xml.gz" \
--save-samples \
--time-marginalization \
--n-max 500000 \
--n-eff 200 \
--n-copies 2 \
--fmax 2000 \
--adapt-weight-exponent 0.2 \
--adapt-floor-level 0.1 \
--n-chunk 4000 \
--approximant EOBNRv2HM \
--l-max 4 \
--declination-cosine-sampler \
--inclination-cosine-sampler  \
--save-P 0.9 > args_ile.txt
   ```
 1. Create a file for arguments of the fitting code 
   ```
   echo X  --n-chunk 4000 --time-marginalization --sim-xml overlap-grid.xml.gz --save-samples --reference-freq 100.0 --adapt-weight-exponent 0.1  --event-time 1126259462.391000032 --save-P 0.1 --cache-file ${PWD}/local.cache --fmin-template 10.0 --n-max 2000000 --fmax 1700.0 --save-deltalnL inf --l-max 2  --n-eff 100  --approximant SEOBNRv4 --adapt-floor-level 0.1 --maximize-only  --d-max 2000.0  --psd-file H1=${PWD}/H1-psd.xml.gz --psd-file L1=${PWD}/L1-psd.xml.gz --channel-name H1=GDS-CALIB_STRAIN --channel-name L1=GDS-CALIB_STRAIN   > args_cip.txt)
   ```
 1. Run the pipeline generation code
   ```
create_event_parameter_pipeline_BasicIteration --ile-args args_ile.txt --cip-args args_cip.txt  --input-grid ${PWD}/overlap-grid.xml.gz --n-samples-per-job ${NPTS_IT_BIG} --working-directory ${PWD}
   ```
 1. Submit the jobs to condor
   ```
   condor_submit_dag marginalize_intrinsic_parameters_BasicIterationWorkflow.dag
   ```
