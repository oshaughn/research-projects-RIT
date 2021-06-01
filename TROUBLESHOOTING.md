

## General troubleshooting


### Diagnosing typical errors

** Last iterations plot ** : The script ``plot_last_iterations.sh``, available in ``test/pp``, provides a powerful way to render multiple past iterations at the same time.  Sample usage:
``
plot_last_iterations.sh --parameter mc --parameter delta_mc --parameter xi --use-legend --plot-1d-extra  --composite-file all.net --lnL-cut 15 --quantiles None  --ci-list  '[0.9]' 
``
The plots should show a deep red region surrounded by many contours.  For later iterations, these contours should agree.  This figure identifies many common failure modes:
  * ``Too sparse grids``: A sparse initial grid generally can be immediately identified from the density of high-lnL points; the fit generally becomes pathological (spiky delta comb on previous iterations); and the code generally fails after a handful of iterations
  * ``Issues with waveform and settings``: The waveform may not fit in the segment length requested (e.g., at sufficiently low mass).  The waveform may not be able to be generated because initial conditions are too close to merger (e.g., at sufficiently high mass).  These features show up as sharp cutoffs in the point density, where the waveform is failing to generate.  Log files contain ``FAILED ANALYSIS``
  * ``Excessive extrapolation``: The posterior generation code can extrapolate. This can include extrapolation outside the region for which the waveform can generate.  The result is a sharp edge in the 'colored' plot, but a posterior that extends well outside that region.
  * ``Misuse of experimental/nonfunctional options``: The user can request a polynomial approximation to the likelihood (e.g., a quadratic).  This approximation is frequently terrible; enforcing this approximation distorts the posterior accordingly.

## OSG/CVMFS/...

RIFT can run ILE worker jobs on the OSG.  These can fail or be held for a fantastic number of reasons (e.g., missing files that need to be transferred, proxy timeout, etc).  We assume you know how to use ``condor_q -run`` and ``condor_q -hold -af HoldReason``, and to read your job condor logs (i.e., ``iteration_0_ile/logs/*.out`` and ``*.log``, respectively)


* **Site-specific problems**: Using ``condor_q -run``, you may notice some sites have an unusually high incidence of problems.  You can avoid specific sites with the UNDESIRED_SITES command.  As an example, here's how to avoid LIGO-WA
```
# as environment variable before creating the workflow
export OSG_UNDESIRED_SITES=LIGO-WA
# in ILE*.sub, if you need to continue a paused workflow or repair a running one
+UNDESIRED_Sites = "LIGO-WA"
```

* **Frame reading failures** : For various reasons, frames may not be accessible. Specific sites may be responsible.  Frame reading failures produce a distinctive ``{}`` into output files (and XLAL errors in the .err files), so you can identify malignant sites with a simple search
```
# look for {} in cache
for i in `grep '{}' iteration_*_ile/logs/*.out | tr ':' ' ' | awk '{print $1}' | sed s/.out/.log/g`; do cat $i; done | grep SiteWMS_Queue
# look for XLAL in err
for i in `grep 'XLAL' iteration_*_ile/logs/*.err | tr ':' ' ' | awk '{print $1}' | sed s/.err/.log/g`; do cat $i; done | grep SiteWMS_Slot | sort | uniq
```

* **GPU driver/config problems** : For various reasons, worker jobs may not be able to access GPU code (i.e., driver failure, mismatched/misconfigured host, problems with container, ...).  Worker jobs without GPU configuration will print ``no cupy`` on their first (few) lines.  To identify the log files of jobs with or without CPU , you can do this

```
 for i in `find . -name 'ILE*.out' -print `; do echo $i ; head -n 1 $i | grep -v 'no cupy' ; done | more
```


* **Jobs don't queue(proxy timeout)**:  You may have ``JOB_IDS`` empty, and your jobs don't enter the queuing system.  You can't access CVMFS frames without a valid proxy!  You can fix this by renewing your proxy.  
```
ligo_proxy_init
```
(As a rule, do ``grid-proxy-info`` before submitting)


## PP plots

PP plots require *all* runs complete.  By widely exploring parameter space, you may generate configurations that (despite our best efforts) produce hard failures.  (For example, IMRPhenomD throws very hard errors when out of range.)  

**Zero spin PP with convergence testing**: By default, PP plot tests run a fixed number of iterations. For fast zero-spin PP plots, we recommend enabling convergence testing, via

```
for name in analysis*/test.sub;
do
 switcheroo '--always-succeed' ' ' $name
done
```

**Out of range errors**: These typically occur becaue the *puffball* algorithm generates parameters that are out of range.  In past experience, this has occurred for fragile waveforms like IMRPhenomD and randomly-generated high-mass parameters, on the initial random generation.  As a first quick pass to enforce one additional iteration, we effectively disable the first puffball.

  * *Identify candidates*
```
for i in analysis_event_*; do COUNT=`ls $i/post*.dat | wc -l`; if [ ${COUNT} -le 1 ]; then echo ${COUNT} $i `cat $i/iteration*test/logs/*.out`; fi; done
```
  
  * *Force regeneration*
```
for i in analysis_event_*; do COUNT=`ls $i/post*.dat | wc -l`; if [ ${COUNT} -le 1 ]; then echo ${COUNT} $i `cat $i/iteration*test/logs/*.out`; (cd $i; cp overlap-grid-1.xml.gz puffball-1.xml.gz; condor_submit_dag  marginalize_intrinsic_parameters_BasicIterationWorkflow.dag);   fi; done
``

It is possible that runs may fail in later iterations, for reasons other than the test.  That's easily identifed from the convergence test report being larger than the termination threshold.

## GPU/cupy installs
cupy may require cuda==10.0 to work.  

**Option 1**: Need to install cupy-cuda100, and make sure to use a CUDA_DIR for the appropriate installation.

**Option 2**: Make a conda install, and use an appropriate conda configuration. Since this installs from source, that takes time.
$ . /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
$ conda create --clone igwn-py38-20210107 --name igwn-py38-20210107-cupy
$ conda install -c conda-forge cupy cudatoolkit=10.0
