
# Provenance testing

The executables here give a drop-in replacement for the usual 'bin' directory, allowing you to run standard configurations (which use ``which`` to find executables) **if* your path is set up correctly.  Typically you need to do something like this:
```
export PATH=${RIFT_ DIR}/MonteCarloMarginalizeCode/Code/test/provenance/bin:${PATH}
```
Once enabled, these scripts change utilities so they (a) write their intput and output files to their standard out and (b) create all necessary output files.
For example, the standard tutorial example would be as follows:
```
util_RIFT_pseudo_pipe.py --gracedb-id G329473 --approx IMRPhenomD --calibration C01 --make-bw-psds --l-max 2 --choose-data-LI-seglen --cip-explode-jobs 3 --internal-propose-converge-last-stage --ile-jobs-per-worker 1000 --add-extrinsic
```
With these settings, you can (a) read the dag and (b) read the output and log files to identify the origin of key files, notably the origin of overlap grid files used handoffs like subdags and extrinsic generation.


**Additional notes**
* Use many ILE points per worker: To make the DAG human-browseable, use ``--ile-jobs-per-worker 1000``, to reduce the length of the DAG's worker jobs
* Disable GPUs: We recommend you use CPU-only mode, to maximize resource use and avoid waiting for scarce resources.  For auto-generated pipelines, try
```
for i in ILE*.sub iter*cip/ILE*.sub; do switcheroo 'request_GPUs = 1' '' $i; done
```


ROS note to self: ``LIGO-ILE-Applications/communications/20220430-Me-ProvenanceCheckForWorkflowSanityMainlySubdags``
