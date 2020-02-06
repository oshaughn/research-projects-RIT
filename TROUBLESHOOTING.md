

## General troubleshooting

## OSG/CVMFS/...

RIFT can run ILE worker jobs on the OSG.  These can fail or be held for a fantastic number of reasons (e.g., missing files that need to be transferred).  We assume you know how to use ``condor_q -run`` and ``condor_q -hold -af HoldReason``, and to read your job condor logs (i.e., ``iteration_0_ile/logs/*.out`` and ``*.log``, respectively)


* **Site-specific problems**: Using ``condor_q -run``, you may notice some sites have an unusually high incidence of problems.  You can avoid specific sites with the UNDESIRED_SITES command.  As an example, here's how to avoid LIGO-WA
```
# as environment variable before creating the workflow
export OSG_UNDESIRED_SITES=LIGO-WA
# in ILE*.sub, if you need to continue a paused workflow or repair a running one
+UNDESIRED_Sites = "LIGO-WA"
```

* **Frame reading failures** : For various reasons, frames may not be accessible. Specific sites may be responsible.  Frame reading failures produce a distinctive ``{}`` into output files (and XLAL errors in the .err files), so you can identify malignant sites with a simple search
```
for i in `grep '{}' iteration_0_ile/logs/*.out | tr ':' ' ' | awk '{print $1}' | sed s/.out/.log/g`; do cat $i; done | grep SiteWMS_Queue
```

* **GPU driver/config problems** : For various reasons, worker jobs may not be able to access GPU code (i.e., driver failure, mismatched/misconfigured host, problems with container, ...).  Worker jobs without GPU configuration will print ``no cupy`` on their first (few) lines.  To identify the log files of jobs with or without CPU , you can do this

 ``
# Find jobs which don't have 'no cupy'
for i in `find . -name 'ILE*.out' -print `; do echo $i ; head -n 1 $i | grep -v 'no cupy' ; done | more
 ``
