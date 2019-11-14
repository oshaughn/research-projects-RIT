
# PP plot pipeline

This driver performs the steps necessary to generate "PP plots": consistency tests of the code, using injections from a known prior.
Following usual RIFT convention, we adopt uniform priors in (redshifted/detector-frame) m1,m2, bounded by specific ranges in mc, q.
Spin priors are either uniform in magnitude (if aligned) or volumetric (if precessing), with ranges controlled by chi_max.


## Tutorial

```
# Generate workflow
pp_RIFT --ini sample_pp_config.ini
cd test_pp; condor_submit_dag
```
