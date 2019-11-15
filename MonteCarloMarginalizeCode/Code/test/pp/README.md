
# PP plot pipeline

This driver performs the steps necessary to generate "PP plots": consistency tests of the code, using injections and recovery from a known prior.
Following usual RIFT convention, we adopt uniform priors in (redshifted/detector-frame) m1,m2, bounded by specific ranges in mc, q.
Spin priors are either uniform in magnitude (if aligned or precessing), with ranges controlled by chi_max.
Lambda priors are uniform up to lambda_max.


## Tutorial

```
# insure add_frames.py is in your path
export PATH=${PATH}:`pwd`
# Generate workflow
pp_RIFT --ini sample_pp_config.ini
cd test_pp; condor_submit_dag master.dag
```
