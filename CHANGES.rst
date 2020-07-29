
0.0.15.3
-----------
Since last release
   - range limit on a2 (rc1)
   - more xpy==cupy checks in factored_likelihood, protect a lalsimutils coordinate conversion against error, OSG update
     conventions for using local pool, CEPP add --condor-nogrid-nonworker option to use it, xmlutils fix py3 reduce
     issue (rc2)


0.0.15.2
------------
Since last release
  - minor py3 errors in PP scripts (rc1)
  - import 0.0.14.9rc1-rc5 (rc2)
  - minor py2->3 fixes for merged code. config_yank (rc3)

release is rc3

0.0.15.1
------------
Since last release
   -  ``*NR*`` scripts : fixes for py3/restructure  (rc1)
     another NR fix (not calling py3 version in NRWriteFrame) (rc2)
   - import 0.0.14.8rc1  (rc3)
   - import 0.0.14.8rc2  (rc4)

Release is rc4

0.0.15.0 
---------------------------
Since last release
  - py3 port, including most of changes up to 0.0.14.7 (rc1)
  - py27 import changes through 0.0.14.7rc4.  Add gpytorch. (rc2)
  - py27 import changes through 0.0.14.7rc5 (rc3)
  - minor fixes for latest py3 (func_code->__code__, 'not subscriptable', / float)  (rc4)
release is rc4

0.0.14.9
-----------
Since last release
  - bugfix for parsing ini files (indentation error; handling overspecified channel names); pp OSG; NRWriteFrame latest
    glue; plot_posterior_corner fix tex label issue (rc1)
  - bugfix ini file parsing (not parsing distance-max)   (rc2)
  - bugfix in ini file use (overriding distance-max if ini used) (rc3)
  - change ILE time integration window default to 75 ms. --propose-flat-strategy. Better --internal-correlate-parameters
    arg parsing. Fix enforce_kerr constraint on conversion. RF protect against out of range error. pseudo_pipe GPS->str
    prevent truncation when moving args around.  add --transverse-prior.   helper for mc>25 uses mc/delta_mc instead of
    mc/eta.  Add PEsummary output option.  Add --general-retries. Pass search --hint-snr in pseudo_pipe.
    Pass --fref to convert, so reference spins specified correctly.  Paths for gwsurrogate.   
     Other minor non-ILE/CIP modifications (rc4)
  - infrastructure speed improvements (puffball distance force away function; interpolated cosmology); error protection
    and handling (workarounds for bugs in error handling in lalsuite); CIP always stream error/out; helper updtes (option for
    --assume-well-placed to flatten architectures if exploration needs minimal; bugfix highq strategy transverse spin
    dependence) (rc5)
   

0.0.14.8
-----------
Since last release
    - bugfix pseudo_pipe (space); pp plot puff enforces mc range; OSG updates (option to copy frames, not cvmfs; local workers; requirements avoid blackhole nodes; minor fixes); 
     workflow generation test; bugfix NR script restructure; TROUBLESHOOTING (rc1)
   - helper fix (cache file name had directory prefix at times); docs (rc2)

Release is rc2

0.0.14.7 
--------------------------
   - bugfix in helper introduced by use_ini. PP pipeline. CIP allows arbitrary user-specified priors. Update
     singularity_base_exe. Fix bitrot to old FactoredLogLikelihood.  (rc1)
   - automated PP pipeline. Modify BNS tidal grid. CVMFS frames on OSG. Improve NN.  (rc2)
   - miscellaneous (pp proxies/permissions; pipeline parameter limits; bugfix parsing v4HM in xml) (rc3)
   - NR surrogates (gwsurrogate API update; lalsim calls to surrogates; lalsim surrogate is default in pipeline);
    puffball more flexible; NN/senni update; other minor (option to cap runtime; plotter; V1 sept 2019 channels) (rc4)
   - OSG updates (alt requirements, local universe for non-workers); pp updates (volumetric spins), puffball (force-away),
     periodic_remove option, bugfix for helper logic for first puffball  (rc5)

 Release is rc5

Reminder: 0.0.14.x will be the last versions with py27 support; from version 0.0.15 and upward, we should exclusively use py3

0.0.14.6
---------------------------
Since last release
   - pipline script in main repo (rc1)
   - bugfix in GMM integrator interface; in pipeline interface (rc2)
   - more bugfixes in GMM (rc3)
   - fix access to gp-sparse in CIP
Note GMM, adapt_cart, gp-sparse, rf all validated with this version.
Note 0.0.14.x will be the last versions with py27 support; from version 0.0.15 and upward, we should exclusively use py3


0.0.14.5
---------------------------
Since last release
   - packaging improvements and fixing bugs introduced in restructuring (rc1-rc4)
   - fix bug in mcsamplerEnsemble (used with --sampler-method GMM) (rc5)
   - helper can parse LI ini files 

0.0.14.4 (2019-10-3)
------------------------------
Since last release
  - Adding CI tests
  - minor bugfixes associated with packaging
  - minor bugfixes and improvements [helper grid placement at high mass; lnL cutoff for GMM; C-1_nonlinear frame label; other]

0.0.14.1 (2019-09-30)
------------------------------

  - This is the initial release.  
