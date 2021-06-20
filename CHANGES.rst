
0.0.15.6
-----------
Since last release
   - pseudo and helper (--use-legacy-gracedb at top level); lalsimutils overlaps using psi4 input; pseudo (gwsurrogate
     logic/reference location fixes)  (rc1)
   - pseudo_pipe (path fixes for osg; add --condor-nogrid-nonworker)   (rc2)
   - helper (typo in V1 data lookup/hard fail; fix corner case for burst trigger hard fail; don't override
     --force-eta-range at low mass); CEPP/dag_utils  (--force-gpu-only, some OSG edits to clean requirements; expand
     --condor-nogrid-nonworker to apply to CIP), pseudo_pipe (add --force-hint-snr)  (rc3)
   - typo fix pseudo_pipe missing colon; convert_ile2inference convention change update py3 print; dag_utils fix PUFF
     issue no_grid-> PUFF fails; merge Yelikar edits to pp_RIFT for OSG, NRSur; request_disk option for ILE in
     CEPP/dag_utils for OSG runs; dag_utils add periodic_release etc update for OSG operation (rc4)
   - carriage return bugfixes in convert_output_format ile2inference (rc5)
   - grids from analytic fisher matrix (util_AnalyticFisherGrid) using gwbench; MOG gets latin hypercube sampling;
     new pseudo_pipe_lowlatency for low latency tunings (rc6)
   - ILE, ILE_batchmode (option export extrinsic per iteration; hope to fix cosmo prior sampling; Dan's suggestion to
     remove np.vectorize calls), mcsampler (help export extrinsic per iteration), general analytic fisher matrix via
     gwbench calls (pseudo_pipe option), converter add missing py2->py3 print statements (rc7)
   - remove ^M from CIPs; add eccentricity capability from Champion; add Henshaw/Gerosa chip_avg; CIP can import
     gaussian likelihood fits; ILE extrinsic export bugfix for likelihood export (missing --manual-logarithm-offset
     undo); CIP can use cos_theta1 and cos_theta2 as coordinates for sampling, and these are made default (rc8)
   - fix merge error in CIP in above - travis fail! (rc9)


0.0.15.5
-----------
Since last release
    - pseudo_pipe ini parsing (halting bug if fake-cache used)  (rc1)
    - fix temporary path issue with ini using abs paths, fix typecast to int (rc2)
    - fmax ini file parsing (rc3)

release is rc3

0.0.15.4
-----------
Since last release
    - dag_utils missing 'no_grid' when building extrinsic (halting bug); add runmon interface; lalsimutils list() in hlmoft_SEOB_dict; 
      convert_...all2xml updated (rc1)
    - ini file srate (rc2)
    - CIP/mcsampler cos_theta sampling; pipeline --manual-ifo-list; workers contribute to net goal piecemeal; ini file
      parser can use fake-cache (rc3)
    - waveforms (NRHybSur3dq8Tidal via gwsurrogate; logic for IMRPhenomXP via ModesFromPolarizations; logic for 
      IMRPhenomTP/TPHM); user control over whether pipeline generates precessing analysis (--assume-precessing, --assume-nonprecessing);
      pseudo_pipe minor (full path to target_params, for ini-file operation)  (rc4)
    - waveforms (fix typos with IMRPhenomTP), ILE add --force-gpu-only to hard fail if GPU not used (rc5)
    - pipeline --force-gpu-only; puffball nan checks; pseudo pipe cache if ini logic fix; FrameZeroNoiseSNR 2to3 (rc6)
    - waveforms (ChooseFDModes: PHM,XHM,PXHM, ...), bugfix in --force-gpu-only logic in pseudo_pipe (rc7)
    - waveforms (lalsimutils, fix patch) (rc8)
    - waveforms (still fixing that damn ChooseFDModes patch) (rc9)
    - updating mcsamplerGPU for testing; minor edits to util_CleanILE (skip files of zero length) and ILE
      (--force-gpu-only logic; change some sys.exit(0) to sys.exit(1))  (rc10)
    - tool to save sklearn GPs (not yet used); ILE cupy.show_config; pseudo_pipe not error with --force-gpu-only (rc11)

release is rc11

0.0.15.3
-----------
Since last release
   - range limit on a2 (rc1)
   - more xpy==cupy checks in factored_likelihood, protect a lalsimutils coordinate conversion against error, OSG update
     conventions for using local pool, CEPP add --condor-nogrid-nonworker option to use it, xmlutils fix py3 reduce
     issue (rc2)
   - ILE_batchmode integration window 75ms, xmlutils more missing py2->py3 (rc3)

Release is rc3

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
