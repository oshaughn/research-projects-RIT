0.0.17.3
------------
development tree is rift_O4c
   - pseudo_pipe backwards compatibility fix for coinc file reading w/o eccentricity. PUFF settings for BNS use both
     LambdaTilde and DeltaLambdaTilde to avoid loss of points. gwsignal phase shift hlm modes (rotation) phi_star=pi/2
     to match convention. CIP/ILE plugin acess without portfolios. pp_RIFT/pp_RIFT_with_ini cleanup.  dag_utils
     periodic_release for ILE file transfer issues.  PUFF lambda1,lambda2 in log coords. helper matter uses
     lambda1,lambda2. **--internal-precompute-ignore-threshold** (default None) to future-protect analysis of
     high-amplitude signals.  asimov-> pesummary handoff of calmarg output. (rc0)
   - pp_RIFT_with_ini improve OSG use, sanity plots, various bugfixes ( mass range, dmax/dmin); GWSignalWriteFrame fixes for inj; PUFF reflection
     fix; various demos (populations) and pp plotting scripts and waveform/CI plot script; pseudo_pipe
     cip-expl-de-jobs-auto last explode sanity for n_eff (rc1)
   - bootstrap asimov using --manual-initial-grid-supplements (new option); subdag memory/disk limits correctly passed;
     wf interface update INRPhenomXAS; mcsamplerNFlow updates; ILE fix precompute-ignore-threshold so non-default value possible.
     calmarg/rift_source interfaces to eccentric waveforms (may need external edits), calmarg argument quoting. cosmo_sourceframe works with
     distance-marginalization. hyperpipe minor. puff file transfer fix.   CIP adds rf_pca, rbf fit options.
     asimov defaults AV/AV no transverse coord as preferred, asimov scitokens; updates for pp test infrastructure (AV allows pinned sky, test/pp localizer
     adjust to be AV-relevant); factored_likelihood/ILE minimize gwsignal import to reduce overhead (pyseobnr
     expensive!); pp_RIFT_with_ini updates/bugfixes from Jake.  ILE_batchmode dmarg NINF-> -inf protect numpy  2.0.
     rift.ini 'n input samples' option to control grid sizes. (rc2)
   - pseudo_pipe quoted waveform arg passing to calibration_reweighting fixes; util_RandomizeOverlapOrder fix in case of
     many workers/few points per worker; remove X509 access; fix e^2 prior normalization (rc3)
   - minor review-requested edits (mostly unused code), but note factored_likelihood precompute was doing redundant
     calculation (since 0.0.17.0). asimov_test_convergence.py. helper corner case if gid input coinc crazy in mc; undo
     RandomizeOverlapOrder --n-min test since needs to be passed consistently, defer until future (rc4)
     
0.0.17.2
------------
development tree is rift_O4c
   - lalsimutils.periodic_params for plot_posterior_corner and puff; pipeline transverse puff better z coordinates **noteworthy**,
     ligo.lw->igwn_ligoow update, backend changes to CIP column order parsing to be more flexible, CIP lambda_min;
     ILE --internal-waveform-extra-kwargs for gwsignal, changing t_window;  pipeline --internal-use-oauthfiles;
     hyperpipe convergence test nodes.  Mean anomaly merge. Disable old ilwdchar-compatible reading to reduce warning
     messages. misc/dag_utils/write_ILE... now quotes, enabling passing arguments. asimov cleanup manual-extra-ile-args
     and allow waveform/arguments. PUFF bugfix (only applied last).  rift_container.def with pyseobnr. asimov with ecc,
     it0 eccentric arch, AV does cupy-based random numbers, fixing conflict with cosmo_sourceframe. dag_utils and related handle osdf
     container identification better, also cip-request-disk. mcsamplerGPU np.isfinite fix. ILE less process_params output
     (rc0)
   - FYI coinc.xml files from MBTA/spiir malformed and produce parse errors.  calmarg: fix misnamed arguments in
     calibration_reweighting causing block, fix handoff from asimov to find bilby ini files; also more RETRY. pp_RIFT_with_ini use
     provided fmin/fmax/seglen correctly. helper initial grids respect force-mc-range. asimov boostrap from previous
     samples, including coinc; note samples_utils gets pesummary/h5 snarfing tool.  eccentricity
     (lalsimutils/pipeline/ILE/CEPP): multiple updates to enable TEOBResumSDali use (pull #104),
     also --internal-cip-use-periodic-ecc-vars and related support, and minor bugfixes associated with meanPerAno merges
     (rc1)
   - PUFF issue resolved (lalsimutils convert_waveform_coordinates corner case for transverse puff+downselect), plus
     provide for \eta=0,1/4 reflections to preserve points; asimov/rift.py logger, bootstraping; dag_utils argument
     quoting to pass waveform dictionaries to ILE; pseudo_pipe/calibration_reweighting take fref, and also make files
     unique. bugfix blocking change with row.alpha for extrinsic export from recent eccentricity merge
     above. calibration_reweighting minor typo fix arg name (blocking error).
     (rc2)
   - asimov/rift.py edit to prevent getting stuck identifying job needing rescue, also more useful error message when
     upstream dependency fail (calmarg needs bilby ini file).
     (rc3)
release is rc3
   
0.0.17.1
------------
development tree is rift_O4c
  - true cosmological distance priors for ILE. Fix --cip-explode-jobs-subdag in cepp_basic to only operate when
    requested.  Update requirements. Note lalsuite==7.23,7.24 releases don't work (and have never worked) due to broken
    executable installs (lal_path2cache): never use those two versions. (rc0)
  - natsort in requirements, EOSManager tabular can extract one EOS realization, puff adds --fail-if-empty option,
    GWSignal/gwsignal interface forcing hlm tapering, plot_posterior_corner access external code to plot other (e.g.,
    exact) marginals, hyperpipe randomize grid*.dat when joining to fix sort issue, hyperpipe
    SINGULARITY_BASE_EXE_DIR_HYPERPIPE, ILE/dmarg tables new cosmo/cosmo_sourceframe priors, pp_RIFT_with_ini boundary
    range fix ported, hyperpipe fix file transfer issues for PUFF outputs (rc1)
release is rc1

0.0.17.0
-----------
development tree is  rift_O4c
   - start roughly 0.0.16.0rc2. asimov integation, documentation, CIP --n-events-to-analyze; mcsamplerNFlow;
     mcsamplerPortfolio; unreliable_oracles including reference samples; implement mcsamplerPortfolio and some oracles
     in ILE and CIP; hyperpipe dag writer integer events; pp_RIFT_with_ini merge; hyperpipe EOS_POST other integrators;
     helper parses d-min from ini file;  hyperpipe change EOS_POST default to include all data; 
     more asimov (n-output-samples, request_disk, additional-files-to-transfer, distance prior, fmax, dmin; gwdata interface), AV integrator pinned params,
     inv-spec-trunc-time access at pipeline, source_redshift in convert_waveform_coordinates vectorized; statutils
     xpy.max fix for new cupy/python; helper_transfer_files access for eg surrogate files; hlmoft **tapering for some
     cases** (ChooseTDModes, SimInspiralTDModesFromPolarizations) not previously start-tapered; 
      Public OSG operation setup, focusing on hyperpipe (condor-local-nonworker-igwn-prefix, etc). 
      Fix barf where pseudo_pipe required coinc with row.alpha4 (eccentricity).
     **bugfix** sign error reflection for some modes (TEOBResumS and SEOBv4HM - not precessing);
     yet more asimov (calmarg, improved rift.py/rift.ini handoff of args, resurrect/completion detection).   ILE hard fail if --use-gwsignal but no
     gwsignal. Tabular EOS fixes. Calmarg fix so weight_files present, and use alt_reweight to avoid prior problems.
     Expose tukey window length to allow top-level user to change it. 
     **important** factored_likelihood change to ComputeModeCrossTermIP to speed up by x2.  (rc0)
  - TEOBResumS external interface bugfixes (m>=0 modes only by accident if aligned; pass phiref too; bugfix data.data missing); asimov interface
    updates in rift.ini/rift.py (approx; cache file collection; q-max; roll off time; rift/environment backend;
    RIFT_BOOLEAN_LIST; 'extra eccentric arguments');  factored_likelihood/internal_hlm_generator now
    requests tapering;  backend issues (e.g., try/except around GetApproximantFromString; lalsimutils/puffball support
    vectorized; mcsamplerPortfolio uses entrypoints; minor scoping issues; code updates to match scipy changes to
    mvnun/integrate.simps; HyperCombine handles empty data; add --internal-XXX-request-memory; r strings to avoid
    warnings; don't call estimateWaveformDuration if not needed to avoid barf if fmin==0; float128/float64; disable
    default printing of long comment/logging messages; str conversion chIeff range fix; upstream change to event_id in coinc.xml). Transverse puffball. cip-explode-jobs-auto-scale.
    **Updates to using NR simulations** from KW. Oracle hill climber improvements.  **Bugfix hoft** : psi not passed
    correctly (only for injections), now have waveform matching; add test/check_waveform_random .  **hlmoft ChooseFDModes**: add
    conditioning to return from ChooseFDModes, work on phase factors/geometry so now excellent matching, including
    fd_centering_factor.  calmarg code clean up  (rc1)
  - public OSG settings (--use-osg-public); basic CIP/ILE subdag system with while loop as option (cip-explode-jobs-subdag) ; minor bugfixes
    (ile-runtime-max-minutes to convergence subdag; remove simple_unique ILE_puff/ILE_fetch argument); minor refactoring
    of ILE nodes in dag writing (internal) in prep for improved subdag system (rc2)

 release is rc2

    
0.0.16.0
-----------
Since last release
    - Start roughly 0.0.15.9 in rift_O4b, merge with 0.0.15.10 into it.  ROMWaveformManager hlmoft backwards compatbile snarf extra options; pp_RIFT_with_ini
      prototype; pp_RIFT minor fixes (lalapps_path2cache); AV eos-tabular-infereence updates; hyperpipe; scitokens;
      fix online PE use case for O4 (PSD; gracedb-id logic if ini provided); request_memory units; EOSPlotUtilities;
      mcsamplerAdaptiveVolume (rc1)
   - merge in 0.0.15.11, as well as below ( ILE_batchmode hard fail on JIT compilation error, from 15.12 below); AV in rosenbrock test; AV n_chunk size for portfolio; hyperpipe
      integer events;  bugfix (-1)^m->(-1)^l reflection for aligned TEOBResumS external call (rc2)
   - asimov integration (as in rift_O4a/0.0.15.12 rc0); documentation update for hyperpipe, etc; CIP --n-events-to-analyze for hyperpipe; 
     expanded asimov integration (from rift_O4c branch) - calmarg, improved rift.py/rift.ini handoff of args

No release intended (modulo LVK requirements), folded into 0.0.17
     
0.0.15.12
-------------
Since last release
   - gracedb get file psd.xml.gz fix for online; ILE hard fail if CUDA/JIT compilation error; bugfix (-1)^m->(-1)^l
     reflection for aligned TEOBResumS external call; asimov integration

No release intended (modulo LVK requirements), folded into above.

0.0.15.11
-----------
Since last release
   - bugfix util_JoinExtrXML to catch last batch; bugfix --calibration-reweighting-initial-extra-args argument passing  (rc1)
   - bugfix Lmag high-order PN coefficient; ourparams glue.ligolw -> ligo.lw (rc2)
   - CI fix so integration test sane (rc3)

Release is rc3

0.0.15.10
-----------
Since last release
   - ChooseFDWaveform J frame and fourier-transform-conditioning fixes as described in T2300304; add RIFT_BOOLEAN_LIST
     environment variable (rc1)
   - getenv=True workaround; CIP spin prior normalization fixes (just needed for evidence); gwsignal implement Lmax in
     hlmoft (rc2)
   - RIFT_GETENV_OSG; enable gwsignal aways even if RIFT_LOWLATENCY active; mcsamplerGPU typo fix (self.n_total) for
     corner use case (rc3)
   - add --allow-subsolar to prevent hardcoding 1Msun limit; dockerfile cleanup; has_GWS scoping fix to avoid crash in
     factored_likelihood; remove glue.ligo_lw reference in util_SimInspiralToCoinc (rc4)
  - bugfixes calmarg, mainly for use-gwsignal which wasn't implemented (rc5)
  - minor fix to extrinsic export scripts to enable arbitrary output sample size (rc6)

Release is rc6

0.0.15.9
-----------
Since last release
   - compatibility minor updates for numpy>=1.24 (see #27); calmarg import for conda build fix; plot_posterior_corner
     psi mod pi plot option; query_singularity_path executable; CI adds test-build.sh (rc1)
   - minor corner-case bugfixes re gpu/cpu typing; ILE_batchmode correctly resets when using GMM each iteration for dL/incl, and
     GMM+force-adapt-all implemented (not silently ignored); misc updates for contemporary online operation (ecp-cert-info; psd inside coinc.xml)
     dump reproducibility info and ini by default; prior_utils better checking cupy active; still more compatibility
     updates for numpy >=1.24, including follow-on changes to mcsamplerGPU n_eff test; placate XML backend change;
     --assume-matter-eos correctly set CIP tide option;  ILE-specific GMM setup cleaned up (rc2)
   - --assume-matter-conservatively (allow crazy tides), --rom-group (gwsurrogate) implies initial tapering, 
      CIP --assume-eos-but-primary-bh, convergence_test_samples JS base 2 not e, vectorized_general_tools histogram try
      to avoid memory errors (rc3)
   - EOSManager+reprimand minor review updates; mcsamplerGPU works on GPUs when fixing parameters; CI updates;
     fix --assume-matter-eos / --assume-eos-but-primary-bh again; CIP lambda export with --use-eos fix; helper fixes for
     --assume-matter-eos; lalsimutils CreateCompatibleComplexOverlap update for
     contemporary python; lmax_nyquist for gwsignal (ILE: --use-gwsignal-lmax-nyquist); helper sets both eta limits on
     initial grid with --force-eta-range; extrinsic export with eccentricity bugfix (rc4)
   - misc hyperpipe/hyperpuff/CEP fixes (filenames/interface issues); plot_posterior_corner allow for composite with
     labelled fields; lalsimutils convert_vector_coordinates prevent fallthrough to non-vectorized; CIP 9-parameter fit
     variable typo; pipeline transfer gp pickle if on OSG; various int casts for modern / in python (rc5)
   - plot_posterior_corner can use composite files with labelled fields; hypercombine product outcome, length
     consistency; EOSManager protect lambda_from_m for BHs; hyperpipe handoffs; CEPP set n_eff ofor last iteration
     tied to cip-explode-jobs-last (rc6)
release is rc6

0.0.15.8
-----------
Since last release
   - bugfix pseudo_pipe so --internal-*-use-lnL passed correctly to helper. CIP_gauss and CQL working correctly (rc1)
   - bugfixes mcsamplerGPU (wrong var name mcsamplerGPU in type check; self.ntotal init at start of loop).
     mcsamplerGPU/statutils protect against cupyx.scipy.special not being present (rc2)
   - documentation; mcsampler GPU/ILE_batchmode exports for use-lnL; dockerfile builds; pipeline --cip-explode-jobs-auto
     to auto-select appropriate CIP worker count; CIP --lnL-downscale-factor to help sample loud signals; pipeline
     --use-downscale-early to auto-select that factor; pipeline can use CIP_gauss in iterations, and can request via     --use-gauss-early,
    merge last TEOBResumS; collections.abc.Iterable for py3.10 support; helper minor misc (rc3) 
   - user control of n-iterations-subdag-max, and puff in all subdag iterations; CIP/fail-unless/n-eff all floating point; plot_posterior_corner.py can use matplotlibrc;
     pipeline correctly reduces goal of labor per worker in many-worker limit; helper edit (tanmay) to help using coinc
     as input; pipeline internal-cip-tripwire and --internal-n-evaluations-per-iteration options; fix extrinsic output
     for binaries with tides; minor misc bugfixes to obscure code paths; bugfix sky rotation and phase rotation; add
     GWSignal interface; tweak zero-spin run settings; ILE add 'supplementary-likelihood-factor' interface to enable
     call to external runtime-specified code; increase worker count for high SNR jobs with cip-explode-jobs-auto; ditto
     more for matter jobs; bugfix tidal export to XML; EOSManager updates.  Note incompatible with lalsuite 7.13 still (rc4)
   - fix CI; dmarg+phasemarg patch from soichiro (nonprecessing only); add --manual-extra-puff-args; add forward-looking approx
     names; add --force-adapt-all to ILE; add non-time-marginalized likelihood output if user is resampling in time
     (i.e., an snr-like output).  EOSManager QueryLS, repirmand and causal-spectral; xml patch for lalsuite >=7.13; cosmo prior
     on gpu debug; dmarg allows pseudo_cosmo prior; misc osg minor updates; ILE --zero-likelihood for testing;
     various --manual-extra-X-args; cal marg from Jake (rc5)
   - fix CI again; hyperpipe/hyperpost, framed for EOS; cal marg debug; Atul EOSManager updates (reprimand, etc); add missing fairdraw code to GMM and AC+lnL
     mcsamplerAC minor normalization cleaning for low-precision GPU arithmetic (right-edge CDF effect); row.time_geocent
     method in lalsimutils (rc6)
   - catch various ILE errors; correctly set n_eff goal for CIP workers for last iteration consistent with
     cip-explode-jobs-last; add (inaccessible) option to manage XPHM version change; bugfix mcsamplerGMM error estimate;
     bugfix mcsamplerGMM in case of use-lnL/use-lnI; NSBH puffball fix lambda_BH=0; calmarg can use --calibration-reweighting-batchsize;
     remove print in hlmoft; calmarg more fixes paths; calibration_reweighting fix missing data for BBHs; add
     RIFT_AVOID_HOSTS variable so user can identify hosts for ILE to avoid (rc7)
   - XPHM J frame workaround draft; cal marg workflow minor fixes so runs; some cupy/cuda workflow  error handling,
     including --force-reset-all from top level (rc8)
   - bugfix argparsing/typo in pipeline from last commits in rc8; add hlmoft conditioning for ChooseFDModes (rc9)
   - cal marg pass --calibration-reweighting-count, alternate recombination methods, access (low-level-only) to
     alternate h_method, and minor bugfixes from typos (rc10)
   - pipeline-level access to XPHM L-frame and condition control; XPHM workaround for L frame; ILE --force-reset-all
     typo fix (rc11)
   - calmarg J--> L frame access (rc12)

release is rc12

0.0.15.7
-----------
Since last release
   - bugfix lalsimutils vectorized coordinate transform (sph coordinate cos_theta_2 use), add test in 'tests' for
     conversion; fix CIP issue where mc prior range could be changed by input grid despite --mc-range;  implement geocenter-time
     posterior option at last stage, along with edits needed to implement (e.g., fairdraw output option for extrinsic
     stage);  bugfix xml event time export; chi_pavg implement updates; EOSManager minor edits; GMM in ILE finer
     control over adaptation variables; overflow protection GMM+ILE now user-accessible; address typing issues in
     mcsampler (returning object type) for selected arguments; bugfix syntax errors introduced in some mergers; 
     CIP/EOSManager methods for quick inference with tabulated sequence of EOSs (EOSSequenceLandry; etc);
     mcsampler avoid infinite loop for 'no contribution to integral,skipping'; CIP_gauss defined (gaussian fit +
     resampling based on gaussian); scitokens-ready ILE  (rc1)
   - dockerfile prototype in this repo; CQL vectorized; OSG updates (local.cache duplication); fix cupy memory warning;
     add CUDA memory limit to avoid landing on overstrained GPU hosts; convergence_test_samples has JS test used elsewhere;
     bugfix mcsamplerGPU adaptive (intermittent array size error); mcsampler default/gpu standardize n_history;
     integrator test update; lalapps_path2cache->lal_path2cache change; TEOBResumS integration as external package;
     new pseudo-cylindrical coordinates; new CIP option to put change-of-coordinates prior reweighting inside adaptive
     integrand, so it is done live instread of at-end reweighting;  fix some fallthroughs in lalsimutils to 'slow' non-vectorized code; update
     vectorized tranform test to cover standard use cases and put into CI; pp_RIFT updates; start sphinx documentation (rc2)
   - CIP running on OSG as option (--use-cip-osg); lalsimutils.convert_waveform_coordinates fix non-production transform
     and update tests; minor bugfixes (formatting ligolw_add arguments; dmarg+sky rotation wasn't coded). **Change default fit to rf**.  cosmo prior
     gpu-ized and exposed for use. Rosenbrock test cleanup for paper. More sphinx documentation. --auto-logarithm-offset
     in ILE (and access via --internal-ile-auto-logarithm-offset).  Tweaks to better automate interpretation of asymmetric binaries like
     BHNS (pipeline  can set lambda1,lambda2 prior upper bounds, --force-chi-small-max, and allow tides only on one
     object).  CIP_gaussian updates. (rc3)
   - helper missing argument for --internal-ile-auto-logarithm-offset; fix --auto-logarithm-offset implementation SNR
     scale (and debugs thereof); ILE request_disk; expose --internal-rotate-phase; add
     --internal-loud-signal-mitigation-suite; add util_ForOSG_MakeTruncatedLocalFramesDir.sh and assocated .py script to
     trim frames for a remote-machine run; pp plot minor typing issues; fix accidental mangling of rosenbrock test
     commit; docs.  (rc4)
   - mcsamplerGPU use-lnL mode, via statutils; pipeline  --cip-sigma-cut,  --scale-mc-range, --internal-ile/cip-use-lnL; intermediate posterior*dat files have tides &
     eccentricity auto-produced (rc5)
   - var name bugfix in helper; uniform lambda prior in iteration 0 option added (rc6)

 Release is rc6, to facilitate early igwn-testing/igwn use. 

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
   - fix another merge problem from CIP above which dropped the gaussians; add ILE hard fail on cuda errors (rc10)
     Probably should have been major release around July 19, 2021

   - GMM updates and bugfixes; 'fetch' mode to grab info from related jobs; chip_av; GP fits informed by lnL errors;
     alternate fits for placement (cov, quadratic), glue->ligo.lw, assorted minor edits
    UWM hackathon outcomes (distance marginalization (Wysocki/Morisaki); AMR grids)   (rc11)

    - **soichiro mu1,mu2 coordinates**; subdag iterate to convergence ('Z'); lalsimutils convert_waveform_coordinates vectorized (duplicate
      implementations for transforms); helper fixes for architecture to use new subdags; ILE_batchmode fix if no events
      to analyze; dag auto-completes if test successful (for subdag system); merge procedure for workers randomizes
      results, so next iteration isn't dominated by one worker; **architecture change** to use transverse spins earlier
      in fit, with suitable prior for sampling, and generally be more efficient for precessing systems; dag checks if composite files are nonempty;
      partial untested import from Vinaya of using Soichiro mu1,mu2 coordinates for util_AMRGrid; 
      **tentative change in 200a505dbad6c3d6911e5043aabfe2880c991545** of xmax in dmarg, pending review [wrong]; 
      pp_RIFT updates including testing d_marg; GMM sampler fix overflow protection, can now analyze high-lnL sources;
      allow last iteration explode size to be larger than others; fix bug with convert_output_format_ile2inference
      introduced by change in upstream astropy; more glue.ligo_lw -> ligo.lw and many changes;  more on 
      util_AMRGrid.py as refinement engine; improved tests for MC integration tools, validating GMM and mcsamplerGPU;
      minor fixes (rc12)

    - pseudo_pipe/helper updates to use ini files/coinc-embedded PSD appropriate to low-latency; lalsimutils update psd
      parser; util_InitMargTable undo tentative change noted in rc12; pp_RIFT more flexible ini file parsing (rc13) 

    - rotated sky coordinates in ILE/ILE_batchmode (not comprehensive, use different adaptation); mcsamplerGPU bugfixes; ILE/ILE_batchmode changes
      to avoid GPU reallocations; bugfixes for join_grids in dag_utils, cepp_basic subdag system and +flock_local for OSG; reduce imports
      and superfluous setup for low latency; only generate hlm(t) once in factored_likelihood; better running variance
      estimate, that GPU-izes; lalsimutils PSD init vectorized; initMargTable save metadata; ongoing increments to
      amrlib/util_AMRGrid; pseudo_pipe 'last-iteration-extrinsic' and 'batched-extrinsic';
      test/provenance to validate information flow; CEPP_basic miscellaneous minor fenceposting
      (--first-iteration-jumpstart); test/ has integrator tests updated (rosenbrock, Ensemble_extended); tools to let
      pseudo_pipe take fake data (and generate coincs); pp plot updates from AY; dmarg fix (soichiro)
      temper-log in all integrators; enhanced initial grids for low mass sources and rf; initial grid wider chieff
      range for low mass pseudo-pipe recommend GMM
      correlate mc,delta,s1z at high q; force-away smaller for low-mass events; integrator CI test (rc14)
      
   - TEOBResumS/TEOBResumSE (eccentricity); MultiMetaPipe; PUFF active in subdag; refactor plot_posterior_corner/samples_utils;
     ILE_batchmode reset sampling if hit certain errors; OSG file transfer mode revitalize (rc15)

  Release is rc15


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
