research-projects (temp-RIT-Tides and descendants)
research-projects-RIT

Repository for the rapid_PE / RIFT code, developed at RIT (forked long ago from work at UWM/CGCA).

## Installation

Please see INSTALL.md

## Science

If you are using this code for a production analysis, please contact us to make sure you follow the instructions included here!
Also, make sure you cite the relevant rapid_pe/RIFT papers

 1.  Pankow et al 2015
 2.  Lange et al 2018 (RIFT)
 3.  Wysocki et al 2019 (RIFT-GPU)
 4.  Wofford et al 2022 (RIFT-Update)
 5.  O'Shaughnessy et al in prep (RIFT-FinerNet)

When preparing your work, please cite 

 1. the relevant rapid_pe/RIFT papers
     1.  Pankow et al 2015
     2.  Lange et al 2018 (RIFT) [paper](https://arxiv.org/abs/1805.10457)
 
 2. If you are using the surrogate-basis approach, please cite
     3. O'Shaughnessy, Blackman, Field 2017 [paper](http://adsabs.harvard.edu/abs/2017CQGra..34n4002O)

 3. If you are using a GPU-optimized version, please acknowledge
     4.  Wysocki, O'Shaughnessy,  Fong, Lange [paper](https://arxiv.org/abs/1902.04934)

 4. If you are using a non-lalsuite waveform interface, please acknowledge
     1.  gwsurrogate interface: (a) O'Shaughnessy, Blackman, Field 2017 [paper](http://adsabs.harvard.edu/abs/2017CQGra..34n4002O); (b) F. Shaik et al (in prep)
     2.  TEOBResumS interface (original):  Lange et al 2018 RIFT paper
     3.  NR interface (parts inside ILE): Lange et al 2017 PRD 96, 404 [NR comparison methods paper](http://adsabs.harvard.edu/abs/2017PhRvD..96j4041L)
     4.  NRSur7dq2 interface: Lange et al 2018 (RIFT) [paper](https://arxiv.org/abs/1805.10457), and ...
     5.  gwsignal interface: O'Shaughnessy et al in prep ('finer net')
     6.  TEOBResumS eccentric interface: Iglesias et al
     7.  TEOBResumS hyperbolic interface: Henshaw, Lange et al 

  5. If you are using an updated Monte Carlo integration package, please acknowledge the authors; papers will be prepared soon
     1.  GMM integrator: Elizabeth Champion; see [original repo](https://git.ligo.org/benjamin.champion/Monte-Carlo-Integrator), implemented via MonteCarloEnsemble and mcsamplerEnsemble; please cite Ristic et al https://arxiv.org/abs/2105.07013
     2.  GPU MC integrator:  Wysocki, O'Shaughnessy; cite Wofford et al https://dcc.ligo.org/P2200059
     3.  AV integrator: O'Shaughnessy et al  in prep ('finer net'), based on Tiwari et al VARAHA
     4.  Portfolio integrator: ditto
     5.  Oracles: ditto

  6. If you are using a distance-marginalized likeliihood, please acknowledge 
     1. Distance marginalization : Soichiro Morisaki, Dan Wysocki, see  Wofford et al https://dcc.ligo.org/P2200059

  7. If you are using a special coordinate system
     1. Rotated inspiral-phase : cite the original Lee, Morisaki, Tagoshi paper ([journal](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.124057) [dcc](https://dcc.ligo.org/LIGO-P2200037)); also cite Wofford et al  [journal](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.024040) [dcc](https://dcc.ligo.org/P2200059)
     1. Pseudo-cylindrical coordinates: see Wofford et al   [journal](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.024040) [dcc](https://dcc.ligo.org/P2200059)
  
  8. If you are using calibration marginalization, please acknowledge Ethan Payne  (PRD 122004 (2020)), with RIFT integration provided by Jacob Lange and Daniel Williams

  9. If you are using the ``hyperpipeline" for EOS inference or other projects, please acknowledge Kedia et al (arxiv:2405.17326)

 10. If you are using LISA-RIFT, please acknowledge Jan et al 

## Authorlists: Opt-in model
Several aspects of this code are very actively developed.  We encourage  close collaboration with the lead developers (O'Shaughnessy and Lange) to produce the best possible results, particularly given comparatively rapid changes to the interface and pipeline in the past and planned for the future as the user- and developer-base expands.

We expect to make the final developed code widely available and free to use, in a release-based distribution model.  But we're not there yet.  To simplify discussions about authorlist and ettiquete, we have adopted the following simple model, to be revised 4 times per year (1/15, 4/15, 7/15, 10/15):  
 * Free to use: Any code commit older than 3 years from the date an analysis <i>commences</i> is free to use for any scientific work.  
 * Opt-in: Any more recent use should offer (opt-in) authorship to the lead developers, as well as to developers who contributed significantly to features used in the version of the code adopted in an analysis.  Loosely speaking, the newer the features you use, the more proactive you should be in contacting relevant developers, to insure all authors are suitably engaged with the final product.
This policy refers only to commits in this repository, and not to resources and code maintained elsewhere or by other developers (e.g., NR Surrogates), who presumably maintain their own policy.


The following authors should be contacted 
  * O'Shaughnessy and Lange: Iterative pipeline, fitting and posterior generation code, external interfaces (EOB, surrogates)
  * Field, O'Shaughnessy, Blackman: Surrogate basis method 
  * Wysocki, O'Shaughnessy,  Fong, Lange: GPU optimizations
  * ...

## Relationship to rapid_pe
RIFT and rapid_pe are two forks of the original implementation presented in Pankow  et al. 2015.
RIFT and rapid_pe are now disseminated in a common git repository https://git.ligo.org/rapidpe-rift/, and share common code (i.e, the ``integrate_likelihood_extrinsic_batchmode`` executable).


## Version numbers

Short term: roughly major.minor.feature_upgrade.internal_rc_candidates.   So the 4th number are upgraded every few major bugfixes or moves; the 3rd number will upgrade if we add a feature.  We hope to eventually reach version 0.1 for production automated unsupervised analysis during O4

Medium-term: Amajor API change or two we are thinking about for how the users specify workflows should be 0.2

Long-term: Version 1 will reduce dependency on hardcoded parameter names. More flexibility in how inference is done. 
