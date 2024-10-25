## Introduction
***
LISA-RIFT: A modified version of RIFT for LISA sources. In its current state, it can used for analyzing MBHB signals. If you are using this code, please cite: [Adapting a novel framework for rapid inference of massive black hole binaries for LISA](http://arxiv.org/abs/2410.15542). 

Original RIFT paper: [Rapid and accurate parameter inference for coalescing, precessing compact binaries](http://arxiv.org/abs/1805.10457) ([git repo](https://github.com/oshaughn/research-projects-RIT/ )

## Structure of this repo:
All codes reside in `MonteCarloMarginalizeCode/Code` with all main RIFT executables in the `bin` directory and modifications in the `RIFT/LISA` directory. 

The samplers reside in `RIFT/integrators` directory. The workhorse for LISA-RIFT is the AdaptiveVolume (AV) sampler, based on the [Varaha]((10.1103/PhysRevD.108.023001)) sampler. 

The FFT routines and waveforms calls are in `RIFT/lalsimutils`. The marginalized likelihood codes are in `RIFT/likelihood` with the LISA specific code being `factored_likelihood_LISA.py`.

## Structure of this code:
The underlying algorithm is a two-stage iterative process, in the first stage marginalized likelihood is evaluated for points on a grid and in the second stage the marginalized likelihood values are used to generate posteriors. Both steps are highly parallelizable, enabling this code to use large datasets and costly models for analysis. The main executables are:

First stage: likelihood evaluation (called ILE) `integrate_likelihood_extrinsic_batchmode`<br>
Second stage: interpolation and posterior construction (called CIP) `util_ConstructIntrinsicPosterior_GenericCoordinates.py`

The entire pipeline is created using `util_RIFT_pseudo_pipe.py` with a template ini file for MBHB analysis in `RIFT/LISA/template_ini`

## Setting up a run:
Ingredients to set up a run. 
1) Data: You can use either model waveforms or numerical relativity waveforms as injections. The injections are stored as h5 files. You can use `RIFT/LISA/injections/generate_injections.py` to generate injections.
2) PSDs: The noise curves for each of the three A, E, T channels. They should be in .xml.gz format. `bin/convert_psd_ascii2xml` can be used to convert a PSD in .txt format to .xml.gz format. This will output figures too for sanity checks.
3) ini file: This file contains the options for your run, including priors, template fmin, modes, number of iterations etc.
4) initial grid: The initial grid over mass, spin, and sky location parameters. `bin/util_ManualOverlapGrid.py` can be used to generate this grid, the output will be a .xml.gz file.  You could also use `RIFT/LISA/initial_grid/fisher_errors.py` to generate the initial grid using information from Fisher errors.

These ingredients are then passed to `bin/util_RIFT_pseudo_pipe.py` and it will create a run directory, including submit files. Then all you need to do is submit your run, and after a few hours you can plot your results using `bin/plot_posterior_corner.py` . 

## Future plans
1) Cleaning up the code.
2) Thorough documentation.
3) User friendly options.
4) An end to end example LISA-RIFT run.



