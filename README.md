## Introduction
***
LISA-RIFT: A modified version of RIFT for LISA sources. In its current state, it can used for analyzing MBHB signals. If you are using this code, please cite: [Adapting a novel framework for rapid inference of massive black hole binaries for LISA](http://arxiv.org/abs/2410.15542). 

Original RIFT paper: [Rapid and accurate parameter inference for coalescing, precessing compact binaries](http://arxiv.org/abs/1805.10457) ([git repo](https://github.com/oshaughn/research-projects-RIT/ )

## Structure of this repo:
All codes reside in `MonteCarloMarginalizeCode/Code` with all main RIFT executables in the `bin` directory and modifications in `RIFT/LISA` directory.

## Structure of this code:
The underlying algorithm for this code is a two stage iterative, in the first stage marginalized likelihood is evaluated for points on a grid and in the second stage the marginalized likelihood values are used to generate posteriors. Both steps are highly parallelizable, enabling this code to use large datasets and costly models for analysis. The main executables are:
likelihood evaluation (first stage, called ILE): `integrate_likelihood_extrinsic_batchmode`
interpolation and posterior construction (second stage, called CIP): `util_ConstructIntrinsicPosterior_GenericCoordinates.py`

The entire pipeline is created using `util_RIFT_pseudo_pipe.py` with a template ini file in `RIFT/LISA/template_ini`



