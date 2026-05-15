#! /usr/bin/env python
"""
util_HyperMargGaussian.py
=========================

Thin CLI shim for :class:`RIFT.hyperpipe.drivers.gaussian.GaussianMargDriver`.

This is the toy 3-D Gaussian marg driver for the RIFT hyperpipeline; it
exists as a top-level ``bin/`` executable so the hyperpipe configuration
can reach it via ``which util_HyperMargGaussian.py`` the same way it
reaches ``util_HyperparameterPuffball.py`` etc.

Example
-------
::

    util_HyperMargGaussian.py \\
        --using-eos file:./blind_gaussian_plus_minus.dat \\
        --eos_start_index 0 --eos_end_index 1000 \\
        --outdir Gaussian_example --conforming-output-name \\
        --fname-output-integral f_out_name_1.txt
"""

from RIFT.hyperpipe.drivers.gaussian import main

if __name__ == "__main__":
    main()
