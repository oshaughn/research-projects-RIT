#! /usr/bin/env python

import ourparams
import numpy as np
import lal
import sys
opts,  rosDebugMessagesDictionary = ourparams.ParseStandardArguments()


if opts.coinc:
    # Extract trigger SNRs
    rhoExpected  = ourparams.PopulateTriggerSNRs(opts)
    rho2Net = 0
    for det in rhoExpected:
        rho2Net += rhoExpected[det]*rhoExpected[det]


if opts.inj:
    # Read injection parameters
    Psig = ourparams.PopulatePrototypeSignal(opts)
    # Calculate the SNRs, using the PSDs provided? Not now...
    # ...or estimate it, from the masses provided.  Assume an aLIGO-scale range.  Ad-hoc angle-average and factors!
    rho = 8.* ((Psig.m1 + Psig.m2)/lal.LAL_MSUN_SI/1.2) * Psig.dist/(200*1e6*lal.LAL_PC_SI)
    rho2Net = rho*rho

# Estimate the 'beta' parameter needed to regularize the likelihood, so even the max-likelihood event doesn't overwhelm the histogram
#    exp(rho^2/2)^beta ~ nchunk*10
# Add an ad-hoc x2 to be in better agreement with manual calibration for zero-noise MDC event
print np.min([1,2* np.log(opts.nskip*10)*2./rho2Net])
sys.exit(0)
