#! /usr/bin/env python

import ourparams
import numpy as np
opts,  rosDebugMessagesDictionary = ourparams.ParseStandardArguments()


# Extract trigger SNRs
rhoExpected  = ourparams.PopulateTriggerSNRs(opts)
rho2Net = 0
for det in rhoExpected:
    rho2Net += rhoExpected[det]*rhoExpected[det]

# Estimate the 'beta' parameter needed to regularize the likelihood, so even the max-likelihood event doesn't overwhelm the histogram
#    exp(rho^2/2)^beta ~ nchunk*10
# Add an ad-hoc x2 to be in better agreement with manual calibration for zero-noise MDC event
print np.min([1,2* np.log(opts.nskip*10)*2./rho2Net])
