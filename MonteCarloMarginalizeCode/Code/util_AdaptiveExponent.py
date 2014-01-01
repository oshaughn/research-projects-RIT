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


if opts.inj and opts.channel_name:
    # Read injection parameters
    Psig = ourparams.PopulatePrototypeSignal(opts)
    # Make fake signal.  Use one instrument.
    import factored_likelihood
    import lalsimulation as lalsim
    import lalsimutils

    data_fake_dict ={}
    psd_dict = {}
    rhoFake = {}
    fminSNR =opts.fmin_SNR
    fmaxSNR =opts.fmax_SNR
    fSample = opts.srate
    fNyq = fSample/2.
    rho2Net =0
    # Insure signal duration and hence frequency spacing specified
    Psig.deltaF=None
    df = lalsimutils.findDeltaF(Psig)
    Psig.deltaF = df
    
    for det, chan in map(lambda c: c.split("="), opts.channel_name):
        Psig.detector = det
        data_fake_dict[det] = factored_likelihood.non_herm_hoff(Psig)

        deltaF = data_fake_dict[det].deltaF

        # Read in psd to compute range.  
        if not(opts.psd_file) and not(opts.psd_file_singleifo):
            analyticPSD_Q = True # For simplicity, using an analytic PSD
            psd_dict[det] = lalsim.SimNoisePSDiLIGOSRD   
        else:
            analyticPSD_Q=False
            # Code path #1 : Single PSD file for all instruments
            if not (opts.psd_file_singleifo):
                psd_dict[det] = lalsimutils.get_psd_series_from_xmldoc(opts.psd_file, det)  # pylal type!
            # Code-path #2: List-based procedure to load in individual PSDs for individual instruments
            else: 
                for inst, psdf in map(lambda c: c.split("="), opts.psd_file_singleifo):
                    if inst == det:
                        psd_dict[det] = lalsimutils.get_psd_series_from_xmldoc(psdf, det)  # pylal type!
            fmin = psd_dict[det].f0
            fmax = fmin + psd_dict[det].deltaF*len(psd_dict[det].data)-deltaF
            psd_dict[det] = lalsimutils.resample_psd_series(psd_dict[det], deltaF)
           
            

    for det in data_fake_dict.keys():
        # Create inner product for the detector. Good enough to compute SNR.
        if analyticPSD_Q:
            IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=deltaF,psd=psd_dict[det],fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q)
        else:
            IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=deltaF,psd=psd_dict[det].data.data,fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q)
        rhoFake[det] = IP.norm(data_fake_dict[det])   # Reset

        print rhoFake[det]
        # Construct rho^2
        rho2Net += rhoFake[det]*rhoFake[det]


    # Calculate the SNRs, using the PSDs provided? Not now...
    # ...or estimate it, from the masses provided.  Assume an aLIGO-scale range.  Ad-hoc angle-average and factors!
    #    rho = 8.* ((Psig.m1 + Psig.m2)/lal.LAL_MSUN_SI/2.4) * Psig.dist/(200*1e6*lal.LAL_PC_SI)
    #    rho2Net = rho*rho

# Estimate the 'beta' parameter needed to regularize the likelihood, so even the max-likelihood event doesn't overwhelm the histogram
#    exp(rho^2/2)^beta ~ nchunk*10
# Add an ad-hoc x2 to be in better agreement with manual calibration for zero-noise MDC event
print np.min([0.8,2* np.log(opts.nskip*10)*2./rho2Net])
sys.exit(0)
