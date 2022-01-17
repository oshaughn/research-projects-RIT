#! /usr/bin/env python
#
# Do not use this code.

import sys

import numpy as np

from ligo.lw import utils, lsctables, table, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)
from lal.series import read_psd_xmldoc

import lal

import RIFT.lalsimutils as lalsimutils
import RIFT.misc.ourparams as ourparams
opts,  rosDebugMessagesDictionary = ourparams.ParseStandardArguments()

import common_cl

def estimate_adaptive_exponent(rho2Net, nskip):
    """
     Estimate the 'beta' parameter needed to regularize the likelihood, so even the max-likelihood event doesn't overwhelm the histogram Add an ad-hoc x2 to be in better agreement with manual calibration for zero-noise MDC event

    exp(rho^2/2)^beta ~ nchunk*10
    """
    return min(0.8, 2*np.log(nskip*10)*2./rho2Net)

if __file__ == sys.argv[0]:

    rho2Net = 0

    if opts.coinc:
        # Extract trigger SNRs
        ci_table = table.get_table(utils.load_filename(opts.coinc,contenthandler =lalsimutils.cthdler), lsctables.CoincInspiralTable.tableName)
        rho2Net = ci_table[0].snr**2

    if opts.inj and opts.channel_name:
        # Make fake signal.  Use one instrument.
    
        # Read injection parameters
        sim_inspiral_table = table.get_table(utils.load_filename(opts.inj,contenthandler =lalsimutils.cthdler), lsctables.SimInspiralTable.tableName)
        Psig = lalsimutils.ChooseWaveformParams()
        Psig.copy_lsctables_sim_inspiral(sim_inspiral_table[opts.event_id])
    
        data_fake_dict ={}
        psd_dict = {}
        rhoFake = {}
        fminSNR =opts.fmin_SNR
        fmaxSNR =opts.fmax_SNR
        fSample = opts.srate
        fNyq = fSample/2.
        rho2Net =0
        # Insure signal duration and hence frequency spacing specified
        Psig.deltaF = lalsimutils.findDeltaF(Psig)
    
        if opts.psd_file is not None:
            xmldoc = utils.load_filename(opts.psd_file)
            psd_dict = read_psd_xmldoc(xmldoc)

            # Remove unwanted PSDs
            if opts.channel_name is not None:
                dets = common_cl.parse_cl_key_value(opts.channel_name).keys()
                for det in psd_dict.keys():
                    if det not in dets:
                        del psd_dict[det]
                    
        else:
            psd_dict = common_cl.parse_cl_key_value(opts.psd_file_singleifo)
            for det, psdf in psd_dict.items():
                psd_dict[det] = lalsimutils.get_psd_series_from_xmldoc(psdf, det)

        """
        # FIXME: Reenable with ability to parse out individual PSD functions from
        # lalsim --- steal from ligolw_inj_snr
        if not(opts.psd_file) and not(opts.psd_file_singleifo):
            analyticPSD_Q = True # For simplicity, using an analytic PSD
            psd_dict[det] = lalsim.SimNoisePSDiLIGOSRD   
        """

        for det in psd_dict:
            Psig.detector = det
            data_fake_dict[det] = lalsimutils.non_herm_hoff(Psig)

            deltaF = data_fake_dict[det].deltaF
            fmin = psd_dict[det].f0
            fmax = fmin + psd_dict[det].deltaF*len(psd_dict[det].data)-deltaF
            psd_dict[det] = lalsimutils.resample_psd_series(psd_dict[det], deltaF)
        
            rhoFake[det] = lalsimutils.singleIFOSNR(data_fake_dict[det], psd_dict[det], fSample/2, fminSNR, fmaxSNR)

            # FIXME: Reenable with options parsing
            #if opts.verbose:
                #print det, rhoFake[det]

            # Construct rho^2
            rho2Net += rhoFake[det]*rhoFake[det]

     # FIXME: Reenable with options parsing
    #if opts.verbose:
        #print "network, squared: %f" % rho2Net
    print(estimate_adaptive_exponent(rho2Net, opts.nskip))
