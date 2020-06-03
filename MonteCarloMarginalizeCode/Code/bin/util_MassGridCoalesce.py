#! /usr/bin/env python
"""
util_MassGrid.py <fnames>
    - prints out <m1> <m2> <lnL>  <neff> \sigma/L for each individual ILE *mass point*, to stdout
    - fnames can be an arbitrary mix of .sqlite or .xml files
    - currently *hardcodes* intrinsic (mass) parameter list

Used to generate ascii dumps summarizing runs
"""
import sys
import os
import RIFT.misc.xmlutils as xmlutils
from optparse import OptionParser
from glue.ligolw import lsctables, table, utils

import numpy as np
import RIFT.misc.weight_simulations as weight_simulations

# optp= OptionParser()
# opts, args = optp.parse_args()

data_at_intrinsic = {}
for fname in sys.argv[1:]:
    fExtension = fname.split(".")[-1]
    # if .sqlite extension
    if fExtension == "sqlite":
        samples = xmlutils.db_to_samples(fname, lsctables.SnglInspiralTable, ['mass1', 'mass2', 'snr', 'tau0', 'event_duration'])
        for row in samples:  # print each individual row, don't reweight yet
            if not (data_at_intrinsic.has_key((row.mass1,row.mass2))):
                data_at_intrinsic[(row.mass1,row.mass2)] = [[row.snr, row.tau0, row.event_duration]]
            else:
                data_at_intrinsic[(row.mass1,row.mass2)].append([row.snr, row.tau0, row.event_duration])
    # otherwise xml: could have just used ligolw_print
    elif fExtension == "xml" or fExtension == "gz":
        samples = table.get_table(utils.load_filename(fname), lsctables.SnglInspiralTable.tableName)
        for row in samples:  # print each individual row, don't reweight yet
            if not (data_at_intrinsic.has_key((row.mass1,row.mass2))):
                data_at_intrinsic[(row.mass1,row.mass2)] = [[row.snr, row.tau0, row.event_duration]]
            else:
                data_at_intrinsic[(row.mass1,row.mass2)].append([row.snr, row.tau0, row.event_duration])


# Loop over each key and print out
for intrinsic in data_at_intrinsic.keys():
    lnL, neff, sigmaOverL =   np.transpose(data_at_intrinsic[intrinsic])
    lnLmax = np.max(lnL)
    sigma = sigmaOverL*np.exp(lnL-lnLmax)  # remove overall Lmax factor, which factors out from the weights constructed from \sigma
    wts = weight_simulations.AverageSimulationWeights(None, None,sigma)   
    lnLmeanMinusLmax = np.log(np.sum(np.exp(lnL - lnLmax)*wts))
    sigmaNetOverL = (np.sqrt(1./np.sum(1./sigma/sigma)))/np.exp(lnLmeanMinusLmax)
    print(intrinsic[0], intrinsic[1], lnLmeanMinusLmax+lnLmax, np.sum(neff), sigmaNetOverL)


