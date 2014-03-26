#! /usr/bin/env python
"""
util_MassGrid.py <fnames>
    - prints out <m1> <m2> <lnL>  <neff> for each individual ILE run, to stdout
    - fnames can be an arbitrary mix of .sqlite or .xml files

Used to generate ascii dumps summarizing runs
"""
import sys
import os
import xmlutils
from optparse import OptionParser
from glue.ligolw import lsctables, table, utils

import numpy as np
import weight_simulations

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
    lnL, neff, sigma =   np.transpose(data_at_intrinsic[intrinsic])
    wts = weight_simulations.AverageSimulationWeights(None, None,sigma)
    sigmaNet = np.sqrt(1./np.sum(1./sigma/sigma))
    print intrinsic[0], intrinsic[1], np.log(np.sum(np.exp(lnL)*wts)), np.sum(neff), sigmaNet


