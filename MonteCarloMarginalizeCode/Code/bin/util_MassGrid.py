#! /usr/bin/env python
"""
util_MassGrid.py <fnames>
    - prints out <m1> <m2> <lnL>  <neff> for each individual ILE run, to stdout
    - fnames can be an arbitrary mix of .sqlite or .xml files

Used to generate ascii dumps summarizing runs
"""
import sys
import os
import RIFT.misc.xmlutils as xmutils
from optparse import OptionParser
from glue.ligolw import lsctables, table, utils

# optp= OptionParser()
# opts, args = optp.parse_args()

for fname in sys.argv[1:]:
    fExtension = fname.split(".")[-1]
    # if .sqlite extension
    if fExtension == "sqlite":
        samples = xmlutils.db_to_samples(fname, lsctables.SnglInspiralTable, ['mass1', 'mass2', 'snr', 'tau0', 'event_duration'])
        for row in samples:  # print each individual row, don't reweight yet
            print(row.mass1, row.mass2, row.snr, row.tau0, row.event_duration)
    # otherwise xml: could have just used ligolw_print
    elif fExtension == "xml" or fExtension == "gz":
        samples = table.get_table(utils.load_filename(fname), lsctables.SnglInspiralTable.tableName)
        for row in samples:  # print each individual row, don't reweight yet
            print(row.mass1, row.mass2, row.snr, row.tau0, row.event_duration)

