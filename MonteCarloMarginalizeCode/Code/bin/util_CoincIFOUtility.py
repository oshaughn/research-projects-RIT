#!/usr/bin/env python
#  util_CoincIFOUtility.py
#
#  Usage 1: Print IFOs taking part in a coinc
#  Usage 2: Construct command-line arguments needed for data selection in ILE
#  Usage 3: Construct data taking needed for doubles (unnecessary)



import RIFT.misc.ourparams as ourparams
import numpy as np
opts,  rosDebugMessagesDictionary = ourparams.ParseStandardArguments()

from ligo.lw import utils, lsctables, table, ligolw

if opts.coinc:
    xmldoc = utils.load_filename(opts.coinc)
    coinc_table = lsctables.CoincInspiralTable.get_table(xmldoc)
    assert len(coinc_table) == 1
    coinc_row = coinc_table[0]
       # Populate the SNR sequence and mass sequence
    sngl_inspiral_table = table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
    detList = []
    for sngl_row in sngl_inspiral_table:
        detList.append(str(sngl_row.ifo))

print(detList)
