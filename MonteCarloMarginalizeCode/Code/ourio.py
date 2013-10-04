#
"""
ourio.py : 
  Input-output routines
"""

import numpy as np


"""
dumpSamplesToFile
Dump output in a format consistent with the bayesian MCMC plotting infrastructure
              This will make uptake and comparison much easier.
FORMAT:
    <key-line>
    <data-line>
where <key-line> must include (in any order)
    indx   logl  m1 m2 ra dec incl phiref psi d(Mpc) t
though in special cases m1, m2 may not be provided.

It is expected that the output will only contain a subset.
The interpretation of our result depends on how many binaries this sample.
For simplicity we will assume samples are distributed througout
"""
def dumpSamplesToFile(fname, samps, labels):
    headline = ' '.join(labels)
    np.savetxt(fname, samps, header=headline,comment='')
    return 0


"""
dumpStatisticsToFile
Dump the maximum likelihood point and 1d width from a set of samples.
This will be used principally to *seed subsequent MC* with good choices, so we converge faster.
[Remember, we are free to choose a sampling distribution for each intrinsic evaluation.]
The routines using this process should take care not to overconverge.
"""
def dumpStatisticsToFile(fname,samps,labels):
    indxmax = np.argmax(samps,-1)
    pairs = [(labels[i], samps[indxmax][i]) for i in np.arange(len(labels))]
    np.savetxt(fname,pairs)
    return 0


def readStatisticsFromFile(fname,samps,labels):
    return 0
