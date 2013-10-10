import sys
import functools
from optparse import OptionParser

import numpy as np

import lal
import lalsimulation as lalsim
from glue.ligolw import utils, lsctables, table
from matplotlib import pylab as plt

import lalsimutils

__author__ = "R. O'Shaughnessy"

rosUseWindowing = True

#
# Option parsing
#

optp = OptionParser()
optp.add_option("-p", "--psd-file", default="psd.xml.gz",help="instrument=psd-file, e.g. H1=H1_PSD.xml.gz. Can be given multiple times for different instruments.")
opts, args = optp.parse_args()

print opts, args
print opts.psd_file
#
# Load in PSDs
#
psdf = opts.psd_file
psd_dict ={}
psd_dict['H1'] = lalsimutils.get_psd_series_from_xmldoc(psdf, 'H1')
psd_dict['L1'] = lalsimutils.get_psd_series_from_xmldoc(psdf, 'L1')
psd_dict['V1'] = lalsimutils.get_psd_series_from_xmldoc(psdf, 'V1')
fvals = psd_dict['H1'] .deltaF*np.arange(len(psd_dict['H1'].data))

# Report on PSD -- in particular, on the nyquist bin
detectors = psd_dict.keys()
print "  === 'Nyquist bin' report  ==== "
for det in detectors:
    df = psd_dict[det].deltaF
    print det, " has length ", len(psd_dict[det].data), " with nyquist bin value ",  psd_dict[det].data[-1], " which had better be EXACTLY zero for things to work; note second-to-last bin is ", psd_dict[det].data[-2], psd_dict[det].data[-int(10/df)],psd_dict[det].data[-int(30/df)]
print " ... so several bins may be anomalously small ... "
for det in psd_dict.keys():
    pairups = np.transpose(np.array([fvals,psd_dict[det].data]))
    badbins = [k[0] for k in pairups if k[0]>100 and  k[1]<100*psd_dict[det].data[-1]]
    print " highest reliable frequency in ", det, " seems to be ", badbins[0]

# Plot the raw  PSD
print " === Plotting === "
for det in detectors:
    if rosUseWindowing:
        psd_dict[det] = lalsimutils.regularize_psd_series_near_nyquist(psd_dict[det], 80) # zero out 80 hz window near nyquist
    plt.loglog(fvals, psd_dict[det].data,label='psd:'+det)
fn = np.frompyfunc(lambda x: lalsim.SimNoisePSDaLIGOZeroDetHighPower(x) if x>30 else 0,1,1)
#psd_guess = np.log10(np.array(fn(fvals)))
psd_guess = np.zeros(len(fvals))
psd_guess[1:-1] = np.array(map(lalsim.SimNoisePSDaLIGOZeroDetHighPower,fvals[1:-1]))
plt.loglog(fvals,psd_guess,label='analytic')
plt.legend()
#plt.xlim(1e1,1e2)
plt.show()

#
# 'Interpolate' PSD and replot  [option to use the 'regularize']
#
psd_extend = {}
plt.clf()
for det in detectors:
#    plt.loglog(fvals, psd_dict[det].data,label='psd:'+det)
    df = psd_dict[det].deltaF
#    plt.loglog(fvals, psd_dict[det].data,label='psd:'+det)
    psd_extend[det] = lalsimutils.extend_psd_series_to_sampling_requirements(psd_dict[det], df/2, len(psd_dict[det].data)*df)
    fvals2 = df/2*np.arange(len(psd_extend[det]))
    plt.loglog(fvals2, psd_extend[det],label=det)
plt.legend()
plt.show()

#
# Generate an inner product using this extended PSD.  [Only useful if we have debugging on]
#
for det in psd_dict.keys():
    fNyq = float(len(psd_dict[det].data-1)*psd_dict[det].deltaF)
    print det, fNyq
#    IP = lalsimutils.ComplexIP(26., fNyq, psd_extend[det], analyticPSD_Q=False)

#
# Interpolate PSD
# 
#    psd_dict[inst] = lalsimutils.extend_psd_series_to_sampling_requirements(psd_dict[inst], deltaF, deltaF*len(data_dict[inst].data.data))


