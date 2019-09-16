import sys
import functools
from optparse import OptionParser

import numpy as np

import lal
import lalsimulation as lalsim
from glue.ligolw import utils, lsctables, table
try:
    from matplotlib import pylab as plt
except:
    print(" No interactive plots for you ! ")

import lalsimutils
import factored_likelihood

__author__ = "R. O'Shaughnessy"

rosUseWindowing = True

#
# Option parsing
#

optp = OptionParser()
optp.add_option("-p", "--psd-file", default="psd.xml.gz",help="instrument=psd-file, e.g. H1=H1_PSD.xml.gz. Can be given multiple times for different instruments.")
opts, args = optp.parse_args()

print(opts, args)
print(opts.psd_file)
#
# Load in PSDs: original API
#
# psdf = opts.psd_file
# psd_dict ={}
# psd_dict['H1'] = lalsimutils.get_psd_series_from_xmldoc(psdf, 'H1')
# psd_dict['L1'] = lalsimutils.get_psd_series_from_xmldoc(psdf, 'L1')
# psd_dict['V1'] = lalsimutils.get_psd_series_from_xmldoc(psdf, 'V1')
# fvals = psd_dict['H1'] .deltaF*np.arange(len(psd_dict['H1'].data))

#
# Load in PSDs: convert to swig first
#
psdf = opts.psd_file
psd_dict ={}
psd_dict['H1'] =lalsimutils.pylal_psd_to_swig_psd( lalsimutils.get_psd_series_from_xmldoc(psdf, 'H1'))
psd_dict['L1'] = lalsimutils.pylal_psd_to_swig_psd( lalsimutils.get_psd_series_from_xmldoc(psdf, 'L1'))
psd_dict['V1'] = lalsimutils.pylal_psd_to_swig_psd( lalsimutils.get_psd_series_from_xmldoc(psdf, 'V1'))
fvals = psd_dict['H1'] .deltaF*np.arange(len(psd_dict['H1'].data.data))


# Report on PSD -- in particular, on the nyquist bin
detectors = psd_dict.keys()
print("  === 'Nyquist bin' report  ==== ")
for det in detectors:
    df = psd_dict[det].deltaF
    print(det, " has length ", len(psd_dict[det].data.data), " with nyquist bin value ",  psd_dict[det].data.data[-1], " which had better be EXACTLY zero if we don't explicitly excise it.  Note also  second-to-last bin, etc are also low:  ", psd_dict[det].data.data[-2], psd_dict[det].data.data[-int(10/df)],psd_dict[det].data.data[-int(30/df)])
print(" ... so several bins may be anomalously small ... ")
for det in psd_dict.keys():
    pairups = np.transpose(np.array([fvals,psd_dict[det].data.data]))
    badbins = [k[0] for k in pairups if k[0]>100 and  k[1]<100*psd_dict[det].data.data[-1]]
    print(" highest reliable frequency in ", det, " seems to be ", badbins[0])

# Plot the raw  PSD
print(" === Plotting === ")
for det in detectors:
    if rosUseWindowing:
        tmp =  lalsimutils.regularize_swig_psd_series_near_nyquist(psd_dict[det], 80) # zero out 80 hz window near nyquist
        psd_dict[det] =  lalsimutils.enforce_swig_psd_fmin(tmp, 30.)
    plt.loglog(fvals, psd_dict[det].data.data,label='psd:'+det)
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
print(" === Interpolating and replotting === ")
psd_extend = {}
plt.clf()
for det in detectors:
    df = psd_dict[det].deltaF
    psd_extend[det] = lalsimutils.extend_swig_psd_series_to_sampling_requirements(psd_dict[det], df/2, (len(psd_dict[det].data.data)-1)*df)
    fvals2 = df/2*np.arange(len(psd_extend[det].data.data))
    plt.loglog(fvals2, psd_extend[det].data.data,label=det)
plt.legend()
plt.show()

#
# Generate an inner product using the original discrete PSD
# Perform an inner product using it *and* using an analytic PSD
#
# Populate signal
m1 = 10*lal.LAL_MSUN_SI
m2 = 10*lal.LAL_MSUN_SI

df = psd_dict[detectors[0]].deltaF
fSample = df * 2 *( len(psd_dict[detectors[0]].data.data)-1)  # rescale
print("To construct a signal, we  reconstruct the sampling rate and time window, consistent with the default PSD sampling: (fSample, 1/df) = ", fSample, 1/df)
Psig = lalsimutils.ChooseWaveformParams(
    m1 = m1,m2 =m2,
    fmin = 30, 
    fref=100, ampO=0,
    tref = lal.GPSTimeNow(),   # factored_likelihood requires GPS be assigned 
    radec=True, theta=1.2, phi=2.4,
    detector='H1', 
    dist=25.*1.e6*lal.LAL_PC_SI,
    deltaT=1./fSample,
    deltaF = df
    )
data_dict={}
data_dict['H1'] = lalsimutils.non_herm_hoff(Psig)
Psig.detector = 'L1'
data_dict['L1'] = lalsimutils.non_herm_hoff(Psig)
Psig.detector = 'V1'
data_dict['V1'] = lalsimutils.non_herm_hoff(Psig)
psd_analytic_dict = {}
psd_analytic_dict['H1'] = lalsim.SimNoisePSDaLIGOZeroDetHighPower# lal.LIGOIPsd
psd_analytic_dict['L1'] = lalsim.SimNoisePSDaLIGOZeroDetHighPower #lal.LIGOIPsd
psd_analytic_dict['V1'] = lalsim.SimNoisePSDaLIGOZeroDetHighPower # lal.LIGOIPsd


for det in psd_dict.keys():
    fNyq = (len(psd_dict[det].data.data)-1)*psd_dict[det].deltaF
    print(fNyq)
    print(" Length consistency requirements : ", 2*(len(psd_dict[det].data.data)-1), 2*fNyq/df  -1 , len(data_dict[det].data.data))
    IP = lalsimutils.ComplexIP(fLow=30, fNyq=fNyq,deltaF=df,psd=psd_dict[det].data.data,analyticPSD_Q=False)
    IPAnalytic = lalsimutils.ComplexIP(fLow=30, fNyq=fNyq,deltaF=df,psd=psd_analytic_dict[det],analyticPSD_Q=True)

    rho1 = IP.norm(data_dict[det])
    rho2 = IPAnalytic.norm(data_dict[det])

    print(det, rho1, rho2)

#
# Interpolate PSD and repeat the above test
# 
#    psd_dict[inst] = lalsimutils.extend_psd_series_to_sampling_requirements(psd_dict[inst], deltaF, deltaF*len(data_dict[inst].data.data))
#
# Populate signal
df = psd_extend[detectors[0]].deltaF
fSample = df * 2 *( len(psd_extend[detectors[0]].data.data)-1)  # rescale
print("To construct a signal, we  reconstruct the sampling rate and time window, consistent with the default PSD sampling: (fSample, 1/df) = ", fSample, 1/df)
Psig = lalsimutils.ChooseWaveformParams(
    m1 = m1,m2 =m2,
    fmin = 30, 
    fref=100, ampO=0,
    tref = lal.GPSTimeNow(),   # factored_likelihood requires GPS be assigned 
    radec=True, theta=1.2, phi=2.4,
    detector='H1', 
    dist=25.*1.e6*lal.LAL_PC_SI,
    deltaT=1./fSample,
    deltaF = df
    )
data_dict={}
data_dict['H1'] = factored_likelihood.non_herm_hoff(Psig)
Psig.detector = 'L1'
data_dict['L1'] = factored_likelihood.non_herm_hoff(Psig)
Psig.detector = 'V1'
data_dict['V1'] = factored_likelihood.non_herm_hoff(Psig)
psd_analytic_dict = {}
psd_analytic_dict['H1'] = lalsim.SimNoisePSDaLIGOZeroDetHighPower# lal.LIGOIPsd
psd_analytic_dict['L1'] = lalsim.SimNoisePSDaLIGOZeroDetHighPower #lal.LIGOIPsd
psd_analytic_dict['V1'] = lalsim.SimNoisePSDaLIGOZeroDetHighPower # lal.LIGOIPsd


for det in psd_extend.keys():
    fNyq = (len(psd_extend[det].data.data)-1)*psd_extend[det].deltaF
    print(fNyq)
    print(" Length consistency requirements : ", 2*(len(psd_extend[det].data.data)-1), 2*fNyq/df  -1 , len(data_dict[det].data.data))
    IP = lalsimutils.ComplexIP(fLow=30, fNyq=fNyq,deltaF=df,psd=psd_extend[det].data.data,analyticPSD_Q=False)
    IPAnalytic = lalsimutils.ComplexIP(fLow=30, fNyq=fNyq,deltaF=df,psd=psd_analytic_dict[det],analyticPSD_Q=True)

    rho1 = IP.norm(data_dict[det])
    rho2 = IPAnalytic.norm(data_dict[det])

    print(det, rho1, rho2)


