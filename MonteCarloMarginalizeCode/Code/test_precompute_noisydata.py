
"""
test_precompute_noisydata.py:  Testing the likelihood evaluation with noisy, 3-ifo data.
  - Loads frame data that Chris generated (assumed in signal_hoft) into 3-detector data sets
  - Uses an assumed PSD
  - Constructs a template which *exactly* matches the known source
  - Plots the individual detector lnL_k(t) timeseries and lnL timeseries
    for the injected parameters (i.e., sky location).  Does it work?
"""
import numpy as np
from pylal import Fr

from factored_likelihood import *
from matplotlib import pylab as plt
import sys

#
# Set by user
#
checkResults = True

#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
psd_dict = {}
rhoExpected ={}
rhoExpectedAlt ={}
analyticPSD_Q = True # For simplicity, using an analytic PSD

fminWaves = 25
fminSNR = 25
fSample = 4096

theEpochFiducial = lal.LIGOTimeGPS(1000000014.000000000)   # Use actual injection GPS time (assumed from trigger)

psd_dict['H1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['L1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['V1'] = lalsim.SimNoisePSDiLIGOSRD


# Load H1 data FFT
data_dict['H1'] =frame_data_to_non_herm_hoff("test1.cache", "H1"+":FAKE-STRAIN")
print data_dict['H1'].data.data[10]

sys.exit(0)

# Plot the H1 data (some time)
fvals = data_dict['H1'].deltaF* np.arange(len(data_dict['H1'].data.data))
plt.plot(fvals, np.abs(data_dict['H1'].fvals))
plt.show()



m1 = 4*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
ampO =0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
fref = 100
