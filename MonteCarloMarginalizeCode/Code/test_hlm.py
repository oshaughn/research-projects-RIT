from factored_likelihood import *
from matplotlib import pylab as plt

m1 = 5*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
ampO = 0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
distMpcReference = 100
distMpcFiducial = 25

Psig = ChooseWaveformParams(fmin = 30., radec=False, incl=0.0, theta=0, phi=0,
         m1=m1,m2=m2,
         ampO=ampO,
        detector='H1', dist=distMpcFiducial*1.e6*lal.LAL_PC_SI)
df = findDeltaF(Psig)
Psig.deltaF = df
Psig.print_params()
data = lsu.non_herm_hoff(Psig)
psd = lal.LIGOIPsd
IP = ComplexOverlap(30, 2048, data.deltaF, psd, True, True)



# Generate and plot hlm(f) and the weights we would use in an inner product
Psig.dist = distMpcReference*1.e6*lal.LAL_PC_SI  # Fiducial distance
hlms = hlmoff(Psig,2)

plt.figure(1)
fvals = np.linspace(-IP.fNyq, IP.fNyq, IP.wvlen)
plt.plot(fvals, 1/np.sqrt(IP.longweights))
plt.xlim(-500,500)
plt.ylim(0, 1e-21)
plt.figure(2)
plt.plot(np.log10(np.abs(fvals)), np.log10(1/np.sqrt(IP.longweights)))
for m in range(-2,3):
    plt.figure(1)
    hlm = lalsim.SphHarmFrequencySeriesGetMode(hlms, 2, m)
    plt.plot(fvals,np.sqrt(np.abs(fvals))*np.abs(hlm.data.data),label=m)
    plt.figure(2)
    plt.plot(np.log10(np.abs(fvals)),np.log10(np.sqrt(np.abs(fvals))*np.abs(hlm.data.data)),label=m)


plt.show()
