from factored_likelihood import *
from matplotlib import pylab as plt

m1 = 5*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
ampO = 0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
distMpcReference = 100
distMpcFiducial = 25

# optimally oriented source in one detector
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

rhoData = IP.norm(data)  # Amplitude of the actual real signal in the data
print("Amplitude in one IFO : ", rhoData)

Psig.dist = distMpcReference*1.e6*lal.LAL_PC_SI  # Fiducial distance
hlms = hlmoff(Psig,2)
hlm = lalsim.SphHarmFrequencySeriesGetMode(hlms, 2, 2)
rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hlm, data)


# we do have wraparound; we need to force it back around
Tmax = rhoTS.deltaT* len(rhoTS.data.data)
t =map( lambda x:  x if x< Tmax/2 else x-Tmax,  rhoTS.deltaT* np.arange(len(rhoTS.data.data)))

fvals = np.linspace(-2048, 2048, len(t))

plt.figure(1)
plt.plot(t, np.sqrt(np.abs(rhoTS.data.data)*distMpcReference/distMpcFiducial)*np.sqrt(5./(4.*lal.LAL_PI)))  # rescale to match
plt.plot(t, np.abs(rhoData)*np.ones(len(t)))
plt.xlim(-0.05,0.05)
plt.show()

# plt.figure(2)
# plt.plot(fvals, np.abs(hlm.data.data))
# plt.plot(fvals, np.abs(data.data.data))
# plt.show()

