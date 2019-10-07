

from lalsimutils import *
import emcee

rosUseGaussian = False

Tbuffer =16
fSample = 4096
deltaT = 1./fSample
psd = lalsim.SimNoisePSDiLIGOSRD

fakeSNR = 10
mtot0 = 10.
P= ChooseWaveformParams()
P.m1 = mtot0/2. * lal.LAL_MSUN_SI
P.m2 =P.m1
P.dist = 1e6*lal.LAL_PC_SI
deltaF = findDeltaF(P)

IP = Overlap(fLow=30,  deltaF=deltaF, fNyq=1./deltaT/2.)
hf1 = norm_hoff(P,IP)

P2 = P.copy()

def lnprob(x):
    global P, P2, IP, hf1
    mtot = x[0]
    if rosUseGaussian:
        ret = -0.5 * np.sum( (x-mtot0)**2/2)
    else:
        P2.m1 = mtot/2. * lal.LAL_MSUN_SI
        P2.m2 =P2.m1
        hf2 = norm_hoff(P2, IP)
        ret = np.exp(- fakeSNR*fakeSNR*(1-IP.ip(hf1,hf2)))
    print x, ret
    return ret

ndim, nwalkers = 1,6
p0 = [ np.array([mtot0]) + 1e-4*np.random.random(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(p0,30)

for i in  sampler.flatchain:
    print i[0]

