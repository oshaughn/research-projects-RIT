from lalsimutils import *
from OurMonteCarloIntegrator import *
from OurDistributions import *



# Bootstrap ChooseWaveformParams to pass intrinsic parameter structure
# Bootstrap ChooseWaveform .. to build hlm infrastructure
def margL(P,data,psd,pPrior, pSampler):
    dRef = 100
    # Generate hlm
    hlmOfTime= ChooseTDWaveoformModes(P)
    # Fourier transform hlm
    hlmOfF = []
    
    # Create a target set of time samples to evaluate everything on (and interpolate from)
    timeSamples = []
    # Precompute P,Q on the timeSamples array.  These should be grids (or interpolating functions)
    PofT, QofT = GetModePQ(hlfOfF, data, psd,timeSamples) 
    
    # create object to evaluate Lmodel
    # create object to evaluate Ldata
    lnModel = lambda nhat,khat,psi: hOfF = "stuff"; 
    lnData = lambda x : 0

    # create integrand function object
    theIntegrand  = lambda t,nhat,khat,psi,d : exp(  lnLmodel(nhat,khat,psi) *(dRef/d)**2 + lnData(t,nhat,khat,psi)* (dRef/d))
    # create sampler function object
    theSampler = pSampler.sampler
    # Monte Carlo over all intrinsic parameters
    return integrate(theIntegrand,theSampler, n=1e6)
