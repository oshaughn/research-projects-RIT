from  lalsimutils import *
from marginalized_likelihood import *

P = ChooseWaveformParams()

psd=lalsim.SimNoisePSDiLIGOModel

Lmax = 5

fSample = 4096.


hlms = lalsim.SimInspiralChooseTDModes(P.phiref, P.deltaT, P.m1, P.m2,
                                      # P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, 
                                       P.fmin, P.fref, P.dist,
                                      # P.incl, 
                                       P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
                                       P.ampO, P.phaseO, Lmax, P.approx)


crossTerms = ComputeModeCrossTermIP(hlms, psd, P.fmin, fSample/2.)

print(crossTerms)

