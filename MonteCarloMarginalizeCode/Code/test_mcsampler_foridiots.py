
import numpy as np
from matplotlib import pylab as plt

import mcsampler
import ourio

# Specify a CDF
samplerPrior = mcsampler.MCSampler()
samplerPrior.add_parameter('x', np.vectorize(lambda x: 1.3), None, -1.5,1)  # is this correctly renormalized?

# Manual plotting. Note the PDF is NOT renormalized
xvals = np.linspace(-1,1,100)
yvals = samplerPrior.pdf['x'](xvals)
ypvals = samplerPrior.pdf['x'](xvals)/samplerPrior._pdf_norm['x']
zvals = samplerPrior.cdf['x'](xvals)
plt.plot(xvals, yvals,label='pdf shape')
plt.plot(xvals, ypvals,label='pdf')
plt.plot(xvals, zvals,label='cdf')
plt.ylim(0,2)
plt.legend()
plt.show()

# Automated plotting 
sampler = mcsampler.MCSampler()
sampler.add_parameter('x', np.vectorize(lambda x: np.exp(-x**2/2)), None, -1,1)
ourio.plotParameterDistributions('sampler-foridiots-example', sampler, samplerPrior)


# Do an MC integral with this sampler (=the measure specified by the sampler).
ret = samplerPrior.integrate(np.vectorize(lambda x:1), 'x', nmax=1e4)
print "Integral of 1 over this range ", [samplerPrior.llim['x'] ,samplerPrior.rlim['x'] ], " is ", ret, " needs to be ", samplerPrior.rlim['x'] -  samplerPrior.llim['x'] ," and (small)"

# Do an MC integral and return the sampling points
sig = 0.1
res, var, ret, neff = samplerPrior.integrate(np.vectorize(lambda x: np.exp(-x**2/(2*sig**2))), 'x', nmax=5*1e4, full_output=True)
print " integral answer is ", res,  " with expected error ", np.sqrt(var), ";  compare to ", np.sqrt(2*np.pi)*sig
print " note neff is ", neff, "; compare neff^(-1/2) = ", 1/np.sqrt(neff), " to relative predicted and actual errors: ", np.sqrt(var)/res, ", ",  (res - np.sqrt(2*np.pi)*sig)/res
print " -- Sampled array of points  -- "
xvals = np.transpose(ret)[0]
yvals = np.exp(np.transpose(ret)[1])
plt.scatter(xvals,yvals)
plt.show()
