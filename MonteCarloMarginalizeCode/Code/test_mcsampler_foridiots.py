
import numpy as np
from matplotlib import pylab as plt

#import mcsampler
import mcsamplerEnsemble as mcsampler
import ourio

#import dill # so I can pickle lambda functions: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa


# Specify a CDF
samplerPrior = mcsampler.MCSampler()
samplerPrior.add_parameter('x', np.vectorize(lambda x: 1.3), None, -1.5,1)  # is this correctly renormalized?

# Do an MC integral with this sampler (=the measure specified by the sampler).
ret = samplerPrior.integrate(np.vectorize(lambda x:1.0), 'x', nmax=1e4,verbose=True)
print("Integral of 1 over this range ", [samplerPrior.llim['x'] ,samplerPrior.rlim['x'] ], " is ", ret, " needs to be ", samplerPrior.rlim['x'] -  samplerPrior.llim['x'] ," and (small)")
# do an integral with a different prior
samplerNewPrior = mcsampler.MCSampler()
samplerNewPrior.add_parameter('y', np.vectorize(lambda x: np.exp(-x**2/2.)), None, -1,1, prior_pdf = np.vectorize(lambda x: 1/2.)) # normalized prior!
ret = samplerNewPrior.integrate(np.vectorize(lambda x:1), 'y', nmax=1e4)
print("Integral of 1 over a normalized pdf  is ", ret, " and needs to be 1")


# Manual plotting. Note the PDF is NOT renormalized
try:
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
except:
    print(" Alternate interfae does not support raw access to priors ")

try:
    # Automated plotting 
    sampler = mcsampler.MCSampler()
    sampler.add_parameter('x', np.vectorize(lambda x: np.exp(-x**2/2)), None, -1,1)
    ourio.plotParameterDistributions('sampler-foridiots-example', sampler)
except:
    print(" Alternate interface does not support alternate plots")

#print "Integral of 1 over this range ", [samplerPrior.llim['y'] ,samplerPrior.rlim['y'] ], " is ", ret, " needs to be 1, because I used a normalized prior"



# Do an MC integral and return the sampling points
# Stop after I reach neff = 100 points (should be fast!)
sig = 0.1
print(" -- Performing integral, stopping after neff = 100 points -- ")
res, var, ret, lnLmarg, neff = samplerPrior.integrate(np.vectorize(lambda x: np.exp(-x**2/(2*sig**2))), 'x', nmax=5*1e4, full_output=True,neff=100)
print(" integral answer is ", res,  " with expected error ", np.sqrt(var), ";  compare to ", np.sqrt(2*np.pi)*sig)
print(" note neff is ", neff, "; compare neff^(-1/2) = ", 1/np.sqrt(neff), " to relative predicted and actual errors: ", np.sqrt(var)/res, ", ",  (res - np.sqrt(2*np.pi)*sig)/res)
# Save the sampled points to a file
ourio.dumpSamplesToFile("sampler-foridiots-example.dat", ret, ['x', 'lnL'])
# plot the array of partial sums, to illustrate convergence
print(" -- Integral convergence ---")
xarr, lnL = np.transpose(ret)
lnLmax = np.maximum.accumulate(lnL)
plt.plot(np.arange(len(lnLmax)), lnLmax,label='lnLmax')      # array of running-maximum lnL values
plt.plot(np.arange(len(lnLmarg)), lnLmarg,label='lnLmarg')    # array of marginalized L values
plt.xlabel('iteration')
plt.ylabel('lnL')
plt.legend()
plt.show()
plt.clf()
# Plot the sampled points (as example for direct access to 'ret')
print(" -- Sampled array of points  -- ")
xvals = np.transpose(ret)[0]
yvals = np.exp(np.transpose(ret)[1])  # lnL is last element of array
plt.scatter(xvals,yvals)
plt.xlabel('x')
plt.ylabel('L')
plt.show()

