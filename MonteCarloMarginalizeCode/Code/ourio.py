#
"""
ourio.py : 
  Input-output routines
   - plot sampling distributions, priors
   - plot (weighted) histograms of results
"""

import numpy as np
try:
    from matplotlib import pylab as plt   # if this fails, don't do it.  Will let us run on the cluster
except:
    print("No matplotlib for you!")
import lal   # rescale distance units if needed

"""
dumpSamplesToFile
Dump output in a format consistent with the bayesian MCMC plotting infrastructure
              This will make uptake and comparison much easier.
FORMAT:
    <key-line>
    <data-line>
where <key-line> must include (in any order)
    indx   logl  m1 m2 ra dec incl phiref psi d(Mpc) t
though in special cases m1, m2 may not be provided.

It is expected that the output will only contain a subset.
The interpretation of our result depends on how many binaries this sample.
For simplicity we will assume samples are distributed througout
"""
def dumpSamplesToFile(fname, samps, labels):
    headline = ' '.join(labels)
    np.savetxt(fname, samps,header=headline)
#    np.savetxt(fname, samps)  # stupid, but needed for cluster work.  No more!
    return 0


"""
dumpStatisticsToFile
Dump the maximum likelihood point and 1d width from a set of samples.
This will be used principally to *seed subsequent MC* with good choices, so we converge faster.
[Remember, we are free to choose a sampling distribution for each intrinsic evaluation.]
The routines using this process should take care not to overconverge.
"""
def dumpStatisticsToFile(fname,samps,labels):
    indxmax = np.argmax(samps,-1)
    pairs = [(labels[i], samps[indxmax][i]) for i in np.arange(len(labels))]
    np.savetxt(fname,pairs)
    return 0


def readStatisticsFromFile(fname,samps,labels):
    return 0



# Plotting routines
def saveParameterDistributions(fnameBase, sampler, samplerPrior):
    nFig = 0
    for param in sampler.params:
        nFig+=1
        plt.figure(nFig)
        plt.clf()
        xLow = sampler.llim[param]
        xHigh = sampler.rlim[param]
        xvals = np.linspace(xLow,xHigh,500)
        pdfPrior = samplerPrior.pdf[param]
        pdfvalsPrior = pdfPrior(xvals)/samplerPrior._pdf_norm[param]  # assume vectorized
        pdf = sampler.pdf[param]
        cdf = sampler.cdf[param]
        pdfvals = pdf(xvals)/sampler._pdf_norm[param]
        cdfvals = cdf(xvals)
        if str(param) ==  "dist":
            xvvals = xvals/(1e6*lal.LAL_PC_SI)       # plot in Mpc, not m.  Note PDF has to change
            pdfvalsPrior = pdfvalsPrior * (1e6*lal.LAL_PC_SI) # rescale units
            pdfvals = pdfvals * (1e6*lal.LAL_PC_SI) # rescale units
        plt.plot(xvals,pdfvalsPrior,label="prior:"+str(param),linestyle='--')
        plt.plot(xvals,pdfvals,label=str(param))
        plt.plot(xvals,cdfvals,label='cdf:'+str(param))
        plt.xlabel(str(param))
        plt.legend()
        plt.savefig(fnameBase+str(param)+".pdf")

def plotParameterDistributions(titleBase, sampler):
    nFig = 0
    for param in sampler.params:
        nFig+=1
        plt.figure(nFig)
        plt.clf()
        xLow = sampler.llim[param]
        xHigh = sampler.rlim[param]
        xvals = np.linspace(xLow,xHigh,500)
        pdfPrior = sampler.prior_pdf[param]
        pdfvalsPrior = pdfPrior(xvals)
        pdf = sampler.pdf[param]
        cdf = sampler.cdf[param]
        pdfvals = pdf(xvals)/sampler._pdf_norm[param]
        cdfvals = cdf(xvals)
        if str(param) ==  "dist":
            xvvals = xvals/(1e6*lal.LAL_PC_SI)       # plot in Mpc, not m.  Note PDF has to change
            pdfvalsPrior = pdfvalsPrior * (1e6*lal.LAL_PC_SI) # rescale units
            pdfvals = pdfvals * (1e6*lal.LAL_PC_SI) # rescale units
        plt.plot(xvals,pdfvalsPrior,label="prior:"+str(param),linestyle='--')
        plt.plot(xvals,pdfvals,label=str(param))
        plt.plot(xvals,cdfvals,label='cdf:'+str(param))
        plt.xlabel(str(param))
        plt.title(titleBase+":"+str(param))
        plt.legend()
    plt.show()

def plotParameterDistributionsFromSamples(titleBase, sampler, ret, keynames):
    nFig = 0
    lnL = np.transpose(ret)[-1] # Assumed.
    for nFig in range(len(keynames)-1):
        param = keynames[nFig]
        plt.figure(nFig)
        plt.clf()
        # Sample distributions
        vals = np.transpose(ret)[nFig]
        if str(param) == "dist":
            fac = (1e6*lal.LAL_PC_SI)
        else:
            fac = 1
        hist, bins = np.histogram(vals/fac, bins=50, density=True, weights=np.exp(lnL))
        center = (bins[:-1]+bins[1:])/2
        plt.plot(center,hist,label="posterior:"+param+":sampled")
        hist, bins  = np.histogram(vals/fac,bins=50,density=True)
        center = (bins[:-1]+bins[1:])/2
        plt.plot(center,hist,label="sampling prior:"+param+":sampled",linestyle='--')
        # Distributions
        xLow = sampler.llim[param]
        xHigh = sampler.rlim[param]
        xvals = np.linspace(xLow,xHigh,500)
        pdfPrior = sampler.prior_pdf[param]
        pdfvalsPrior = pdfPrior(xvals)
        pdf = sampler.pdf[param]
        pdfvals = pdf(xvals)/sampler._pdf_norm[param]
        xvvals = xvals/fac
        pdfvalsPrior = pdfvalsPrior *fac
        pdfvals = pdfvals *fac
        plt.plot(xvals,pdfvalsPrior,label="prior:"+str(param),linestyle='--')
        plt.plot(xvals,pdfvals,label="sampling prior:"+str(param))
        plt.xlabel(str(param))
        plt.title(titleBase+":"+str(param))
        plt.legend()
    plt.show()
