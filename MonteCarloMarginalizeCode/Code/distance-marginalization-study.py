#!/opt/local/bin/python

# This code was used to study the properties of the integral over distance
# in our parameter estimation code. The results of the study are documented
# in the notes overview.tex in the last section of the appendix. I have not
# assessed the speedup relative to a monte carlo integrator. I did use a
# a smart gridded itegrator which suggests that a lot of samples are needed
# get good accuracy. A comparison of the cost to get accurate results from 
# the integrator versus using the approximation gives huge speed ups ~40. But
# this must be taken with a grain of salt since a smart MC integrator may not
# need so many samples in distance to get the same accuracy. 

import numpy as np
import pylab as pl
from scipy.integrate import simps
from scipy.special import erf

pl.rc('text', usetex=True)

npoints=1024*8*8
sigmahat = np.array([0.0001,0.001,0.01,0.10,0.2,0.3,0.4,0.5])
rho = np.linspace(-5.0, 15.0, 1000)
rho = rho[:,np.newaxis]

for a in sigmahat:
  rhosigma = rho * a
  s2 = a * a
 
  ymax = 2*2.0 * np.max(rho) / a
  ymid = 0.25 * ymax # 0.5 / a
  
  #
  # Marginalize over distance numerically
  #
  y = np.linspace(1.0,ymid,npoints)
  dy = y[1]-y[0]
  y4 = y * y
  y4 = y4 * y4
  evidence = ( np.sum( np.exp( ( rhosigma - 0.5 * s2 * y ) * y ) / y4 , axis=1 ) * dy ) 
  y = np.linspace(ymid,ymax,npoints)
  dy = y[1]-y[0]
  y4 = y * y
  y4 = y4 * y4
  evidence = evidence + ( np.sum( np.exp( ( rhosigma - 0.5 * s2 * y ) * y ) / y4 , axis=1 ) * dy ) 

  #
  # Construct an approximation to the evidence
  #
  xplus = ( rho - 1.0 )
  xminus = ( rho - 10.0 )
  fplus = 0.5*(1.0 + np.tanh( xplus ))
  fminus = 0.5*(1.0 - np.tanh( xminus ) )

  # High snr approximation
  sgn = np.sign( rho - a )
  factor =  0.5 * (np.sign( rho - 2.0 ) + 1) * ( a * a * a ) * ( np.sqrt( np.pi / 2.0 ) * (1.0 + sgn * erf( sgn * (rho - a)/np.sqrt(2.0) ) ) / np.power(rho, 4.0) - 4.0 * np.exp( - 0.5 * a * a + a * rho - 0.5 * rho * rho ) / np.power(rho, 5.0) )
  evidence1 = factor * np.exp( ( ( 0.5 *rho * rho ) ) )
 
  # Low snr approximation
  evidence1 = evidence1 + 0.3333333333 * np.exp( ( - 0.5 * a * a + rho * a ) )

  #
  # Plot the evidence and the approximation
  #
  pl.figure(1)
  mylabel = "$\hat{\sigma} = %0.4f$" % a
  pl.semilogy(rho, evidence, '-', label=mylabel)
  pl.semilogy(rho, evidence1, 'o')

  # 
  # Plot the fractional error as a function of evidence
  #
  pl.figure(2)
  ratio = evidence/evidence1.ravel()
  pl.semilogx(evidence, np.abs(ratio), label=mylabel)

  print("This is a visual pacifier")

pl.figure(1)
pl.legend(loc='upper left')
pl.xlabel(r'$\rho$', fontsize=16)
pl.ylabel(r'$I/r_{\mathrm{max}^3}$', fontsize=16)
pl.grid(True)
pl.savefig("i-v-rho.png")
pl.xlim(-1.0,10.0)
pl.ylim(1.0e-1,1.0e10)
pl.savefig("i-v-rho-zoom.png")

pl.figure(2)
pl.legend(loc='upper right')
pl.xlabel(r'$I/r_{\mathrm{max}}^3$', fontsize=16)
pl.ylabel(r'$I/I_{\mathrm{approx}}$', fontsize=16)
pl.grid(True)
pl.savefig("i-error-v-i.png")
pl.xlim(1.0e-1,1.0e10)
pl.savefig("i-error-v-i-zoom.png")

