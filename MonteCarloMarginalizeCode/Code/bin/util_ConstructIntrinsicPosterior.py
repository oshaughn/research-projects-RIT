#!/usr/bin/env python
#
# GOAL
#   - load in lnL data
#   - fit peak to quadratic (standard), GP, etc. 
#   - pass as input to mcsampler, to generate posterior samples
#
# FORMAT
#   - pankow simplification of standard format
#
# COMPARE TO
#   util_NRQuadraticFit.py
#   postprocess_1d_cumulative
#   util_QuadraticMassPosterior.py
#

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import corner

import BayesianLeastSquares

import argparse
import sys
import numpy as np
import scipy
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools
import itertools

from matplotlib import pyplot as plt


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from glue.ligolw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)

import RIFT.integrators.mcsampler as mcsampler


# TeX dictionary
tex_dictionary  = {
 "mtot": '$M$',
 "mc": '${\cal M}_c$',
 "m1": '$m_1$',
 "m2": '$m_2$',
  "q": "$q$",
  "eta": "$\eta$",
  "chi_eff": "$\chi_{eff}$",
  "xi": "$\chi_{eff}$"

}


parser = argparse.ArgumentParser()
parser.add_argument("--fname",help="filename of *.dat file [standard ILE output]")
parser.add_argument("--fname-lalinference",help="filename of posterior_samples.dat file [standard LI output], to overlay on corner plots")
parser.add_argument("--fname-output-samples",default="output-ILE-samples",help="output posterior samples (default output-ILE-samples -> output-ILE)")
parser.add_argument("--fname-rom-samples",default=None,help="*.rom_composite output. Treated identically to set of posterior samples produced by mcsampler after constructing fit.")
parser.add_argument("--n-output-samples",default=1000,type=int,help="output posterior samples (default 1000)")
parser.add_argument("--desc-lalinference",type=str,default='',help="String to adjoin to legends for LI")
parser.add_argument("--desc-ILE",type=str,default='',help="String to adjoin to legends for ILE")
parser.add_argument("--parameter", action='append')
parser.add_argument("--use-precessing",action='store_true')
parser.add_argument("--lnL-offset",type=float,default=10,help="lnL offset")
parser.add_argument("--lnL-cut",type=float,default=None,help="lnL cut [MANUAL]")
parser.add_argument("--M-max-cut",type=float,default=1e5,help="Maximum mass to consider (e.g., if there is a cut on distance, this matters)")
parser.add_argument("--sigma-cut",type=float,default=0.6,help="Eliminate points with large error from the fit.")
parser.add_argument("--lnL-peak-insane-cut",type=float,default=np.inf,help="Throw away lnL greater than this value. Should not be necessary")
parser.add_argument("--fmin",type=float,default=None)
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
parser.add_argument("--inj-file", help="Name of injection file")
parser.add_argument("--event-num", type=int, default=0,help="Zero index of event in inj_file")
parser.add_argument("--coordinates-mc-eta", action='store_true')
parser.add_argument("--report-best-point",action='store_true')
parser.add_argument("--coordinates-M-q", action='store_true')
parser.add_argument("--coordinates-m1-m2", action='store_true')
parser.add_argument("--coordinates-chi1-chi2",action='store_true')
parser.add_argument("--adapt",action='store_true')
parser.add_argument("--fit-uses-reported-error",action='store_true')
parser.add_argument("--n-max",default=3e5,type=float)
parser.add_argument("--n-eff",default=3e3,type=int)
parser.add_argument("--fit-method",default="quadratic")
opts=  parser.parse_args()


###
### Comparison data (from LI)
###
if opts.fname_lalinference:
    print " Loading lalinference samples for direct comparison ", opts.fname_lalinference
    samples_LI = np.genfromtxt(opts.fname_lalinference,names=True)



test_converged={}
#test_converged['neff'] = functools.partial(mcsampler.convergence_test_MostSignificantPoint,0.01)  # most significant point less than 1/neff of probability.  Exactly equivalent to usual neff threshold.
#test_converged["normal_integral"] = functools.partial(mcsampler.convergence_test_NormalSubIntegrals, 25, 0.01, 0.1)   # 20 sub-integrals are gaussian distributed [weakly; mainly to rule out outliers] *and* relative error < 10%, based on sub-integrals . Should use # of intervals << neff target from above.  Note this sets our target error tolerance on  lnLmarg.  Note the specific test requires >= 20 sub-intervals, which demands *very many* samples (each subintegral needs to be converged).


###
### Prior functions : a dictionary
###

# mcmin, mcmax : to be defined later
def M_prior(x):
    return x/(mc_max-mc_min)
def q_prior(x):
    return x/(1+x)**2  # not normalized
def m1_prior(x):
    return 1./200
def m2_prior(x):
    return 1./200
def s1z_prior(x):
    return 1./2
def s2z_prior(x):
    return 1./2
def mc_prior(x):
    return x/(mc_max-mc_min)
def eta_prior(x):
    return 1./np.power(x,6./5.)/np.power(1-4.*x, 0.5)/1.44

prior_map  = { "mtot": M_prior, "q":q_prior, "s1z":s1z_prior, "s2z":s2z_prior, "mc":mc_prior, "eta":eta_prior}
prior_range_map = {"mtot": [1, 200], "q":[0.01,1], "s1z":[-0.99,0.99], "s2z":[-0.99,0.99], "mc":[0.9,90], "eta":[0.01,0.2499999]}


###
### Linear fits. Resampling a quadratic. (Export me)
###

def fit_quadratic_alt(x,y,x0=None,symmetry_list=None):
    the_quadratic_results = BayesianLeastSquares.fit_quadratic( x, y)#x0=None)#x0_val_here)
    peak_val_est, best_val_est, my_fisher_est, linear_term_est,fn_estimate = the_quadratic_results

    np.savetxt("lnL_peakval.dat",[peak_val_est])   # generally not very useful
    np.savetxt("lnL_bestpt.dat",best_val_est)  
    np.savetxt("lnL_gamma.dat",my_fisher_est,header=' '.join(coord_names))
        
    return fn_estimate

def fit_quadratic(x,y,x0=None,symmetry_list=None):
    """
    x = array so x[0] , x[1], x[2] are points.
    """

    poly = PolynomialFeatures(degree=2)
    X_  = poly.fit_transform(x)

    clf = linear_model.LinearRegression()
    clf.fit(X_,y)

    print  " Fit: std: ", np.std(y - clf.predict(X_)),  "using number of features ", len(y)  # should NOT be perfect
    
    return lambda x: clf.predict(poly.fit_transform(x))


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def fit_gp(x,y,x0=None,symmetry_list=None):
    """
    x = array so x[0] , x[1], x[2] are points.
    """

#    kernel = C([1.0,0.05],[ (1e-3, 1e2), (1e-3, 1)]) * RBF([1,0.05], [ (1e-3, 1e2), (1e-3, 1)])
    kernel = C(1, (1e-3,1e1))*RBF(1, (1e-3,1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    gp.fit(x,y)

    print  " Fit: std: ", np.std(y - gp.predict(x)),  "using number of features ", len(y)  # should NOT be perfect

    return lambda x: gp.predict(x)


coord_names = opts.parameter
fixed_coords = True
if opts.coordinates_mc_eta:
    coord_names = ['mc', 'eta', 'xi'] 
elif opts.coordinates_M_q:
    coord_names = ['mtot', 'q', 'xi'] 
elif opts.coordinates_m1_m2:
    coord_names = ['m1', 'm2', 'xi'] 
else:
    coord_names = opts.parameter
    fixed_coords = False
print " Coordinate names :, ", coord_names
print " Rendering coordinate names : ", map(lambda x: tex_dictionary[x], coord_names)

# initialize
dat_mass  = [] 
weights = []
n_params = -1
q_min = 0; mc_min =0; mc_max = 500;
eta_min = 0; eta_max = 0.25; xi_min = -1; xi_max = 1
if not opts.fname_rom_samples:

 ###
 ### Retrieve data
 ###
 #  id m1 m2  lnL sigma/L  neff
 col_lnL = 9
 dat_orig = dat = np.loadtxt(opts.fname)
 dat_orig = dat[dat[:,col_lnL].argsort()] # sort  http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
 print " Original data size = ", len(dat), dat.shape


 ###
 ### Convert data.  Use lalsimutils for flexibility
 ###
 P_list = []
 dat_out =[]
 

 symmetry_list =[]
 P= lalsimutils.ChooseWaveformParams()
 for line in dat:
  # Skip precessing binaries unless explicitly requested not to!
  if not opts.use_precessing and (line[3]**2 + line[4]**2 + line[6]**2 + line[7]**2)>0.01:
      continue
  if line[1]+line[2] > opts.M_max_cut:
      print " Skipping ", line, " as too massive, with mass ", line[1]+line[2]
      continue
  if line[10] > opts.sigma_cut:
      print " Skipping ", line
      continue
  if line[col_lnL] < opts.lnL_peak_insane_cut:
    P.m1 = line[1]*lal.MSUN_SI
    P.m2 = line[2]*lal.MSUN_SI
    P.s1x = line[3]
    P.s1y = line[4]
    P.s1z = line[5]
    P.s2x = line[6]
    P.s2y = line[7]
    P.s2z = line[8]
 #    print line,  P.extract_param('xi')
    line_out = np.zeros(len(coord_names)+1)
    for x in np.arange(len(coord_names)):
        line_out[x] = P.extract_param(coord_names[x])
 #        line_out[x] = getattr(P, coord_names[x])
    line_out[-1] = line[col_lnL]
    dat_out.append(line_out)
 dat_out = np.array(dat_out)
 # scale out mass units
 for p in ['mc', 'm1', 'm2', 'mtot']:
    if p in coord_names:
        indx = coord_names.index(p)
        dat_out[:,indx] /= lal.MSUN_SI


 # Repack data
 X =dat_out[:,0:len(coord_names)]
 Y = dat_out[:,-1]

 # Eliminate values with Y too small
 max_lnL = np.max(Y)
 indx_ok = Y>np.max(Y)-opts.lnL_offset
 print " Points used in fit : ", sum(indx_ok), " given max lnL ", max_lnL
 if max_lnL < 10:
    # nothing matters, we will reject it anyways
    indx_ok = np.ones(len(Y),dtype=bool)
 elif sum(indx_ok) < 10: # and max_lnL > 30:
    # mark the top 10 elements and use them for fits
    # this may be VERY VERY DANGEROUS if the peak is high and poorly sampled
    idx_sorted_index = np.lexsort((np.arange(len(Y)), Y))  # Sort the array of Y, recovering index values
    indx_list = np.array( [[k, Y[k]] for k in idx_sorted_index])     # pair up with the weights again
    indx_list = indx_list[::-1]  # reverse, so most significant are first
    indx_ok = map(int,indx_list[:10,0])
    print " Revised number of points for fit: ", sum(indx_ok), indx_ok, indx_list[:10]
 X_raw = X.copy()

 my_fit= None
 if opts.fit_method == "quadratic":
    X=X[indx_ok]
    Y=Y[indx_ok]
    if opts.report_best_point:
        my_fit = fit_quadratic_alt(X,Y)
        pt_best_X = np.loadtxt("lnL_bestpt.dat")
        for indx in np.arange(len(coord_names)):
            fac = 1
            if coord_names[indx] in ['mc','m1','m2','mtot']:
                fac = lal.MSUN_SI
            p_to_assign = coord_names[indx]
            if p_to_assign == 'xi':
                p_to_assign = "chieff_aligned"
            P.assign_param(p_to_assign,pt_best_X[indx]*fac) 
           
        print " ====BEST BINARY ===="
        print " Parameters from fit ", pt_best_X
        P.print_params()
        sys.exit(0)
    my_fit = fit_quadratic(X,Y)
 else:
    my_fit = fit_gp(X,Y)

 # Sort for later convenience (scatterplots, etc)
 indx = Y.argsort()#[::-1]
 X=X[indx]
 Y=Y[indx]


 # Make grid plots for all pairs of points, to facilitate direct validation of where posterior support lies
 import itertools
 for i, j in itertools.product( np.arange(len(coord_names)),np.arange(len(coord_names)) ):
  if i < j:
    plt.scatter( X[:,i],X[:,j],label='rapid_pe:'+opts.desc_ILE,c=Y); plt.legend(); plt.colorbar()
    plt.xlabel( tex_dictionary[coord_names[i]])
    plt.ylabel( tex_dictionary[coord_names[j]])
    plt.title("rapid_pe evaluations (=inputs); no fits")
    plt.savefig("scatter_"+coord_names[i]+"_"+coord_names[j]+".png"); plt.clf()



 ###
 ### Integrate posterior
 ###


 sampler = mcsampler.MCSampler()
 # Chris needs variable names to do things
 #mc_prior = np.vectorize(lambda x : x)  # normalized for our limits
 #eta_prior = np.vectorize(lambda x: 1./np.power(x,6./5.)/np.sqrt(1.-4*x)/1.44)  # Normalized!  Integrate[1/x^(6/5) /Sqrt[1 - 4 x],  {x, 0.19, 1/4}]

 # should CHANGE chirp mass range based on input data

 # this is M min or mc min, depending on the coordinate system
 mc_min = np.min(X_raw[:,0])
 mc_max = np.max(X_raw[:,0])
 q_min = np.min(dat[:,2]/dat[:,1])
 m2_min = np.min(dat[:,2])
 m2_max = np.max(dat[:,2])
 m1_min = np.min(dat[:,1])
 m1_max = np.max(dat[:,1])
 eta_min = lalsimutils.symRatio(1,0.3*q_min)  # 10:1 mass ratio possible
 xi_min = -0.99#np.min(X[:,2])
 #xi_max = 0.9 # np.max(X[:,2])
 #xi_min = np.min(X[:,2])
 xi_max = 0.99 # np.max(X[:,2])

 print " xi range ", xi_min, xi_max

 if opts.coordinates_mc_eta:
    print " Chirp mass integration range ", mc_min, mc_max
 else:
    print " Total mass integration range ", mc_min, mc_max

 if opts.coordinates_mc_eta:
    def mc_prior(x):
        return x/(mc_max-mc_min)
    def eta_prior(x):
        return 1./np.power(x,6./5.)/np.power(1-4.*x, 0.5)/1.44
    sampler.add_parameter('mc',pdf=np.vectorize(lambda x:1),prior_pdf=mc_prior,  left_limit=mc_min, right_limit=mc_max,adaptive_sampling=True)   # nominal real joint priot
    sampler.add_parameter('eta',pdf =np.vectorize(lambda x:1),prior_pdf=eta_prior,  left_limit=eta_min,right_limit=0.2499999,adaptive_sampling=True)  # tricky
    sampler.add_parameter('xi', \
                              pdf = np.vectorize(lambda x: 0.5), \
                              prior_pdf=lambda x: 0.5, \
                              left_limit=xi_min,
                          right_limit=xi_max,adaptive_sampling=True
                          )  # tricky

    # Chris requires functions have names that match the variables
    def likelihood_function(mc,eta,xi):  
        if isinstance(eta,float):
            return np.exp(my_fit([mc,eta,xi]))
        else:
            # print mc
            # print eta
            # print my_fit(np.array([mc,eta,xi]).T)
            return np.exp(my_fit(np.array([mc,eta,xi]).T))
    res, var, neff, dict_return = sampler.integrate(likelihood_function, 'mc', 'eta', 'xi', verbose=True,nmax=int(opts.n_max),n=1e5,save_intg=True,tempering_adapt=True, floor_level=1e-3,igrand_threshold_p=1e-3,convergence_tests=test_converged,adapt_weight_exponent=0.1)  # weight ecponent needs better choice
 elif opts.coordinates_M_q:
    # def M_prior(x):
    #     return x/(mc_max-mc_min)
    # def q_prior(x):
    #     return x/(1+x)**2  # not normalized
    def likelihood_function(mtot,q,xi):  
        if isinstance(q,float):
            return np.exp(my_fit([mtot,q,xi]))
        else:
            return np.exp(my_fit(np.array([mtot,q]).T))
    sampler.add_parameter('mtot',pdf=np.vectorize(lambda x:1),prior_pdf=M_prior,  left_limit=mc_min, right_limit=mc_max)   # nominal real joint priot
    sampler.add_parameter('q',pdf =np.vectorize(lambda x:1),prior_pdf=q_prior,  left_limit=q_min,right_limit=0.999999)  
    sampler.add_parameter('xi', \
                              pdf = np.vectorize(lambda x: 0.5), \
                              prior_pdf=lambda x: 0.5, \
                              left_limit=xi_min,
                          right_limit=xi_max
                          )  # tricky
    # Chris requires functions have names that match the variables
    res, var, neff, dict_return = sampler.integrate(likelihood_function, 'mtot', 'q', 'xi',verbose=True,nmax=int(opts.n_max),n=1e5,save_intg=True, tempering_adapt=True, floor_level=1e-3,igrand_threshold_p=1e-3,convergence_tests=test_converged)
 elif opts.coordinates_m1_m2:
    # def m1_prior(x):
    #     return 1./200
    # def m2_prior(x):
    #     return 1./200
    def likelihood_function(m1,m2,xi):  
        if isinstance(m2,float):
#            print my_fit([m1,m2,xi])
            return np.exp(my_fit([m1,m2,xi]))
        else:
            val = my_fit(np.array([m1,m2,xi]).T)
            return np.exp(my_fit(np.array([m1,m2,xi]).T))
    sampler.add_parameter('m1',pdf=np.vectorize(lambda x:1),prior_pdf=m1_prior,  left_limit=m1_min, right_limit=m1_max)   # nominal real joint priot
    sampler.add_parameter('m2',pdf =np.vectorize(lambda x:1),prior_pdf=m2_prior,  left_limit=m2_min,right_limit=m2_max)  
    sampler.add_parameter('xi', \
                              pdf = np.vectorize(lambda x: 0.5), \
                              prior_pdf=lambda x: 0.5, \
                              left_limit=xi_min,
                          right_limit=xi_max
                          )  # tricky
    # Chris requires functions have names that match the variables
    res, var, neff, dict_return = sampler.integrate(likelihood_function, 'm1', 'm2','xi', verbose=True,nmax=int(opts.n_max),n=1e5,save_intg=True)
 else:
     # http://stackoverflow.com/questions/1409295/set-function-signature-in-python
     for indx in np.arange(len(coord_names)):
         p = coord_names[indx]
         p_min = prior_range_map[p][0] #np.min(dat_out[:,indx])   # NOT safe
         p_max = prior_range_map[p][1] # np.max(dat_out[:,indx])
         sampler.add_parameter(p, pdf=np.vectorize(lambda x:1), prior_pdf=prior_map[p], left_limit=p_min,right_limit=p_max)
     argstr = ", ".join(coord_names)
     fakefunc_src = "def likelihood_function(%s):\n\t return np.exp(real_func([%s]).T)" % (argstr, argstr)
     print fakefunc_src
##     eval(fakefunc_src,{"real_func":my_fit,"np":np})
     fakefunc_code = compile(fakefunc_src, "fakesource", "exec")
     fakeglobals = {}
     eval(fakefunc_code, {"real_func":my_fit, "np":np},fakeglobals)
     likelihood_function = fakeglobals["likelihood_function"]
     # Test that likelihood function works
     print X[0], np.log(likelihood_function(X[0])), Y[0]
     print X[-1], np.log(likelihood_function(X[-1])), Y[-1]
     res, var, neff, dict_return = sampler.integrate(likelihood_function, *coord_names, verbose=True,nmax=int(opts.n_max),n=1e5,save_intg=True)


 ###
 ### Output setup (for non-ROM)
 ###


 samples = sampler._rvs
 print samples.keys()
 n_params = len(coord_names)
 dat_mass = np.zeros((len(samples[coord_names[0]]),n_params+3))
 dat_logL = np.log(samples["integrand"])
 print " Max lnL ", np.max(dat_logL)

 # Throw away stupid points that don't impact the posterior
 indx_ok = np.logical_and(dat_logL > np.max(dat_logL)-opts.lnL_offset ,samples["joint_s_prior"]>0)
 for p in coord_names:
    samples[p] = samples[p][indx_ok]
 dat_logL  = dat_logL[indx_ok]
 print samples.keys()
 samples["joint_prior"] =samples["joint_prior"][indx_ok]
 samples["joint_s_prior"] =samples["joint_s_prior"][indx_ok]
 for indx in np.arange(len(samples[coord_names[0]])):   # this is a stupid loop, but easy to debug
    if opts.coordinates_mc_eta:
        m1v,m2v = lalsimutils.m1m2(samples[coord_names[0]][indx], float(samples["eta"][indx]))
        dat_mass[indx][0] = np.max([m1v,m2v])
        dat_mass[indx][1] =np.min([m1v,m2v])
    elif opts.coordinates_m1_m2 or True:
        dat_mass[indx][0] = np.max([samples["m1"][indx],samples["m2"][indx]])
        dat_mass[indx][1] =np.min([samples["m1"][indx],samples["m2"][indx]])
    elif opts.coordinates_M_q:
        dat_mass[indx][0] = samples["mtot"][indx]/(1+samples['q'][indx])
        dat_mass[indx][1] =samples["mtot"][indx]*samples['q'][indx]/(1+samples['q'][indx])

    dat_mass[indx][2] = samples["xi"][indx]
    dat_mass[indx][n_params]=dat_logL[indx]
    dat_mass[indx][n_params+1]=samples["joint_prior"][indx]  # prior
    dat_mass[indx][n_params+2]=samples["joint_s_prior"][indx] # sampling prior
 dat_mass = dat_mass[dat_mass[:,4]>0]
 #dat_mass = np.array(lalsimutils.m1m2(samples["mc"],np.array(samples["eta"],dtype=np.float))).T
 np.savetxt("quadratic_fit_results.dat", dat_mass)   # FULL POSTERIOR SAMPLES, in specific coordinates

else:  # ROM ACTIVE
    tmp = np.loadtxt(opts.fname_rom_samples)
    coord_names = ['mc', 'eta','xi']
    n_params = len(coord_names)
    print tmp[1]
    dat_mass = np.zeros( (len(tmp), len(coord_names)+3),dtype=np.float)
    dat_mass[:,0] = tmp[:,0]
    dat_mass[:,1] = tmp[:,1]
#    dat_mass[:,0] = np.maximum(tmp[:,1],tmp[:,2])
#    dat_mass[:,1] = np.minimum(tmp[:,1],tmp[:,2])
    dat_mass[:,2] = 0.001*np.random.uniform(size=len(tmp[:,3])) #+ np.zeros(len(tmp[:,3]))  # no spin information provided at present
    dat_mass[:,n_params] = tmp[:,-1]   #lnL
    dat_mass[:,n_params+1] = tmp[:,-3] # ps
    dat_mass[:,n_params+2] = tmp[:,-2]*(tmp[:,0]+tmp[:,1]) # p...MULTIPLY BY M because I am working at fixed mass
    mc_min = np.min(lalsimutils.mchirp(dat_mass[:,0],dat_mass[:,1]))
    mc_max = np.max(lalsimutils.mchirp(dat_mass[:,0],dat_mass[:,1]))
    chivals = dat_mass[:,2]
    # dat_mass[2] = tmp[:,3] # s1x
    # dat_mass[3] = tmp[:,4] # s1y
    # dat_mass[4] = tmp[:,5] # s1z
    # dat_mass[5] = tmp[:,6] # s2x
    # dat_mass[6] = tmp[:,7] # s2y
    # dat_mass[7] = tmp[:,8] # s2z


###
### 1d posteriors in m1, m2
###

p = dat_mass[:,n_params+1]
ps =dat_mass[:,n_params+2]
lnL = dat_mass[:,n_params]
lnLmax = np.max(lnL)
weights = np.exp(lnL-lnLmax)*p/ps
print ps
# Expected range
#m1min,junk = lalsimutils.m1m2(mc_min,0.25)
#m1max,junk = lalsimutils.m1m2(mc_max,eta_min)
#print " Mass 1 range ", m1min, m1max


# Load in reference parameters
Pref = lalsimutils.ChooseWaveformParams()
if  opts.inj_file is not None:
    Pref = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj_file)[opts.event_num]
Pref.print_params()

# Quick and dirty cumulatives
print " --- cumulative 1 ---- "
if True:
    dat_out = []; dat_out_LI = []
    for x in np.linspace(np.min(dat_mass[:,0]),np.max(dat_mass[:,0]),200):
         dat_out.append([x, np.sum( weights[ dat_mass[:,0]< x])/np.sum(weights)])
         if opts.fname_lalinference:
             dat_out_LI.append([x, (1.0*np.sum( samples_LI["m1"]< x))/len(samples_LI) ])
    np.savetxt("m1_cdf.dat", np.array(dat_out))
    dat_out = np.array(dat_out); dat_out_LI=np.array(dat_out_LI)
    m1_val = Pref.m1/lal.MSUN_SI
    plt.plot(dat_out[:,0],dat_out[:,1],label="rapid_pe:"+opts.desc_ILE,color='b')
    if opts.fname_lalinference:
        plt.plot(dat_out_LI[:,0],dat_out_LI[:,1],label="LI:"+opts.desc_lalinference,color='r')
    plt.axvline(m1_val,color='k',linestyle='--')
    plt.xlabel('$m_1 (M_\odot)$'); plt.legend()
    plt.savefig("m1_cdf.png"); plt.clf()

print " --- cumulative 2 --- "
if True:
    dat_out = []; dat_out_LI = []
    for x in np.linspace(np.min(dat_mass[:,1]),np.max(dat_mass[:,1]),200):
        dat_out.append([x, np.sum( weights[ dat_mass[:,1]< x])/np.sum(weights)])
        if opts.fname_lalinference:
             dat_out_LI.append([x, (1.0*np.sum( samples_LI["m2"]< x))/len(samples_LI) ])
    np.savetxt("m2_cdf.dat", np.array(dat_out))
    dat_out = np.array(dat_out); dat_out_LI=np.array(dat_out_LI)
    plt.plot(dat_out[:,0],dat_out[:,1],label='rapid_pe:'+opts.desc_ILE, color='b')
    if opts.fname_lalinference:
        plt.plot(dat_out_LI[:,0],dat_out_LI[:,1],label="LI:"+opts.desc_lalinference,color='r')
    m2_val = Pref.m2/lal.MSUN_SI
    plt.axvline(m2_val,color='k',linestyle='--')
    plt.xlabel('$m_2 (M_\odot)$'); plt.legend()
    plt.savefig("m2_cdf.png"); plt.clf()


print " --- cumulative 3 --- "
if True:
    dat_out = []; dat_out_LI = []
    for x in np.linspace(np.min(dat_mass[:,0]+dat_mass[:,1]),np.max(dat_mass[:,0]+dat_mass[:,1]),50):
        dat_out.append([x, np.sum( weights[ dat_mass[:,0]+dat_mass[:,1]< x])/np.sum(weights)])
        if opts.fname_lalinference:
             dat_out_LI.append([x, (1.0*np.sum( samples_LI["mtotal"]< x))/len(samples_LI) ])
    np.savetxt("mtot_cdf.dat", np.array(dat_out))
    dat_out = np.array(dat_out); dat_out_LI=np.array(dat_out_LI)
    x = (Pref.m1 + Pref.m2)/lal.MSUN_SI#Pref.extract_param('mtot')
    plt.axvline(x,color='k',linestyle='--')
    plt.plot(dat_out[:,0],dat_out[:,1],label='rapid_pe:'+opts.desc_ILE, color='b')
    if opts.fname_lalinference:
        plt.plot(dat_out_LI[:,0],dat_out_LI[:,1],label='LI:'+opts.desc_lalinference,color='r')
    plt.xlabel('$M (M_\odot)$'); plt.legend()
    plt.savefig("mtot_cdf.png"); plt.clf()

print " --- cumulative 4 --- "
if True:
    dat_out = []; dat_out_LI = []
    for x in np.linspace(q_min,np.max(dat_mass[:,1]/dat_mass[:,0]),50):
        dat_out.append([x, np.sum( weights[ dat_mass[:,1]/dat_mass[:,0] < x])/np.sum(weights)])
        if opts.fname_lalinference:
             dat_out_LI.append([x, (1.0*np.sum( samples_LI["q"]< x))/len(samples_LI) ])
    np.savetxt("q_cdf.dat", np.array(dat_out))
    dat_out = np.array(dat_out); dat_out_LI=np.array(dat_out_LI)
    x = Pref.m2/Pref.m1
    plt.axvline(x,color='k',linestyle='--')
    plt.plot(dat_out[:,0],dat_out[:,1],label='rapid_pe:'+opts.desc_ILE,color='b')
    if opts.fname_lalinference:
        plt.plot(dat_out_LI[:,0],dat_out_LI[:,1],label='LI:'+opts.desc_lalinference,color='r')
    plt.xlim(q_min,1)
    plt.xlabel('$q$'); plt.legend()
    plt.savefig("q_cdf.png"); plt.clf()


print " --- cumulative 5 --- "
if True:
    dat_out = []; dat_out_LI =[]
    for x in np.linspace(xi_min,np.max(dat_mass[:,2]),50):
        dat_out.append([x, np.sum( weights[ dat_mass[:,2] < x])/np.sum(weights)])
        if opts.fname_lalinference:
             dat_out_LI.append([x, (1.0*np.sum( samples_LI["chi_eff"]< x))/len(samples_LI) ])
    np.savetxt("xi_cdf.dat", np.array(dat_out))
    dat_out = np.array(dat_out); dat_out_LI=np.array(dat_out_LI)
    x = Pref.m2/Pref.m1
    plt.axvline(x,color='k',linestyle='--')
    plt.plot(dat_out[:,0],dat_out[:,1],color='b',label='rapid_pe:'+opts.desc_ILE)
    if opts.fname_lalinference:
        plt.plot(dat_out_LI[:,0],dat_out_LI[:,1],color='r',label='LI:'+opts.desc_lalinference)
    plt.xlabel('$\\xi$'); plt.legend()
    plt.savefig("xi_cdf.png"); plt.clf()


print " --- s1z, s2z scatter (input only) --- "
# http://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python
lnLmax_all = np.max(dat[:,col_lnL])  # peak max
if lnLmax_all > opts.lnL_peak_insane_cut:
    lnLmax_all = lnLmax
dat_ok = dat[ np.logical_and(dat[:,col_lnL] < opts.lnL_peak_insane_cut ,dat[:,col_lnL] > lnLmax_all -8)]
print " Remaining points ", len(dat_ok)
if len(dat_ok) >0:
    dat_ok = dat_ok[ dat_ok[:,col_lnL].argsort()]
    print np.max(dat_ok[:,col_lnL]), lnLmax, lnLmax-4, opts.lnL_peak_insane_cut # peak lnL is much higher for precessing
    print dat_ok[0],dat_ok[-1]
    plt.clf();
    dat_1p = -np.sqrt(dat_ok[:,3]**2 +dat_ok[:,4]**2 )
    dat_2p = np.sqrt(dat_ok[:,6]**2 +dat_ok[:,7]**2 )
    plt.scatter(dat_1p, dat_ok[:,5], c=dat_ok[:,col_lnL])
    plt.scatter(dat_2p, dat_ok[:,8], c=dat_ok[:,col_lnL])
    plt.xlabel('$\chi_{1,\perp},\chi_{2,\perp}$')
    plt.ylabel('$\chi_{1,z},\chi_{2,z}$')
    plt.colorbar()
    plt.title("rapid_pe evaluations (=inputs); no fits")
    plt.savefig("spin_disk_input.png")
    plt.clf()
    plt.scatter(dat_ok[:,5],dat_ok[:,8], c=dat_ok[:,col_lnL])
    plt.title("rapid_pe evaluations (=inputs); no fits")
    plt.xlabel("$\chi_{1,z}$")
    plt.ylabel("$\chi_{2,z}$")
    plt.colorbar()
    plt.savefig('chi1z_chi2z_input.png')


print " --- corner --- "




CIs = [0.99,0.95,0.9,0.68]
quantiles_1d = [0.05,0.95]
labels_raw = ['m1','m2','xi']
labels_tex = map(lambda x: tex_dictionary[x], labels_raw)
range_here = [(np.min(dat_mass[:,k]),np.max(dat_mass[:,k])) for k in [0,1,2] ]  # plot range set by surviving grid points. CHEAT -- I know coordinates
if range_here[2][0] ==range_here[2][1]:
    rnage_here[2][0]=-1
    range_here[2][1]=1
fig_base = corner.corner(dat_mass[:,:len(coord_names)], weights=weights/np.sum(weights),labels=labels_tex, quantiles=quantiles_1d,plot_datapoints=False,plot_density=False,no_fill_contours=True,fill_contours=False,levels=CIs,range=range_here)
if not opts.fname_rom_samples:
    # Overlay grid points with high support
    dat = dat_orig
    dat_here = np.zeros( (len(dat),3))
    indx_ok = dat[:,col_lnL] > lnLmax - 8
    dat_here[:,0] = dat[:,1]
    dat_here[:,1] = dat[:,2]
    dat_here[:,2] = (dat[:,1]*dat[:,5] + dat[:,2]*dat[:,8])/(dat[:,1]+ dat[:,2])
    dat_here = dat_here[indx_ok]
    print range_here
    print " Plotting overlay: ILE evaluations near the peak, with npts= ", len(dat_here)
    fig_base = corner.corner(dat_here,plot_datapoints=True,plot_density=False,plot_contours=False,quantiles=None,fig=fig_base,weights = 1*np.ones(len(dat_here))/len(dat_here), data_kwargs={'color':'g'},hist_kwargs={'color':'g', 'linestyle':'--'},labels=['m1','m2','xi'])
if opts.fname_lalinference:
    fig_base = corner.corner( np.array([samples_LI["m1"],samples_LI["m2"],samples_LI["chi_eff"]]).T,color='r',labels=labels_tex,weights=np.ones(len(samples_LI))*1.0/len(samples_LI),quantiles=quantiles_1d,fig=fig_base,plot_datapoints=False,no_fill_contours=True,fill_contours=False,plot_density=False,levels=CIs,range=range_here)
plt.savefig("posterior_corner_m1m2.png"); plt.clf()

print " --- corner 2 --- "
mtot_vals = dat_mass[:,0]+dat_mass[:,1]
q_vals = dat_mass[:,1]/dat_mass[:,0]
xi_vals = dat_mass[:,2]
labels_raw = ['mtot','q','xi']
labels_tex = map(lambda x: tex_dictionary[x], labels_raw)
fig_base=corner.corner(np.array([mtot_vals, q_vals , xi_vals]).T, weights=weights/np.sum(weights),labels=labels_tex,quantiles=quantiles_1d,plot_datapoints=False,plot_density=False,no_fill_contours=True,fill_contours=False,levels=CIs)
range_here = [
        (np.min(dat_mass[:,0]+dat_mass[:,1]),np.max(dat_mass[:,0]+dat_mass[:,1])),
        (np.min(dat_mass[:,1]/dat_mass[:,0]),np.max(dat_mass[:,1]/dat_mass[:,0])),
        (-1,1)  ]  # plot range set by surviving grid points
if not opts.fname_rom_samples:
    # Overlay grid points with high support
    dat = dat_orig
    dat_here = np.zeros( (len(dat),3))
    indx_ok = dat[:,col_lnL] > lnLmax - 8
    dat_here[:,0] = dat[:,1]+dat[:,2]
    dat_here[:,1] = dat[:,2]/dat[:,1]
    dat_here[:,2] = (dat[:,1]*dat[:,5] + dat[:,2]*dat[:,8])/(dat[:,1]+ dat[:,2])
    dat_here = dat_here[indx_ok]
    print " Plotting overlay for points ", len(dat_here), " with range ", range_here
    chivals =  (dat[:,1]*dat[:,5] + dat[:,2]*dat[:,8])/(dat[:,1]+ dat[:,2])
    range_here = [
        (np.min(dat_mass[:,0]+dat_mass[:,1]),np.max(dat_mass[:,0]+dat_mass[:,1])),
        (np.min(dat_mass[:,1]/dat_mass[:,0]),np.max(dat_mass[:,1]/dat_mass[:,0])),
        (np.min(chivals),np.max(chivals))  ]  # plot range set by surviving grid points
    fig_base = corner.corner(dat_here,plot_datapoints=True,plot_density=False,plot_contours=False,fig=fig_base,data_kwargs={'color':'g'},hist_kwargs={'color':'g', 'linestyle':'--'},range=range_here)

if opts.fname_lalinference:
    corner.corner( np.array([samples_LI["mtotal"],samples_LI["q"],samples_LI["chi_eff"]]).T,color='r',labels=labels_tex,weights=np.ones(len(samples_LI))*1.0/len(samples_LI),fig=fig_base,quantiles=quantiles_1d,no_fill_contours=True,plot_datapoints=False,plot_density=False,fill_contours=False,levels=CIs,range=range_here)
plt.savefig("posterior_corner_Mqxi.png")


###
### Posterior samples
###

if opts.fname_rom_samples:
    import sys
    sys.exit(0)

print " --- Exporting " + str(opts.n_output_samples) + " posterior samples to " + opts.fname_output_samples  + " ---- "

# pick random numbers
p_thresholds =  np.random.uniform(low=0.0,high=1.0,size=opts.n_output_samples)
# find sample indexes associated with the random numbers
#    - FIXME: first truncate the bad ones
# idx_sorted_index = numpy.lexsort((numpy.arange(len(weights)), weights))  # Sort the array of weights, recovering index values
# indx_list = numpy.array( [[k, weights[k]] for k in idx_sorted_index])     # pair up with the weights again
# cum_weights = np.cumsum(indx_list[:,1)
# cum_weights = cum_weights/cum_weights[-1]
# indx_list = [indx_list[k, 0] for k, value in enumerate(cum_sum > deltaP) if value]  # find the indices that preserve > 1e-7 of total probabilit
cum_sum  = np.cumsum(weights)
cum_sum = cum_sum/cum_sum[-1]
indx_list = map(lambda x : np.sum(cum_sum < x),  p_thresholds)  # this can lead to duplicates
P_list =[]
P = lalsimutils.ChooseWaveformParams()
P.approx = lalsim.SEOBNRv2  # DEFAULT
P.fmin = 20 # DEFAULT
for indx_here in indx_list:
        line = [samples[p][indx_here] for p in coord_names]
        Pgrid = P.manual_copy()
        include_item =True
        # Set attributes that are being changed as necessary, leaving all others fixed
        for indx in np.arange(len(coord_names)):
#            if param_names[indx] in downselect_dict:
#                if line[indx] < downselect_dict[param_names[indx]][0] or line[indx] > downselect_dict[param_names[indx]][1]:
#                    include_item = False
#                    if opts.verbose:
#                        print " Skipping " , line
            # if parameter involes a mass parameter, scale it to sensible units
            fac = 1
            if coord_names[indx] in ['mc', 'mtot', 'm1', 'm2']:
                fac = lal.MSUN_SI
            # do assignment of parameters anyways, as we will skip it momentarily
            coord_to_assign = coord_names[indx]
            if coord_to_assign == 'xi':
                coord_to_assign= 'chieff_aligned'
            Pgrid.assign_param(coord_to_assign, line[indx]*fac)
            
        if opts.verbose:
            Pgrid.print_params()
        # Downselect.
        # for param in downselect_dict:
        #     if Pgrid.extract_param(param) < downselect_dict[param][0] or Pgrid.extract_param(param) > downselect_dict[param][1]:
        #         print " Skipping " , line
        #         include_item =False
        if include_item:
         if Pgrid.m2 <= Pgrid.m1:  # do not add grid elements with m2> m1, to avoid possible code pathologies !
            P_list.append(Pgrid)
         else:
            Pgrid.swap_components()  # IMPORTANT.  This should NOT change the physical functionality FOR THE PURPOSES OF OVERLAP (but will for PE - beware phiref, etc!)
            P_list.append(Pgrid)
        else:
            True

print " Writing file ", opts.fname_output_samples, " size ", len(P_list)
lalsimutils.ChooseWaveformParams_array_to_xml(P_list, fname=opts.fname_output_samples, fref=P.fref)


# loop over
