#! /usr/bin/env python
#
# GOAL
#   - read in parameter XML
#   - assess parameter covariance (in some named parameters)
#   - generate random numbers, based on that covariance, to add to those parameters, 'puffing up' the distribution
#
# USAGE
#   - tries to follow util_CIP_GC
#
# EXAMPLES
#    util_ManualOverlapGrid.py  --skip-overlap --parameter mc --parameter-range [1,2] --parameter eta --parameter-range [0.1,0.2] --parameter s1z --parameter-range [-1,1] --parameter s2z --parameter-range [-1,1] 
#    python util_ParameterPuffball.py  --parameter mc --parameter eta --no-correlation "['mc','eta']" --parameter s1z --parameter s2z --inj-file ./overlap-grid.xml.gz  --no-correlation "['mc','s1z']"
#   python util_ParameterPuffball.py  --parameter mc --parameter eta --parameter s1z --parameter s2z --inj-file ./overlap-grid.xml.gz   --force-away 0.4

# PROBLEMS
#    - if points are too dense (i.e, if the output size gets too large) then we will reject everything, even for uniform placement.  
#    - current implementation produces pairwise distance matrix, so can be memory-hungry for many points


import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import functools
import itertools


from igwn_ligolw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)


parser = argparse.ArgumentParser()
parser.add_argument("--inj-file", help="Name of XML file")
parser.add_argument("--inj-file-out", default="output-puffball", help="Name of XML file")
parser.add_argument("--puff-factor", default=1,type=float)
parser.add_argument("--fail-if-empty", action='store_true', help="Fail if the output file is empty. Useflu diagnostic to stop runs with undesired behavior which otherwise quietly have no puff input")
parser.add_argument("--force-away", default=0,type=float,help="If >0, uses the icov to compute a metric, and discards points which are close to existing points")
parser.add_argument("--approx-output",default="SEOBNRv2", help="approximant to use when writing output XML files.")
parser.add_argument("--fref",default=None,type=float, help="Reference frequency used for spins in the ILE output.  (Since I usually use SEOBNRv3, the best choice is 20Hz). Default is to use what is in the original overlap-grid.xml.gz file")
parser.add_argument("--fmin",type=float,default=None,help="Min frequency, default is to use what is in original file")
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--no-correlation", type=str,action='append', help="Pairs of parameters, in format [mc,eta]  The corresponding term in the covariance matrix is eliminated")
#parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
parser.add_argument("--random-parameter", action='append',help="These parameters are specified at random over the entire range, uncorrelated with the grid used for other parameters.  Use for variables which correlate weakly with others; helps with random exploration")
parser.add_argument("--random-parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used.  ")
parser.add_argument("--mc-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--eta-range",default=None,help="Eta range. Important if we have a BNS or other item that has a strong constraint.")
parser.add_argument("--mtot-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points. Note mc-range, eta-range,mtot-range are syntactic sugar for this usage ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--enforce-duration-bound",default=None,type=float,help="If present, enforce a duration bound. Used to prevent grid placement for obscenely long signals, when the window size is prescribed")
parser.add_argument("--regularize",action='store_true',help="Add some ad-hoc terms based on priors, to help with nearly-singular matricies")
opts=  parser.parse_args()

if opts.random_parameter is None:
    opts.random_parameter = []

# these parameters are POSITIVE-DEFINITE, so we should perform puff in their natural log to prevent negative values
log_coord_names = ['lambda1', 'lambda2', 'LambdaTilde']
    
# Extract parameter names
coord_names = opts.parameter # Used  in fit
#if opts.parameter_nofit:
#    coord_names = coord_names + opts.parameter_nofit
if coord_names is None:
    sys.exit(0)

# match up pairs in --no-correlation
corr_list = None
if not(opts.no_correlation is None):
    corr_list = []
    corr_name_list = list(map(eval,opts.no_correlation))
#    print opts.no_correlation, corr_name_list
    for my_pair in corr_name_list:
        
        i1 = coord_names.index(my_pair[0])
        i2 = coord_names.index(my_pair[1])

        if i1>-1 and i2 > -1:
            corr_list.append([i1,i2])
#        else:
#            print i1, i2
#    print opts.no_correlation, coord_names, corr_list

downselect_dict = {}

# Add some pre-built downselects, to avoid common out-of-range-error problems
# Don't add full spin constraint unless called for. If so, we need to retain transverse values!  That will be done AT END
#downselect_dict['chi1'] = [0,1]
#downselect_dict['chi2'] = [0,1]
downselect_dict['eta'] = [0,0.25]
if opts.eta_range:
    downselect_dict['eta'] = eval(opts.eta_range)
downselect_dict['m1'] = [0,1e10]
downselect_dict['m2'] = [0,1e10]
if opts.mc_range:
    downselect_dict['mc'] = eval(opts.mc_range)
if opts.mtot_range:
    downselect_dict['mtot'] = eval(opts.mtot_range)

if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = list(map(eval,opts.downselect_parameter_range))
else:
    dlist = []
    dlist_ranges = []
    opts.downselect_parameter =[]
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]




# Load data
P_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj_file)

# extract parameters to measure the coordinates. 
dat_out = []
for P in P_list:
    # Force override of fmin, fref ... not always correctly populated. DANGEROUS, relies on helper to pass correct arguments
    if not(opts.fmin is None):
        P.fmin = opts.fmin
    if not(opts.fref is None):
        P.fref = opts.fref
    line_out = np.zeros(len(coord_names))
    for x in np.arange(len(coord_names)):
        fac=1
        if coord_names[x] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        line_out[x] = P.extract_param(coord_names[x])/fac
        
    dat_out.append(line_out)

# Scale out physical mass

# relabel data
dat_out = np.array(dat_out)
X =dat_out[:,0:len(coord_names)]

# Perform log transformation on variables
for indx, name  in enumerate(coord_names):
    if name in log_coord_names:
        print(" log transform for ", name)
        X[:,indx] = np.log(X[:,indx])

# Measure covariance matrix and generate random errors
if len(coord_names) >1:
    cov_in = np.cov(X.T)
    cov = cov_in*opts.puff_factor*opts.puff_factor

    # Check for singularities
    if np.min(np.linalg.eig(cov)[0])<1e-10:
        print(" ===> WARNING: SINGULAR MATRIX: are you sure you varied this parameters? <=== ")
        icov_pseudo = np.linalg.pinv(cov)
        # Prior range for each parameter is 1000, so icov diag terms are 10^(-6)
        # This is somewhat made up, but covers most things
        diag_terms = 1e-6*np.ones(len(cov))
        # 
        icov_proposed = icov_pseudo+np.diag(diag_terms)
        cov= np.linalg.inv(icov_proposed)

    cov_orig = np.array(cov)  # force copy
    # Remove targeted covariances
    if not(corr_list is None):
      for my_pair in corr_list:
        if my_pair[0] != my_pair[1]:
            cov[my_pair[0],my_pair[1]]=0
            cov[my_pair[1],my_pair[0]]=0
            

    # Compute errors
    rv = scipy.stats.multivariate_normal(mean=np.zeros(len(coord_names)), cov=cov,allow_singular=True)  # they are just complaining about dynamic range of parameters, usually
    delta_X = rv.rvs(size=len(X))
    X_out = X+delta_X
    if 'eta' in coord_names:
        indx_eta = coord_names.index('eta')
        X_out[:,indx_eta] = np.where(X_out[:,indx_eta] > 1/4, 1/2- X_out[:,indx_eta], X_out[:,indx_eta]) # reflection boundary condition, preserve points
        X_out[:,indx_eta] = np.where(X_out[:,indx_eta] < 0, -X_out[:,indx_eta], X_out[:,indx_eta]) # reflection on other side
else:
    sigma = np.std(X)
    cov = sigma*sigma
    delta_X =np.random.normal(size=len(coord_names), scale=sigma)
    X_out = X+delta_X
    if 'eta' in coord_names:
        indx_eta = coord_names.index('eta')
        X_out[:,indx_eta] = np.where(X_out[:,indx_eta] > 1/4, 1/2- X_out[:,indx_eta], X_out[:,indx_eta]) # reflection boundary condition, preserve points
        X_out[:,indx_eta] = np.where(X_out[:,indx_eta] < 0, -X_out[:,indx_eta], X_out[:,indx_eta]) # reflection on other side

# Undo natural logarithm
for indx, name  in enumerate(coord_names):
    if name in log_coord_names:
        print(" undoing log transform for ", name)
        X_out[:,indx] = np.exp(X_out[:,indx])


# Sanity check parameters
#for indx in np.arange(len(coord_names)):
    # if coord_names[indx] == 'eta':
    #     X_out[:,indx] = np.minimum(X_out[:,indx], 0.25)
    #     X_out[:,indx] = np.maximum(X_out[:,indx], opts.internal_eta_min)  # overrides downselect_dict
    # if coord_names[indx] == 's1z' or coord_names[indx]=='s2z':
    #     X_out[:,indx] = np.minimum(X_out[:,indx], 0.99)
    #     X_out[:,indx] = np.maximum(X_out[:,indx], -0.99)

# Apply downselect constraints, using conversion utility for speed
#   WARNING: For spin, we are NOT retaining transverse DOF by default!
names_downselect = list(downselect_dict.keys())
x_out_down = lalsimutils.convert_waveform_coordinates(X_out, coord_names=names_downselect, low_level_coord_names=coord_names)
indx_ok = np.ones(len(x_out_down),dtype=bool)
for indx, name in enumerate(names_downselect):
    indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isnan(x_out_down[:,indx])))
    indx_ok = np.logical_and(indx_ok,  x_out_down[:,indx]<= downselect_dict[name][1] )
    indx_ok = np.logical_and(indx_ok,  x_out_down[:,indx]>= downselect_dict[name][0] )
    print('   Increment downselect : {} {} '.format(name, np.sum(indx_ok) ))
print(" Range downselect : ", np.sum(indx_ok), len(indx_ok))
X_out = X_out[indx_ok]
P_list = list(itertools.compress(P_list, indx_ok))  # https://stackoverflow.com/questions/18665873/filtering-a-list-based-on-a-list-of-booleans

# Discard points which are 'close' to the original data set
#   - there are MUCH faster codes eg in scipy which should do this
if opts.force_away > 0:
    icov= np.linalg.pinv(cov_orig)
    Y = np.min(scipy.spatial.distance.cdist(X,X_out, metric='mahalanobis',VI=icov),axis=0)
    Y_id = Y > opts.force_away
    print(" Puffball distance rejection size change " , np.sum(Y_id), len(Y_id))
    X_out = X_out[Y_id]
    P_list = list(itertools.compress(P_list, Y_id))  # https://stackoverflow.com/questions/18665873/filtering-a-list-based-on-a-list-of-booleans


    # X_out_shorter = []
    # P_list_shorter = []
    # for indx in np.arange(len(X_out)):
    #     if test_index_distance(indx): #test_point_distance(X_out[indx]):
    #         X_out_shorter.append(X_out[indx])
    #         P_list_shorter.append(P_list[indx])
    # X_out_shorter=np.array(X_out_shorter)
    # print(" Puffball distance rejection size change " , len(X_out), len(X_out_shorter))
    # X_out = X_out_shorter
    # P_list = P_list_shorter



#print X_out
cov_out = np.cov(X_out.T)
print(" Covariance change: The following two matrices should be (A) and (1+puff^2)A, where puff= ", opts.puff_factor)
print(cov)
print(cov_out)
if len(coord_names)>1:
    print(" The one dimensional widths are ", np.sqrt(np.diag(cov_out)))
else:
    print(" The one dimensional width is", np.sqrt(cov_out))

print(" The number of raw points is ", len(P_list))



# Copy parameters back in.  MAKE SURE THIS IS POSSIBLE
P_out = []
for indx_P in np.arange(len(P_list)):
    include_item=True
    P = P_list[indx_P]
    for indx in np.arange(len(coord_names)):
        fac=1
        # sanity check restrictions, which may cause problems with the coordinate converters
        if coord_names[indx] == 'eta' and (X_out[indx_P,indx]>0.25 or X_out[indx_P,indx]<1e-5) :
            continue
        if coord_names[indx] == 'delta_mc' and (X_out[indx_P,indx]>1 or X_out[indx_P,indx]<0.) :
            continue
        if coord_names[indx] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        if coord_names[indx] in lalsimutils.periodic_params:
            X_out[indx_P] = np.mod(X_out[indx_P], lalsimutils.periodic_params[coord_names[indx]])
        P_list[indx_P].assign_param( coord_names[indx], X_out[indx_P,indx]*fac)

    if np.isnan(P.m1) or np.isnan(P.m2):  # don't allow nan mass
        continue

    if not(opts.enforce_duration_bound is None):
      if lalsimutils.estimateWaveformDuration(P)> opts.enforce_duration_bound:
        include_item = False
    # Apply unit spin magnitude constraint, which can ONLY be applied with the P_list construction, which has all spin information
    if P_list[indx_P].extract_param('chi1')>1 or P_list[indx_P].extract_param('chi2')>1:
        include_item =False
    # for param in downselect_dict:
    #     val = P.extract_param(param)
    #     if np.isnan(val):
    #         include_item=False   # includes check on m1,m2
    #         continue # stop trying to calculate with this parameter
    #     if param in ['mc','m1','m2','mtot']:
    #         val = val/ lal.MSUN_SI
    #     if val < downselect_dict[param][0] or val > downselect_dict[param][1]:
    #         include_item =False
    if include_item:
        P_out.append(P)




# Randomize parameters that have been requested to be randomized
#   - note there is NO SANITY CHECKING if you do this
#   - target: tidal parameters, more efficiently hammer on low-tide corner if necessary
if len(opts.random_parameter) >0:
  random_ranges = {}   
  for indx in np.arange(len(opts.random_parameter)):
    param = opts.random_parameter[indx]
    random_ranges[param] = np.array(eval(opts.random_parameter_range[indx]))
  for P in P_out: 
    for param in opts.random_parameter:
        val = np.random.uniform( random_ranges[param][0], random_ranges[param][1])
        if param in ['mc','m1','m2','mtot']:
            val = val* lal.MSUN_SI
        P.assign_param(param,val)

print(" The number of exported points is ", len(P_out))

if opts.fail_if_empty and len(P_out)<1:
    raise Exception(" Puff file will be empty ! Fail  without output ! You probably have settings which lead to either (a) a singular puff matrix (eg., duplicated coordinates or unused variables) or (b) your puff is far too large")

# Export
lalsimutils.ChooseWaveformParams_array_to_xml(P_out,fname=opts.inj_file_out,fref=P.fref)

    

