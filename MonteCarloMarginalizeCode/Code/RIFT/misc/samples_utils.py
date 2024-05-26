
import numpy as np
import RIFT.lalsimutils as lalsimutils
remap_ILE_2_LI = {
 "s1z":"a1z", "s2z":"a2z", 
 "s1x":"a1x", "s1y":"a1y",
 "s2x":"a2x", "s2y":"a2y",
 "chi1_perp":"chi1_perp",
 "chi2_perp":"chi2_perp",
 "chi1":'a1',
 "chi2":'a2',
 "cos_phiJL": 'cos_phiJL',
 "sin_phiJL": 'sin_phiJL',
 "cos_theta1":'costilt1',
 "cos_theta2":'costilt2',
 "theta1":"tilt1",
 "theta2":"tilt2",
  "xi":"chi_eff", 
  "chiMinus":"chi_minus", 
  "delta":"delta", 
  "delta_mc":"delta", 
 "mtot":'mtotal', "mc":"mc", "eta":"eta","m1":"m1","m2":"m2",
  "cos_beta":"cosbeta",
  "beta":"beta",
  "LambdaTilde":"lambdat",
  "DeltaLambdaTilde": "dlambdat",
  "thetaJN":"theta_jn"}
remap_LI_to_ILE = { "a1z":"s1z", "a2z":"s2z", "chi_eff":"xi", "lambdat":"LambdaTilde", 'mtotal':'mtot', "distance":"dist", 'ra':'phi', 'dec':'theta',"phiorb":"phiref"}


def extract_combination_from_LI(samples_LI, p):
    """
    extract_combination_from_LI
      - reads in known columns from posterior samples
      - for selected known combinations not always available, it will compute them from standard quantities
    """
    if p in samples_LI.dtype.names:  # e.g., we have precomputed it
        return samples_LI[p]
    if p in remap_ILE_2_LI.keys():
       if remap_ILE_2_LI[p] in samples_LI.dtype.names:
         return samples_LI[ remap_ILE_2_LI[p] ]
    if (p == 'chi_eff' or p=='xi') and 'a1z' in samples_LI.dtype.names:
         m1 = samples_LI['m1']
         m2 = samples_LI['m2']
         a1z = samples_LI['a1z']
         a2z = samples_LI['a2z']
         return (m1 * a1z + m2*a2z)/(m1+m2)
    # Return cartesian components of spin1, spin2.  NOTE: I may already populate these quantities in 'Add important quantities'
    if p == 'chiz_plus':
        print(" Transforming ")
        if 'a1z' in samples_LI.dtype.names:
            return (samples_LI['a1z']+ samples_LI['a2z'])/2.
        if 'theta1' in samples_LI.dtype.names:
            return (samples_LI['a1']*np.cos(samples_LI['theta1']) + samples_LI['a2']*np.cos(samples_LI['theta2']) )/2.
#        return (samples_LI['a1']+ samples_LI['a2'])/2.
    if p == 'chiz_minus':
        print(" Transforming ")
        if 'a1z' in samples_LI.dtype.names:
            return (samples_LI['a1z']- samples_LI['a2z'])/2.
        if 'theta1' in samples_LI.dtype.names:
            return (samples_LI['a1']*np.cos(samples_LI['theta1']) - samples_LI['a2']*np.cos(samples_LI['theta2']) )/2.
#        return (samples_LI['a1']- samples_LI['a2'])/2.
    if  'theta1' in samples_LI.dtype.names:
        if p == 's1x':
            return samples_LI["a1"]*np.sin(samples_LI[ 'theta1']) * np.cos( samples_LI['phi1'])
        if p == 's1y' :
            return samples_LI["a1"]*np.sin(samples_LI[ 'theta1']) * np.sin( samples_LI['phi1'])
        if p == 's2x':
            return samples_LI["a2"]*np.sin(samples_LI[ 'theta2']) * np.cos( samples_LI['phi2'])
        if p == 's2y':
            return samples_LI["a2"]*np.sin(samples_LI[ 'theta2']) * np.sin( samples_LI['phi2'])
        if p == 'chi1_perp' :
            return samples_LI["a1"]*np.sin(samples_LI[ 'theta1']) 
        if p == 'chi2_perp':
            return samples_LI["a2"]*np.sin(samples_LI[ 'theta2']) 
    if 'lambdat' in samples_LI.dtype.names:  # LI does sampling in these tidal coordinates
        lambda1, lambda2 = lalsimutils.tidal_lambda_from_tilde(samples_LI["m1"], samples_LI["m2"], samples_LI["lambdat"], samples_LI["dlambdat"])
        if p == "lambda1":
            return lambda1
        if p == "lambda2":
            return lambda2
    if p == 'delta' or p=='delta_mc':
        return (samples_LI['m1']  - samples_LI['m2'])/((samples_LI['m1']  + samples_LI['m2']))
    # Return cartesian components of Lhat
    if p == 'product(sin_beta,sin_phiJL)':
        return np.sin(samples_LI[ remap_ILE_2_LI['beta'] ]) * np.sin(  samples_LI['phi_jl'])
    if p == 'product(sin_beta,cos_phiJL)':
        return np.sin(samples_LI[ remap_ILE_2_LI['beta'] ]) * np.cos(  samples_LI['phi_jl'])

    if p == 'mc':
        m1v= samples_LI["m1"]
        m2v = samples_LI["m2"]
        return lalsimutils.mchirp(m1v,m2v)
    if p == 'eta':
        m1v= samples_LI["m1"]
        m2v = samples_LI["m2"]
        return lalsimutils.symRatio(m1v,m2v)

    if p == 'phi1':
        return np.angle(samples_LI['a1x']+1j*samples_LI['a1y'])
    if p == 'chi_pavg':
        samples = np.array([samples_LI["m1"], samples_LI["m2"], samples_LI["a1x"], samples_LI["a1y"], samples_LI["a1z"], samples_LI["a2x"], samples_LI["a2y"], samples_LI["a2z"]]).T
        with Pool(12) as pool:   
            chipavg = np.array(pool.map(fchipavg, samples))          
        return chipavg

    if p == 'chi_p':
        samples = np.array([samples_LI["m1"], samples_LI["m2"], samples_LI["a1x"], samples_LI["a1y"], samples_LI["a1z"], samples_LI["a2x"], samples_LI["a2y"], samples_LI["a2z"]]).T
        with Pool(12) as pool:   
            chip = np.array(pool.map(fchip, samples))          
        return chip

    # Backup : access lambdat if not present
    if (p == 'lambdat' or p=='dlambdat') and 'lambda1' in samples_LI.dtype.names:
        Lt,dLt = lalsimutils.tidal_lambda_tilde(samples_LI['m1'], samples_LI['m2'],  samples_LI['lambda1'], samples_LI['lambda2'])
        if p=='lambdat':
            return Lt
        if p=='dlambdat':
            return dLt

    if p == "q"  and 'm1' in samples_LI.dtype.names:
        return samples_LI["m2"]/samples_LI["m1"]

    if 'inverse(' in p:
        # Drop first and last characters
        a=p.replace(' ', '') # drop spaces
        a = a[:len(a)-1] # drop last
        a = a[8:]
        if a =='q' and 'm1' in samples_LI.dtype.names:
            return samples_LI["m1"]/samples_LI["m2"]

    print(" No access for parameter ", p)
    return np.zeros(len(samples_LI['m1']))  # to avoid causing a hard failure

def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b





def standard_expand_samples(samples):
    """
    Do some things which add a bunch of standard fields to the samples, if I don't have them.  
    Used in plot_posterior_corner.py for example
    """
    if not 'mtotal' in samples.dtype.names and 'mc' in samples.dtype.names:  # raw LI samples use 
        q_here = samples['q']
        eta_here = q_here/(1+q_here)
        mc_here = samples['mc']
        mtot_here = mc_here / np.power(eta_here, 3./5.)
        m1_here = mtot_here/(1+q_here)
        samples = add_field(samples, [('mtotal', float)]); samples['mtotal'] = mtot_here
        samples = add_field(samples, [('eta', float)]); samples['eta'] = eta_here
        samples = add_field(samples, [('m1', float)]); samples['m1'] = m1_here
        samples = add_field(samples, [('m2', float)]); samples['m2'] = mtot_here * q_here/(1+q_here)
        
    if "theta1" in samples.dtype.names and not('chi1_perp' in samples.dtype.names):
        a1x_dat = samples["a1"]*np.sin(samples["theta1"])*np.cos(samples["phi1"])
        a1y_dat = samples["a1"]*np.sin(samples["theta1"])*np.sin(samples["phi1"])
        chi1_perp = samples["a1"]*np.sin(samples["theta1"])

        a2x_dat = samples["a2"]*np.sin(samples["theta2"])*np.cos(samples["phi2"])
        a2y_dat = samples["a2"]*np.sin(samples["theta2"])*np.sin(samples["phi2"])
        chi2_perp = samples["a2"]*np.sin(samples["theta2"])

                                      
        samples = add_field(samples, [('a1x', float)]);  samples['a1x'] = a1x_dat
        samples = add_field(samples, [('a1y', float)]); samples['a1y'] = a1y_dat
        samples = add_field(samples, [('a2x', float)]);  samples['a2x'] = a2x_dat
        samples = add_field(samples, [('a2y', float)]);  samples['a2y'] = a2y_dat
        samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
        samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp
        if not 'chi_eff' in samples.dtype.names:
            samples = add_field(samples, [('chi_eff',float)]); samples['chi_eff'] = (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["m1"]+samples["m2"])
 
    elif 'a1x' in samples.dtype.names and not 'chi1_perp' in samples.dtype.names:
        chi1_perp = np.sqrt(samples['a1x']**2 + samples['a1y']**2)
        chi2_perp = np.sqrt(samples['a2x']**2 + samples['a2y']**2)
        samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
        samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp

        # Askold: this part will check if phi1, phi2, phi12 are present. If not, compute and add the missing ones
        phi_fields = ['phi1', 'phi2', 'phi12']
        phi_func_dict = {
            'phi1': lambda samples: np.arctan2(samples['a1x'], samples['a1y']),
            'phi2': lambda samples: np.arctan2(samples['a2x'], samples['a2y']),
            'phi12': lambda samples: samples['phi2'] - samples['phi1']
        }

        for field_name in phi_fields:
            if not (field_name in samples.dtype.names):
                samples = add_field(samples, [(field_name, float)])
                samples[field_name] = phi_func_dict[field_name](samples)

    if 'lambda1' in samples.dtype.names and not ('lambdat' in samples.dtype.names):
        Lt,dLt = lalsimutils.tidal_lambda_tilde(samples['m1'], samples['m2'],  samples['lambda1'], samples['lambda2'])
        samples = add_field(samples, [('lambdat', float)]); samples['lambdat'] = Lt
        samples = add_field(samples, [('dlambdat', float)]); samples['dlambdat'] = dLt


    return samples


######### MUlTIPROCESSING FUNCTIONS ############
from multiprocessing import Pool 

def fchipavg(sample):
            P=lalsimutils.ChooseWaveformParams()
            P.m1 = sample[0]
            P.m2 = sample[1]
            P.s1x = sample[2]
            P.s1y = sample[3]
            P.s1z = sample[4]
            P.s2x = sample[5]
            P.s2y = sample[6]
            P.s2z = sample[7]
            if (P.s1x == 0 and P.s1y == 0 and P.s2x == 0 and P.s2y == 0):
                chipavg = 0
            elif (P.s1x == 0 and P.s1y == 0 and P.s1z == 0) or (P.s2x == 0 and P.s2y == 0 and P.s2z == 0):
                chipavg = P.extract_param('chi_p')
            else:
                chipavg = P.extract_param('chi_pavg')
            return chipavg     

def fchip(sample):
            P=lalsimutils.ChooseWaveformParams()
            P.m1 = sample[0]
            P.m2 = sample[1]
            P.s1x = sample[2]
            P.s1y = sample[3]
            P.s1z = sample[4]
            P.s2x = sample[5]
            P.s2y = sample[6]
            P.s2z = sample[7]
            chip = P.extract_param('chi_p')
            return chip  
