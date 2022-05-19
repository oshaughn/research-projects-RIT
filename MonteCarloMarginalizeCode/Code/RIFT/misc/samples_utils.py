
import numpy as np
import RIFT.lalsimutils as lalsimutils

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
        
    if "theta1" in samples.dtype.names:
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
 
    elif 'a1x' in samples.dtype.names:
        chi1_perp = np.sqrt(samples['a1x']**2 + samples['a1y']**2)
        chi2_perp = np.sqrt(samples['a2x']**2 + samples['a2y']**2)
        samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
        samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp

    if 'lambda1' in samples.dtype.names and not ('lambdat' in samples.dtype.names):
        Lt,dLt = lalsimutils.tidal_lambda_tilde(samples['m1'], samples['m2'],  samples['lambda1'], samples['lambda2'])
        samples = add_field(samples, [('lambdat', float)]); samples['lambdat'] = Lt
        samples = add_field(samples, [('dlambdat', float)]); samples['dlambdat'] = dLt


    return samples
