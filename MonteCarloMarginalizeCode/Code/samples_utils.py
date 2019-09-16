
import numpy as np


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



