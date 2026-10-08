"""Experimental RF fitting features only. No prior/sampling/likelihood changes.
Units: component masses in solar masses, dimensionless L-frame spins.
Reference frequency must match supplied spins; no spin transport is performed.
"""
import re
import numpy as np
MTSUN = 4.9254909476412675e-6


def geometry(m1, m2, s1, s2, frequency=20., epsilon=.1):
    m1,m2=np.asarray(m1),np.asarray(m2)
    s1,s2=np.asarray(s1),np.asarray(s2)
    if not np.isfinite(m1).all() or not np.isfinite(m2).all() or np.any(m1<=0) or np.any(m2<=0) or not np.isfinite(frequency) or not np.isfinite(epsilon) or frequency<=0 or epsilon<=0:
        raise ValueError('Require positive masses, frequency and epsilon')
    q=m2/m1; eta=q/(1+q)**2; M=m1+m2
    v=(np.pi*M*MTSUN*frequency)**(1/3)
    w1=1/(1+q)**2; w2=q*q*w1
    L=eta/v; D=L+w1*s1[...,2]+w2*s2[...,2]
    T=w1[...,None]*s1[...,:2]+w2[...,None]*s2[...,:2]
    T2=np.sum(T*T,axis=-1); J=np.sqrt(D*D+T2)
    d=np.hypot(D,epsilon*L)
    a1=2+1.5*q; a2=2+1.5/q
    G=(a1*w1)[...,None]*s1[...,:2]+(a2*w2)[...,None]*s2[...,:2]
    # J-D evaluated without catastrophic cancellation for D>=0.
    JminusD=np.where(D>=0,np.divide(T2,J+D,out=np.zeros_like(J),where=(J+D)>0),J-D)
    return dict(q=q,eta=eta,v=v,L=L,D=D,J=J,T=T,G=G,d=d,w1=w1,w2=w2,
                a1=a1,a2=a2,JminusD=JminusD)


def scalar_features(m1,m2,s1,s2,frequency=20.,epsilon=.1):
    """Redundant RF split candidates, never replacements for four spin components.
    cone2 is regularized tan(beta)^2; phase_deficit is (J-D)/L.
    torque2 captures instantaneous two-spin vector interference.
    """
    g=geometry(m1,m2,s1,s2,frequency,epsilon)
    cone2=np.sum(g['T']**2,axis=-1)/g['d']**2
    phase_deficit=g['JminusD']/g['L']
    torque2=np.sum(g['G']**2,axis=-1)/(g['eta']*g['v']**2)**2
    return np.stack([cone2,phase_deficit,torque2],axis=-1)


def geometric4(m1, m2, s1, s2, frequency=20., phase_excess=False):
    """Four-coordinate total-transverse-spin chart in the existing L frame.

    |S_perp|/M^2, total-spin azimuth in [-pi,pi), and two
    signed residuals in the total-spin frame. At exact cancellation choose
    azimuth zero; this chart has a seam but loses no transverse information.
    phase_excess=True instead uses H=(J-|J_parallel|)/L_N as the radius.
    """
    s1,s2=np.asarray(s1,float),np.asarray(s2,float)
    if s1.shape[-1:]!=(3,) or s2.shape[-1:]!=(3,) or not np.isfinite(s1).all() or not np.isfinite(s2).all():
        raise ValueError('Require finite three-component spins')
    g=geometry(m1,m2,s1,s2,frequency)
    radius=np.hypot(g['T'][...,0],g['T'][...,1])
    theta=np.where(radius==0.,0.,np.arctan2(g['T'][...,1],g['T'][...,0]))
    theta=(theta+np.pi)%(2*np.pi)-np.pi
    den=g['L']*(g['J']+np.abs(g['D']))
    first=np.divide(radius**2,den,out=np.zeros_like(radius),where=den>0) if phase_excess else radius
    h=np.hypot(g['w1'],g['w2'])
    r=(-g['w2'][...,None]*np.asarray(s1)[...,:2]+g['w1'][...,None]*np.asarray(s2)[...,:2])/h[...,None]
    co,si=np.cos(theta),np.sin(theta)
    return np.stack([first,theta,co*r[...,0]+si*r[...,1],-si*r[...,0]+co*r[...,1]],axis=-1)


def geometric4_inverse(m1,m2,z1,z2,features,frequency=20.,phase_excess=False):
    """Diagnostic inverse at fixed masses/aligned spins; never a new sampler."""
    f=np.asarray(features,float)
    if f.shape[-1]!=4 or not np.isfinite(f).all() or np.any(f[...,0]<0):
        raise ValueError('Require four finite coordinates and nonnegative radius')
    s1=np.zeros(f.shape[:-1]+(3,));s2=np.zeros_like(s1)
    s1[...,2]=z1;s2[...,2]=z2
    g=geometry(m1,m2,s1,s2,frequency)
    first,theta,parallel,perpendicular=np.moveaxis(f,-1,0)
    radius=g['L']*np.sqrt(first*(first+2*np.abs(g['D'])/g['L'])) if phase_excess else first
    co,si=np.cos(theta),np.sin(theta)
    T=np.stack([radius*co,radius*si],axis=-1)
    r=np.stack([parallel*co-perpendicular*si,parallel*si+perpendicular*co],axis=-1)
    h=np.hypot(g['w1'],g['w2'])
    s1[...,:2]=(g['w1'][...,None]*T-g['w2'][...,None]*h[...,None]*r)/h[...,None]**2
    s2[...,:2]=(g['w2'][...,None]*T+g['w1'][...,None]*h[...,None]*r)/h[...,None]**2
    return s1,s2

GEOMETRIC4_NAMES=('rf_total_perp','rf_total_azimuth','rf_residual_parallel','rf_residual_perpendicular')
PHASE_EXCESS_NAMES=('rf_phase_excess',)+GEOMETRIC4_NAMES[1:]
GEOMETRIC4_MODES=('geometric4','geometric4-phase-excess')
# physics3 appends three scalars to the eight native fit coordinates: 11 fit
# coordinates for 8 degrees of freedom. This RF transverse policy requires
# a nonredundant basis; other CIP configurations can use redundant fit features.
RETIRED_MODES=('physics3',)
RETIRED_MESSAGE=('RF transverse-spin mode physics3 is retired: it gives CIP 11 fit coordinates '
    'for 8 degrees of freedom. Use geometric4 (auto selects it) or off.')

def geometric_names(mode):
    return PHASE_EXCESS_NAMES if mode=='geometric4-phase-excess' else GEOMETRIC4_NAMES

FEATURE_NAMES = ('rf_cone2', 'rf_phase_deficit', 'rf_torque2')
TRANSVERSE = ('s1x', 's1y', 's2x', 's2y')
NATIVE_FEATURES = ('delta_mc','mu1','mu2','chiMinus') + TRANSVERSE

def extract(P, name):
    """Fit-only extraction; P masses are SI, reference frequency follows P."""
    import lal
    if name not in FEATURE_NAMES + GEOMETRIC4_NAMES + PHASE_EXCESS_NAMES:
        return P.extract_param(name)
    is_geometric=name in GEOMETRIC4_NAMES + PHASE_EXCESS_NAMES
    function = geometric4 if is_geometric else scalar_features
    names = PHASE_EXCESS_NAMES if name=='rf_phase_excess' else GEOMETRIC4_NAMES if is_geometric else FEATURE_NAMES
    extra={'phase_excess':name=='rf_phase_excess'} if is_geometric else {}
    values = function(P.m1/lal.MSUN_SI, P.m2/lal.MSUN_SI,
        np.array([P.s1x,P.s1y,P.s1z]), np.array([P.s2x,P.s2y,P.s2z]), P.fref, **extra)
    return values[names.index(name)]

def convert(x, coord_names, low_level_coord_names, frequency, converter, **kwargs):
    """Build fitting features using the exact same physical conversion as native CIP."""
    if 'rf_phase_excess' in coord_names and 'rf_total_perp' in coord_names:
        raise ValueError('Do not mix geometric4 radius and phase-excess coordinates')
    names = PHASE_EXCESS_NAMES if 'rf_phase_excess' in coord_names else GEOMETRIC4_NAMES if set(GEOMETRIC4_NAMES).intersection(coord_names) else FEATURE_NAMES
    base = [p for p in coord_names if p not in names]
    native = converter(x, coord_names=base, low_level_coord_names=low_level_coord_names, **kwargs)
    physical_names = ['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    # Avoid native per-row fallback for aligned components in the standard spherical
    # sampler. Use exactly chi*cos(theta); native converter still owns masses/Kerr.
    spherical = all(p in low_level_coord_names for p in
        ('chi1','chi2','cos_theta1','cos_theta2','phi1','phi2'))
    if spherical:
        requested = [p for p in physical_names if p not in ('s1z','s2z')]
        converted = converter(x, coord_names=requested,
            low_level_coord_names=low_level_coord_names, **kwargs)
        physical = np.empty((len(x),8))
        xf = np.asarray(x,dtype=float)
        for i,p in enumerate(physical_names):
            if p in ('s1z','s2z'):
                n = p[1]
                physical[:,i] = xf[:,low_level_coord_names.index('chi'+n)]*xf[:,low_level_coord_names.index('cos_theta'+n)]
            else:
                physical[:,i] = converted[:,requested.index(p)]
    else:
        physical = converter(x, coord_names=physical_names,
            low_level_coord_names=low_level_coord_names, **kwargs)
    # Native enforce_kerr conversion marks invalid rows with -inf. Preserve rejection
    # instead of letting one invalid proposal abort the entire valid prediction batch.
    valid = np.isfinite(physical).all(axis=1) & (physical[:,0]>0) & (physical[:,1]>0)
    if kwargs.get('enforce_kerr', False):
        valid &= (np.sum(physical[:,2:5]**2,axis=1)<=1) & (np.sum(physical[:,5:8]**2,axis=1)<=1)
    values = np.full((len(x),len(names)), -np.inf)
    function = geometric4 if names in (GEOMETRIC4_NAMES,PHASE_EXCESS_NAMES) else scalar_features
    extra={'phase_excess':names==PHASE_EXCESS_NAMES} if function is geometric4 else {}
    values[valid] = function(physical[valid,0], physical[valid,1],
        physical[valid,2:5], physical[valid,5:8], frequency, **extra)
    out = np.empty((len(x),len(coord_names)))
    for i,p in enumerate(coord_names):
        out[:,i] = values[:,names.index(p)] if p in names else native[:,base.index(p)]
    out[~valid] = -np.inf
    return out

def sampling_problem(fit_names, sampled_names):
    """Check distinct sampling coordinates, not repeated CLI argument entries."""
    if len(set(sampled_names)) != len(sampled_names):
        return 'RF transverse-spin sampling coordinates must be unique'
    if len(fit_names) > len(sampled_names):
        return "Fit uses {} coordinates {} but samples only {} {}".format(
            len(fit_names), fit_names, len(sampled_names), sampled_names)
    return None


def _stage_sampling_problem(tokens):
    def values(flag): return [tokens[i+1] for i,t in enumerate(tokens[:-1]) if t==flag]
    parameters = values('--parameter')
    return sampling_problem(parameters + values('--parameter-implied'),
                            parameters + values('--parameter-nofit'))


def stage_arguments(line, mode, detector_chirp_mass, applicable, frequency):
    """Leave reduced/non-RF stages unchanged; activate only the complete L-frame RF fit."""
    import shlex
    if not enabled(mode, detector_chirp_mass, applicable):
        return line
    tokens=shlex.split(line)
    if not _supports_physics3(tokens):
        return line
    active_mode = 'geometric4' if mode == 'auto' else mode
    if active_mode in GEOMETRIC4_MODES:
        _require_geometric4_basis(tokens)
    if '--rf-transverse-spin-coordinates' in tokens:
        raise ValueError('Duplicate RF transverse-spin activation')
    if not np.isfinite(float(frequency)) or float(frequency)<=0:
        raise ValueError('RF reference frequency must be finite and positive')
    # Explicit fref replaces a stage-local value; it is the ILE spin reference, not fmin.
    # Edit the text in place: re-quoting every token would quote [lo,hi] ranges.
    line=re.sub(r'(^|\s)--fref(\s+|=)\S+', ' ', line)
    return line.rstrip()+' --rf-transverse-spin-coordinates '+active_mode+' --fref '+str(float(frequency))


def _supports_physics3(tokens):
    def values(flag): return [tokens[i+1] for i,t in enumerate(tokens[:-1]) if t==flag]
    fit=values('--parameter')+values('--parameter-implied')
    return values('--fit-method')==['rf'] and set(NATIVE_FEATURES).issubset(fit)


def _require_geometric4_basis(tokens):
    fit=[tokens[i+1] for i,t in enumerate(tokens[:-1]) if t in ('--parameter','--parameter-implied')]
    if len(fit)!=8 or set(fit)!=set(NATIVE_FEATURES):
        raise ValueError('geometric4 requires exactly the eight native mass/aligned/transverse fitting coordinates')


def revalidate_stage(line):
    """Recheck an activated stage after pipeline rewrites of helper output.

    A rewrite that drops the native basis (e.g. delta_mc -> eta) would otherwise
    fail in every CIP job of that stage, after ILE has run. Fail at build time
    instead, for every activated transverse-spin mode.
    """
    import shlex
    tokens=[]
    for t in shlex.split(line):
        tokens += t.split('=',1) if t.startswith('--') and '=' in t else [t]
    modes=[tokens[i+1] for i,t in enumerate(tokens[:-1]) if t=='--rf-transverse-spin-coordinates']
    if modes:
        mode=modes[-1]  # argparse uses the last value for scalar options.
        if mode in RETIRED_MODES:
            raise ValueError(RETIRED_MESSAGE)
        if mode in GEOMETRIC4_MODES:
            _require_geometric4_basis(tokens)
            problem = _stage_sampling_problem(tokens)
            if problem:
                raise ValueError(problem)
        if not _supports_physics3(tokens):
            raise ValueError('A pipeline rewrite removed the RF basis from an activated stage; '
                'set the RF transverse-spin option to off or drop the rewriting option: '+line.strip())
    return line


def enabled(mode, detector_chirp_mass, applicable):
    """Resolve the opt-in policy before constructing the native phase-fit schedule."""
    if mode not in (None,'off','auto','physics3','geometric4','geometric4-phase-excess'):
        raise ValueError('Unknown RF transverse-spin mode')
    if mode in RETIRED_MODES:
        raise ValueError(RETIRED_MESSAGE)
    if mode in (None,'off'):
        return False
    if not applicable:
        if mode in GEOMETRIC4_MODES:
            raise ValueError('RF transverse-spin coordinates require a precessing BBH analysis')
        return False
    try: mc=float(detector_chirp_mass) if not isinstance(detector_chirp_mass,(bool,np.bool_)) else float('nan')
    except (TypeError,ValueError): mc=float('nan')
    if mode == 'auto' and not (np.isfinite(mc) and 0 < mc < 20):
        return False
    return True
