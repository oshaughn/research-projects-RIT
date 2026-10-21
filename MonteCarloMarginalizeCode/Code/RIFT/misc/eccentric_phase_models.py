"""Selectable waveform-free circular coordinates, including frozen EOB maps.

The tabulated option is a coordinate definition, not an evolution accuracy
claim. Outside its declared box it smoothly returns to full-e Peters. Fit-only
mode keeps native sampling. The optional sampling adapter includes the inverse
angular Jacobian for tabulated clocks.
"""
import base64
from functools import lru_cache
import json
import numpy as np

PROCEDURES=('peters-small-e','peters-full-e','eob-tabulated')
MAX_DATA=48000


@lru_cache(maxsize=1)
def nodes():
    x,w=np.polynomial.legendre.leggauss(64)
    return (x+1)/2,w/2


def full_e_advance(m1,m2,e,reference,target):
    """Peters quadrupole radial phase at equal mean-frequency endpoints.

    Solve the full-e Peters invariant at fixed Gauss nodes in log frequency.
    Circular/small-e limits are continuous. Chunking bounds working memory.
    """
    from RIFT.misc.eccentric_phase import phase_advance,MTSUN
    # Reuse physical/configuration validation; not its truncated phase.
    phase_advance(m1,m2,e,reference,target)
    m1,m2,e=np.broadcast_arrays(np.asarray(m1,float),np.asarray(m2,float),np.asarray(e,float))
    shape=e.shape;m1,m2,e=[v.ravel() for v in [m1,m2,e]]
    out=np.empty(len(e));length=(2/3)*np.log(target/reference)
    if length<0:raise ValueError('Full-e coordinate requires reference <= target')
    if length==0:return np.zeros(shape)
    t,w=nodes();b=870/2299;c=121/304
    for start in range(0,len(e),1024):
        sl=slice(start,start+1024);ee=e[sl,None];total=m1[sl]+m2[sl]
        eta=m1[sl]*m2[sl]/total**2;x0=(np.pi*total*MTSUN*reference)**(2/3)
        le0=np.log(np.maximum(ee,1e-150));u=length*t[None,:]
        g0=np.log1p(-ee*ee)-(12/19)*le0-b*np.log1p(c*ee*ee)
        lo=le0-(19/12)*length-4;hi=np.broadcast_to(le0,(len(total),len(t))).copy()
        le=le0-(19/12)*u
        for _ in range(32):
            e2=np.exp(2*le)
            residual=np.log1p(-e2)-(12/19)*le-b*np.log1p(c*e2)-g0-u
            if np.max(abs(residual))<2e-14:break
            lo=np.where(residual>0,le,lo);hi=np.where(residual<0,le,hi)
            derivative=-2*e2/(1-e2)-12/19-2*b*c*e2/(1+c*e2)
            update=le-residual/derivative
            le=np.where((update>=lo)&(update<=hi),update,(lo+hi)/2)
        e2=np.where(ee==0,0,np.exp(2*le))
        flux=(1-e2)**3.5/(1+73/24*e2+37/96*e2**2)
        out[sl]=5/(64*eta)*x0**(-2.5)*length*((np.exp(-2.5*u)*flux)@w)
    return out.reshape(shape)


def encode_map(record):
    data=base64.urlsafe_b64encode(json.dumps(record,separators=(',',':'),allow_nan=False).encode()).decode()
    load_map(data)
    return data


@lru_cache(maxsize=8)
def load_map(data):
    if not isinstance(data,str) or len(data)>MAX_DATA:raise ValueError('Invalid/oversized eccentric map payload')
    try:r=json.loads(base64.b64decode(data,altchars=b'-_',validate=True))
    except Exception as error:raise ValueError('Invalid eccentric map payload') from error
    if not isinstance(r,dict) or any(k not in r for k in ['center','scale','bounds','powers','coefficients','reference_frequency','target_frequency']):
        raise ValueError('Incomplete eccentric map record')
    if r.get('schema') not in ('rift-eccentric-map-v1','rift-eccentric-map-v2'):raise ValueError('Unknown eccentric map schema')
    expected=['M','delta','e','chi_eff','chi_a'] if r['schema']=='rift-eccentric-map-v1' else ['mc_detector','delta','e','s1z','s2z']
    if r.get('features')!=expected:raise ValueError('Unsupported eccentric map features')
    center=np.asarray(r['center'],float);scale=np.asarray(r['scale'],float)
    bounds=np.asarray(r['bounds'],float);powers=np.asarray(r['powers'],float);coef=np.asarray(r['coefficients'],float)
    if center.shape!=(5,) or scale.shape!=(5,) or bounds.shape!=(2,5):raise ValueError('Invalid eccentric map domain')
    if powers.ndim!=2 or powers.shape[1]!=5 or len(powers)>56 or coef.ndim!=2 or coef.shape[0]!=len(powers) or not 2<=coef.shape[1]<=17:raise ValueError('Invalid eccentric map polynomial')
    if np.any(abs(coef)>1e9):raise ValueError('Unbounded eccentric map coefficients')
    if any(not np.isfinite(v).all() for v in [center,scale,bounds,powers,coef]) or np.any(scale<=0) or np.any(bounds[0]>=bounds[1]):raise ValueError('Nonfinite/degenerate eccentric map')
    if np.any(powers<0) or np.any(powers!=np.floor(powers)) or np.any(powers.sum(axis=1)>3):raise ValueError('Invalid polynomial powers')
    if not isinstance(r.get('provenance'),dict) or not r['provenance']:raise ValueError('Map provenance is required')
    from RIFT.misc.eccentric_phase import validate_frequencies
    validate_frequencies(r['reference_frequency'],r['target_frequency'])
    if r['target_frequency']<r['reference_frequency']:raise ValueError('Backward map endpoint')
    for v in [center,scale,bounds,powers,coef]:v.flags.writeable=False
    return r,center,scale,bounds,powers.astype(int),coef


def map_profile(m1,m2,e,s1z,s2z,reference,target,data):
    r,center,scale,bounds,powers,coef=load_map(data)
    if r['reference_frequency']!=reference or r['target_frequency']!=target:raise ValueError('Frozen map frequencies differ from workflow decision')
    # Spin associations follow the actual bodies, including swapped labels.
    total=m1+m2;d=(m1-m2)/total
    ce=(m1*s1z+m2*s2z)/total;ca=np.where(d>=0,(s1z-s2z)/2,(s2z-s1z)/2)
    if r['schema']=='rift-eccentric-map-v2':
        primary=np.where(d>=0,s1z,s2z);secondary=np.where(d>=0,s2z,s1z)
        feats=np.stack([total*((1-d*d)/4)**.6,abs(d),e,primary,secondary],axis=-1)
    else:feats=np.stack([total,abs(d),e,ce,ca],axis=-1)
    z=np.clip((feats-center)/scale,-4,4)
    predicted=np.zeros(feats.shape[:-1]+(coef.shape[1],))
    for power,coeff in zip(powers,coef):predicted+=np.prod(z**power,axis=-1)[...,None]*coeff
    middle=bounds.mean(axis=0);radius=(bounds[1]-bounds[0])/2
    edge=np.clip(4*(1-abs((feats-middle)/radius)),0,1)
    weight=np.prod(edge*edge*(3-2*edge),axis=-1)
    advance=full_e_advance(m1,m2,e,reference,target)
    advance+=weight*(predicted[...,0] if r['schema']=='rift-eccentric-map-v2' else predicted[...,0]-advance)
    harmonics=predicted[...,1:]*weight[...,None]
    # A fixed coordinate regularization gives a global dtheta/dangle >= .2,
    # independent of interpolation quality; no physical row is rejected.
    norm=np.sum(abs(harmonics)*np.arange(1,harmonics.shape[-1]+1),axis=-1)
    harmonics*=np.minimum(1,.8/np.maximum(norm,1e-300))[...,None]
    return advance,harmonics


def transform(m1,m2,e,angle,reference,target,procedure,data,s1z,s2z,inverse=False):
    from RIFT.misc.eccentric_phase import phase_advance
    if procedure not in PROCEDURES:raise ValueError('Unknown eccentric coordinate procedure')
    if procedure=='eob-tabulated' and data is None:raise ValueError('EOB coordinates require a frozen numerical map')
    if procedure!='eob-tabulated' and data is not None:raise ValueError('A numerical map requires eob-tabulated')
    m1,m2,e,angle,s1z,s2z=np.broadcast_arrays(*[np.asarray(v,float) for v in [m1,m2,e,angle,s1z,s2z]])
    baseline=phase_advance(m1,m2,e,reference,target)
    if not np.isfinite(angle).all():raise ValueError('Require finite angles')
    if procedure=='eob-tabulated' and not np.isfinite(s1z+s2z).all():raise ValueError('Require finite aligned spin components')
    if reference==target:return angle%(2*np.pi)  # native high-mass fallback
    if procedure=='peters-small-e':advance=baseline;harmonics=None
    elif procedure=='peters-full-e':advance=full_e_advance(m1,m2,e,reference,target);harmonics=None
    else:advance,harmonics=map_profile(m1,m2,e,s1z,s2z,reference,target,data)
    if harmonics is None:return (angle-advance if inverse else angle+advance)%(2*np.pi)
    k=np.arange(1,harmonics.shape[-1]+1)
    if not inverse:return (angle+np.sum(harmonics*np.sin(angle[...,None]*k),axis=-1)+advance)%(2*np.pi)
    theta=(angle-advance)%(2*np.pi);lo=np.zeros_like(theta);hi=np.full_like(theta,2*np.pi)
    # Certified monotonic bisection includes seams and scalar/batched inputs.
    for _ in range(48):
        mid=(lo+hi)/2;value=mid+np.sum(harmonics*np.sin(mid[...,None]*k),axis=-1)
        lo=np.where(value<theta,mid,lo);hi=np.where(value>=theta,mid,hi)
    return ((lo+hi)/2)%(2*np.pi)


def add_arguments(parser,build=False):
    parser.add_argument('--eccentric-phase-procedure',choices=PROCEDURES,default='peters-full-e',help='Explicit fit/proposal coordinate; physical priors and waveform inputs remain native')
    parser.add_argument('--eccentric-phase-map-data',default=None,help='Frozen base64 numerical map emitted by the workflow builder')
    if build:parser.add_argument('--eccentric-phase-map',default=None,help='Optional numerical map override; otherwise EOB is generated automatically from the coincidence and resolved priors')


def worker_options(options):
    return dict(procedure=getattr(options,'eccentric_phase_procedure','peters-full-e'),
                map_data=getattr(options,'eccentric_phase_map_data',None))


def map_path(path):
    from pathlib import Path
    if str(path).startswith('builtin:'):raise ValueError('No event-specific maps are packaged; use automatic build-time EOB generation')
    return Path(path).resolve(strict=True)


def freeze(options,native,target,allow_pending=False):
    import hashlib
    from pathlib import Path
    selected=worker_options(options);path=getattr(options,'eccentric_phase_map',None)
    if path:
        if selected['map_data'] is not None:raise ValueError('Specify a map file or frozen map data, not both')
        selected['map_data']=encode_map(json.loads(map_path(path).read_text()))
    procedure,data=selected['procedure'],selected['map_data']
    if procedure not in PROCEDURES:raise ValueError('Unknown eccentric coordinate procedure')
    if procedure=='eob-tabulated' and data is None and allow_pending:
        return dict(coordinate_procedure=procedure,coordinate_map_data=None,coordinate_map_sha256=None,coordinate_map_pending=True)
    if procedure=='eob-tabulated':
        r=load_map(data)[0]
        if native!=target and (r['reference_frequency']!=native or r['target_frequency']!=target):raise ValueError('EOB map frequencies differ from the resolved workflow anchor')
    elif data is not None:raise ValueError('Numerical map requires eob-tabulated')
    options.eccentric_phase_map_data=data
    return dict(coordinate_procedure=procedure,coordinate_map_data=data,coordinate_map_pending=False,
                coordinate_map_sha256=None if data is None else hashlib.sha256(data.encode()).hexdigest())
