"""Bounded automatic public-EOB coordinate construction, after final priors.

Only conservative quadratures and averaged public RR are evaluated. No waveform,
likelihood, posterior archive or case-specific coefficient table is used.
"""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import re
import shlex
import numpy as np

VERSION='auto-public-eob-v1'
CACHE='eccentric_phase_auto_cache.json'
MAX_PROFILES=48
FEATURES=['mc_detector','delta','e','s1z','s2z']


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def option(line,flag,default=None):
    tokens=shlex.split(line);out=default
    for i,t in enumerate(tokens):
        if t==flag:
            if i+1==len(tokens):raise ValueError('Missing value for '+flag)
            out=tokens[i+1];j=i+2
            while out.count('[')>out.count(']') and j<len(tokens):out+=' '+tokens[j];j+=1
        elif t.startswith(flag+'='):out=t.split('=',1)[1]
    return out


def interval(text,name):
    # Accept numpy's numeric repr without executing arbitrary configuration code.
    text=re.sub(r'(?:np\.)?float(?:32|64)\(([-+.eE0-9]+)\)',r'\1',str(text))
    try:values=np.asarray(ast.literal_eval(text),float)
    except (ValueError,SyntaxError,TypeError) as error:raise ValueError('Invalid '+name+' interval') from error
    if values.shape!=(2,) or not np.isfinite(values).all() or values[0]>values[1]:raise ValueError('Invalid '+name+' interval')
    return values.tolist()


def prior_specification(lines,zero_spin=False):
    """Union of resolved fit-stage supports; no changes to the physical prior."""
    from .eccentric_phase_pipeline import eligible
    stages=[]
    for line in lines:
        if not eligible(line):continue
        z=float(option(line,'--source-redshift',0));factor=1+z
        if not np.isfinite(z) or z<0:raise ValueError('Invalid source redshift')
        mc=np.array(interval(option(line,'--mc-range','[0.9,250]'),'mc'))*factor
        eta=interval(option(line,'--eta-range','[0.01,0.2499999]'),'eta')
        total=np.array(interval(option(line,'--mtot-range','[1,300]'),'mtot'))*factor
        ecc=[float(option(line,'--ecc-min',0)),float(option(line,'--ecc-max',.9))]
        kind=option(line,'--eccentricity-prior','uniform')
        if kind=='log_uniform' and ecc[0]==0:ecc[0]=.001  # same native CIP support rule
        spin=float(option(line,'--chi-max',1));small=float(option(line,'--chi-small-max',spin))
        aligned=option(line,'--aligned-prior','uniform')
        if not (0<mc[0]<=mc[1] and 0<eta[0]<=eta[1]<=.25 and 0<total[0]<=total[1]
                and 0<=ecc[0]<=ecc[1]<1 and np.isfinite([spin,small,*ecc]).all() and spin>=0 and small>=0):
            raise ValueError('Invalid resolved mass/eccentricity/spin prior support')
        lower=0 if aligned=='alignedspin-zprior-positive' else -0.999
        upper=1 if lower==0 else .999
        spins=[[lower*spin,upper*spin],[lower*small,upper*small]] if not zero_spin else [[0.,0.],[0.,0.]]
        stages.append(dict(mc_detector=mc.tolist(),eta=eta,mtot_detector=total.tolist(),e=ecc,
                           s1z=spins[0],s2z=spins[1],source_redshift=z,eccentricity_prior=kind,aligned_prior=aligned))
    if not stages:raise ValueError('No eligible eccentric fit stage for automatic EOB map')
    result={name:[min(s[name][0] for s in stages),max(s[name][1] for s in stages)]
            for name in ['mc_detector','eta','mtot_detector','e','s1z','s2z']}
    result.update(stages=stages,zero_spin=bool(zero_spin))
    return result


def sources():
    # The stored correction is relative to full_e_advance, so its modules are keyed too.
    import scipy
    from . import eccentric_phase,eccentric_phase_models
    from .eccentric_phase_eob import analytic_eob,aligned_spin,radial_core,public_rr,public_transport
    modules=[analytic_eob,aligned_spin,radial_core,public_rr,public_transport,eccentric_phase,eccentric_phase_models]
    return {m.__name__.rsplit('.',1)[-1]:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules} | {
        'builder':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy_version':np.__version__,'scipy_version':scipy.__version__,
        'cumulative_simpson':public_transport.cumulative_simpson.__name__}


def profile(point,reference,target):
    from .eccentric_phase_eob.aligned_spin import AlignedSpinEOB
    from .eccentric_phase_eob.public_transport import transport,initial_orbit,mean_to_clock
    from .eccentric_phase_models import full_e_advance
    from .eccentric_phase import MTSUN
    mc,d,e,s1,s2=point;eta=(1-d*d)/4;mass=mc/eta**.6
    factory=lambda nu:AlignedSpinEOB(nu,s1,s2)
    w0=np.pi*mass*MTSUN*reference;wa=np.pi*mass*MTSUN*target
    result=transport(eta,e,w0,wa,n=32,rtol=2e-5,model_factory=factory)
    _,orbit=initial_orbit(eta,e,w0,n=32,model_factory=factory)
    angles=np.linspace(0,2*np.pi,128,endpoint=False)
    clock=mean_to_clock(orbit,e,angles,n=257)
    residual=np.angle(np.exp(1j*(clock-angles)))
    harmonics=[2*np.mean(residual*np.sin(k*angles)) for k in range(1,9)]
    m1,m2=mass*(1+d)/2,mass*(1-d)/2
    correction=result['advance']-float(full_e_advance(m1,m2,e,reference,target))
    return np.r_[correction,harmonics],dict(max_cycle_fraction=result['max_cycle_fraction'],min_peri=result['min_peri'])


def domain(seed,priors):
    """A local prior-clipped chart around the coincidence, with a global exterior."""
    m1,m2=seed['m1'],seed['m2'];mass=m1+m2;d=abs(m1-m2)/mass;eta=m1*m2/mass**2
    spins=[seed.get('s1z',0.),seed.get('s2z',0.)]
    if m1<m2:spins=spins[::-1]
    dlo,dhi=np.sqrt(np.maximum(0,1-4*np.asarray(priors['eta'][::-1])))
    limits=np.array([priors['mc_detector'],[dlo,dhi],priors['e'],priors['s1z'],priors['s2z']],float).T
    # Circular clocks and super-Kerr orbits are not needed for the EOB patch.
    limits[0,2]=max(limits[0,2],.005);limits[0,3:]=np.maximum(limits[0,3:],-.98)
    limits[1,3:]=np.minimum(limits[1,3:],.98)
    center=np.array([mass*eta**.6,d,seed.get('eccentricity') if seed.get('eccentricity') is not None else sum(priors['e'])/2,*spins],float)
    if np.any(limits[0]>limits[1]):return None,None,None
    center=np.clip(center,limits[0],limits[1])
    radii=np.array([max(.15*center[0],.01),.25,.25,.5,.5])
    bounds=np.array([np.maximum(limits[0],center-radii),np.minimum(limits[1],center+radii)])
    center=bounds.mean(axis=0);scale=(bounds[1]-bounds[0])/2
    return center,np.where(scale>0,scale,1.),bounds


def generate(seed,priors,reference,target,cache_path=CACHE,evaluator=None):
    from .eccentric_phase_models import encode_map,load_map
    from .eccentric_phase_eob.radial_core import DomainError,AccuracyError,AdiabaticError
    model_sources=sources()
    signature=dict(version=VERSION,seed=seed,priors=priors,reference=reference,target=target,sources=model_sources)
    signature_hash=digest(signature);path=Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        saved=json.loads(path.read_text())
        if saved.get('signature_sha256')==signature_hash:
            load_map(saved['map_data'])
            if digest(saved['record'])!=saved['record_sha256'] or encode_map(saved['record'])!=saved['map_data']:raise ValueError('Corrupt automatic EOB map cache')
            return saved['map_data'],dict(saved['report'],cache_reused=True)
    center,scale,bounds=domain(seed,priors)
    chart_available=center is not None
    if center is None:center=np.array([1.,0.,.01,0.,0.]);scale=np.ones(5);bounds=np.array([center,center])
    active=np.flatnonzero(bounds[1]>bounds[0]);powers=np.array([p for p in itertools.product(range(3),repeat=5) if sum(p)<=2 and all(p[k]==0 for k in set(range(5))-set(active))])
    provenance=dict(generator=VERSION,public_equations='https://arxiv.org/abs/2104.11705v2',source_sha256=model_sources,
                    input_sha256=signature_hash,seed=seed,priors={k:v for k,v in priors.items() if k!='stages'},
                    initialization='geometric-e and Kepler-mean; public LO SO/SS Hamiltonian and relative 2PN RR',
                    coverage='Prior-clipped local chart around coincidence; smooth full-e Peters exterior; native prior support unchanged')
    # Full weight (inner 3/4 of the taper radius) is exactly the sampled box;
    # extrapolation is confined to the taper. Degenerate prior dimensions use
    # a neutral taper coordinate, not a hole.
    radius=np.where(bounds[1]>bounds[0],scale/.75,1.5*scale)
    table_bounds=np.array([center-radius,center+radius])
    values=[];design_points=[];failures=[];validation=[];attempts=0
    evaluator=profile if evaluator is None else evaluator
    if reference==target:state='identity-frequency-fallback'
    elif not chart_available or np.any(bounds[0]>bounds[1]) or priors['e'][1]<.005:state='peters-no-EOB-chart'
    else:
        rng=np.random.default_rng(int(signature_hash[:8],16));state='generated'
        desired=min(MAX_PROFILES-5,len(powers)+8)
        candidates=np.vstack([center,rng.uniform(bounds[0],bounds[1],(MAX_PROFILES-1,5))])
        for point in candidates:
            if attempts>=MAX_PROFILES:break
            attempts+=1
            mass=point[0]/((1-point[1]**2)/4)**.6
            if not priors['mtot_detector'][0]<=mass<=priors['mtot_detector'][1]:
                failures.append({'point':point.tolist(),'reason':'outside joint total-mass prior'});continue
            try:
                value,diagnostic=evaluator(point,reference,target)
                if np.asarray(value).shape!=(9,) or not np.isfinite(value).all():raise AccuracyError('invalid EOB coordinate profile')
            except (DomainError,AccuracyError,AdiabaticError,ValueError,ArithmeticError,RuntimeError,np.linalg.LinAlgError) as error:
                failures.append({'point':point.tolist(),'reason':type(error).__name__+': '+str(error)});continue
            row=np.prod(((point-center)/scale)[None,:]**powers,axis=1)
            entry=dict(point=point.tolist(),values=np.asarray(value).tolist(),diagnostic=diagnostic)
            if len(values)<desired:values.append(entry);design_points.append(row)
            else:validation.append((row,np.asarray(value)));values.append(entry)
            if len(design_points)>=desired and len(validation)>=5:break
    training=values[:len(design_points)];attenuation=1.;error=None
    if reference==target:
        coef=np.zeros((1,9));powers=np.zeros((1,5),int)
    elif len(training)>=len(powers):
        matrix=np.asarray(design_points);truth=np.array([v['values'] for v in training]);coef,_,rank,_=np.linalg.lstsq(matrix,truth,rcond=1e-10)
        if rank<len(powers):state='peters-rank-fallback';coef=np.zeros((1,9));powers=np.zeros((1,5),int)
        elif validation:
            errors=[abs((row@coef-value)[0])+sum(abs((row@coef-value)[1:])) for row,value in validation]
            error=float(max(errors));attenuation=min(1.,.5/max(error,1e-12));coef*=attenuation
    else:state='peters-no-EOB-chart';coef=np.zeros((1,9));powers=np.zeros((1,5),int)
    report=dict(generator=VERSION,state=state,attempts=attempts,successful_profiles=len(values),failed_profiles=len(failures),
                training_profiles=len(training),validation_profiles=len(validation),sampled_phase_error_rad=error,
                correction_attenuation=attenuation,cache_reused=False,signature_sha256=signature_hash,
                local_domain=bounds.tolist(),prior_support=priors)
    provenance.update({k:v for k,v in report.items() if k not in ['local_domain','prior_support','cache_reused']})
    record=dict(schema='rift-eccentric-map-v2',features=FEATURES,center=center.tolist(),scale=scale.tolist(),bounds=table_bounds.tolist(),
                powers=powers.tolist(),coefficients=coef.tolist(),reference_frequency=reference,target_frequency=target,provenance=provenance)
    data=encode_map(record)
    if path is not None:
        payload=dict(signature_sha256=signature_hash,record_sha256=digest(record),record=record,map_data=data,report=report,profiles=values,failures=failures)
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    return data,report
