"""CIT research transport; default public nonspin H + leading minimal RR.

An explicit model_factory selects another independently validated adapter.

The native initialization is an EXPLICIT proxy: geometric turning-point e,
orbit-averaged azimuthal f22, and Kepler mean-anomaly radius at that epoch.
It is not the spinning waveform model's initialized state. Frequencies use
physical time. The endpoint is a slow frequency surface, never a periapsis
event. All losses use the identical mechanical work as direct dynamics.
"""
from functools import lru_cache
from math import pi,sqrt
import numpy as np
from scipy.integrate import solve_ivp
try:
    from scipy.integrate import cumulative_simpson
except ImportError:  # supported older SciPy; clock remains explicitly invertible
    from scipy.integrate import cumulative_trapezoid as cumulative_simpson
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from .analytic_eob import bound_orbit
from .public_rr import RadiatingEOB
from .radial_core import CycleIntegrals,DomainError,AdiabaticError,AccuracyError


@lru_cache(None)
def nodes(n):
    x,w=np.polynomial.legendre.leggauss(n)
    return (x+1)*pi/4,w*pi/4


def orbit_at(eta,p,e,model_factory=None):
    if model_factory is not None:
        return model_factory(eta).bound_orbit(p/(1+e),p/(1-e))
    orbit=bound_orbit(eta,p/(1+e),p/(1-e))
    orbit.adapter=RadiatingEOB(eta)
    return orbit


def fixed_cycle(orbit,n=64):
    u,w=nodes(n)
    # H is even in radial momentum; both minimal-force work and torque
    # are even, so each branch contributes the same integral here.
    values=np.array([orbit._integrand(float(v),1) for v in u])
    result=2*(w@values)
    return CycleIntegrals(*result,(float('nan'),)*4,result[0]/2)


def initial_orbit(eta,e,omega0,n=64,model_factory=None):
    guess=(1-e*e)/omega0**(2/3)
    def residual(p):
        orbit=orbit_at(eta,p,e,model_factory)
        return fixed_cycle(orbit,n).omega_phi-omega0
    def safe(p):
        try:return residual(p)
        except DomainError:return None
    # Default bracket first; widen only if an end leaves the bound domain or
    # misses the root (high e/mass), so a bracketed root is never discarded.
    ratios=[.8,1.3];values=[safe(r*guess) for r in ratios]
    if None in values or values[0]*values[1]>0:
        ratios=[.8,.9,1.,1.1,1.2,1.3,1.45,1.6,1.8,2.];values=[safe(r*guess) for r in ratios]
    for (a,fa),(b,fb) in zip(zip(ratios,values),zip(ratios[1:],values[1:])):
        if fa is not None and fb is not None and fa*fb<=0:
            p=brentq(residual,a*guess,b*guess,xtol=1e-9,rtol=1e-10)
            return p,orbit_at(eta,p,e,model_factory)
    raise DomainError('no bound orbit brackets the initial azimuthal frequency')


def mean_to_clock(orbit,e,mean_anomaly,n=1025):
    """Proxy Kepler-mean input -> uniform physical-time radial phase."""
    l=np.asarray(mean_anomaly)%(2*pi)
    eccentric=l.copy() if e<=.8 else np.full_like(l,pi)
    for _ in range(50):eccentric-=(eccentric-e*np.sin(eccentric)-l)/(1-e*np.cos(eccentric))
    if not np.all(abs(eccentric-e*np.sin(eccentric)-l)<1e-12):raise AccuracyError('Kepler equation did not converge')
    a=(orbit.r_peri+orbit.r_apo)/2
    radius=a*(1-e*np.cos(eccentric))
    u=np.arcsin(np.sqrt(np.clip((radius-orbit.r_peri)/orbit.width,0,1)))
    grid=np.linspace(0,pi/2,n)
    density=np.array([orbit._time_density(float(v),1)[0] for v in grid])
    times=cumulative_simpson(density,x=grid,initial=0)
    elapsed=PchipInterpolator(grid,times)(u)
    out=pi*elapsed/times[-1]
    return np.where(l<=pi,out,2*pi-out)%(2*pi)


def transport(eta,e0,omega0,omega1,n=64,rtol=2e-6,maximum_cycle_fraction=.1,model_factory=None):
    """Leading orbit-averaged path, returning an UNWRAPPED radial advance.

    Uses p as the independent variable and the exact E(p,e),J(p,e) Jacobian.
    Cycle changes relative to binding energy and J must remain below the
    declared ceiling. This is a necessary averaging diagnostic, not an
    error bound or a substitute for separatrix/transition checks.
    """
    if not 0<e0<1 or not 0<omega0<=omega1 or not 0<eta<=.25:
        raise DomainError('require noncircular bound forward-frequency inputs')
    p0,initial=initial_orbit(eta,e0,omega0,n,model_factory)
    maxfraction=0.;minperi=initial.r_peri
    def cycle_at(p,e):
        nonlocal maxfraction,minperi
        orbit=orbit_at(eta,p,e,model_factory);cyc=fixed_cycle(orbit,n)
        if cyc.energy_loss<=0 or cyc.angular_momentum_loss<=0:
            raise AdiabaticError('cycle losses must be positive')
        fraction=max(cyc.energy_loss/abs(orbit.energy),cyc.angular_momentum_loss/orbit.j)
        maxfraction=max(maxfraction,fraction);minperi=min(minperi,orbit.r_peri)
        if fraction>=maximum_cycle_fraction:
            raise AdiabaticError('declared averaging ceiling exceeded')
        return orbit,cyc
    def rhs(p,state):
        e=state[0];orbit,cyc=cycle_at(p,e)
        hp=p*1e-5;he=min(1e-5,e*.001)
        op,om=orbit_at(eta,p+hp,e,model_factory),orbit_at(eta,p-hp,e,model_factory)
        ep,em=orbit_at(eta,p,e+he,model_factory),orbit_at(eta,p,e-he,model_factory)
        E_p=(op.energy-om.energy)/(2*hp);J_p=(op.j-om.j)/(2*hp)
        E_e=(ep.energy-em.energy)/(2*he);J_e=(ep.j-em.j)/(2*he)
        ratio=cyc.angular_momentum_loss/cyc.energy_loss
        de=(ratio*E_p-J_p)/(J_e-ratio*E_e)
        dE=E_p+E_e*de
        if dE<=0:raise AdiabaticError('energy does not decrease with decreasing p')
        return [de,-2*pi*dE/cyc.energy_loss]
    def event(p,state):return fixed_cycle(orbit_at(eta,p,state[0],model_factory),n).omega_phi-omega1
    event.terminal=True;event.direction=1
    if omega1==omega0:
        return {'advance':0.,'initial_p':p0,'final_p':p0,'final_e':e0,'max_cycle_fraction':0.,'min_peri':minperi}
    endpoint=p0*(omega0/omega1)**(2/3)*.8
    solution=solve_ivp(rhs,[p0,endpoint],[e0,0.],events=event,rtol=rtol,atol=[1e-9,1e-6],max_step=p0*.025)
    if not solution.success or not len(solution.t_events[0]):
        raise AdiabaticError('forward averaged-frequency landmark not reached')
    state=solution.y_events[0][0]
    return {'advance':float(state[1]),'initial_p':p0,'final_p':float(solution.t_events[0][0]),
            'final_e':float(state[0]),'max_cycle_fraction':maxfraction,'min_peri':minperi}
