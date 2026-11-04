"""Independent aligned-BH-spin EOB research adapter, public equations only.

Khalil et al., arXiv:2104.11705v2, real H spin terms Eqs17-18;
RR Eqs40,43,45,47-48,50-54 in ONE gauge, including BH quadrupoles.
Nonspin H is the existing uncalibrated 3PN resummation; spin H retains
LO SO(1.5PN) and LO SS(2PN), including momentum-dependent SS terms.
RR retains relative2PN plus the paper's local leading-tail approximation.
Published force polynomials carry nu^2; divide by nu for our mu-scaled
momenta/work. Newtonian Peters and circular-flux tests fix this convention.
No private implementation or NR-calibrated Hamiltonian is used.
"""
from math import sqrt,isfinite,pi
import numpy as np
from scipy.optimize import brentq
from .analytic_eob import AnalyticEOB
from .radial_core import FrozenOrbit,DomainError,AccuracyError


def public_forces_2pn(r,pr,j,nu,chi1,chi2,delta):
    """Eqs40,43,45,47-48,50-54, converted to mu-scaled canonical forces."""
    v=pr*pr+j*j/r**2;w=pr*pr;n=nu
    fj=(8*n*j/(15*r**3))*(10*v-39*w-22/r)
    fr=-(16*n*pr/(15*r**3))*(-5*v+12*w+11/r)
    common=-6*(4*n+93)*v*v+180*(7*n-5)*w*w-(484*n+3833)*v/r+(1684*n+6213)/r**2
    fj+=n*j/(105*r**3)*(common+18*(4*n+93)*w*v+(9780*n+19198)*w/r)
    fr+=n*pr/(105*r**3)*(common-198*(6*n-13)*w*v+4*(691*n+3958)*w/r)
    fj+=pi*n/r**2*(334*v*v/(15*r)-718*v/(15*r*r)-334*v*w/(5*r)-308*w/(15*r*r)+49*w**3/225)
    fr+=pi*n*j*pr/r**4*(334*v/(15*r)-w*w/18-7034*w/(45*r)-718/(15*r*r))
    # Spin-independent relative2PN terms; distinct radial/azimuthal coefficients.
    common2=(5276*n*n/315+109609*n/630-9175/378)*v**3
    common2+=(1519/54-9964*n*n/315-100847*n/630)*v*v/r
    common2+=(3512*n*n/315-4234*n/45+3355/21)*v/r**2
    common2+=(52/3-152*n*n/9-310*n/9)*w**3
    common2+=(8*n*n/15+9728*n/315-190244/2835)/r**3
    angular=common2+(9175/126-5276*n*n/105-109609*n/210)*v*v*w
    angular+=(104296/945-15121*n*n/105-254732*n/315)*w*v/r
    angular+=(-185*n*n/63+2171*n/63-269/63)*w*w*v
    angular+=(4112/35-278*n*n/45+344*n/35)*w*w/r
    angular+=(-1604*n*n/315-2054*n/7-8803/21)*w/r**2
    radial=common2+(9713/126-2129*n*n/45-350537*n/630)*v*v*w
    radial+=(26459/945-1745*n*n/63-449227*n/315)*w*v/r
    radial+=(293*n*n/21+1447*n/21-1361/63)*w*w*v
    radial+=(41612/315-1894*n*n/45+28648*n/315)*w*w/r
    radial+=(34528*n*n/945-481624*n/945-18761/45)*w/r**2
    fj+=n*j/r**3*angular;fr+=n*pr/r**3*radial
    # 1<->2 exchanges both spin and signed mass difference.
    for chi,d in [(chi1,delta),(chi2,-delta)]:
        fj+=2*n*chi/(15*r**3)*(v*v*(22*n-9*d-9)+5*(3+3*d+16*n)*w*w
              +(66*n-23*d-23)*w/r+(179+179*d-146*n)*v/r+6*(9+9*d-22*n)*v*w)
        fr+=2*n*pr*j*chi/(15*r**5)*((179*d-146*n+179)/r+(22*n-9*d-9)*v-5*(3+3*d+16*n)*w)
        x=(1+d)/2
        commonss=(400*n*d-209*d-400*n*n+818*n-209)*v
        commonss+=(355*d-704*n*d+704*n*n-1414*n+355)/r
        fj+=n*j*chi*chi/(30*r**5)*(24*x*x*(15*v-90*w-49/r)
            +(361*d-632*n*d+632*n*n-1354*n+361)*w+commonss)
        fr+=n*pr*chi*chi/(30*r**5)*(24*x*x*(15*v-55*w-49/r)
            +(301*d-512*n*d+512*n*n-1114*n+301)*w+commonss)
    fj+=n*j/(30*r**5)*2*n*chi1*chi2*((378-400*n)*v+(632*n-2250)*w+(704*n-1182)/r)
    fr+=n*pr/(30*r**5)*2*n*chi1*chi2*((378-400*n)*v+(512*n-1410)*w+(704*n-1182)/r)
    return fr,fj


class AlignedSpinEOB(AnalyticEOB):
    def __init__(self,eta,chi1,chi2,delta=None,strength=1.):
        super().__init__(eta)
        if not all(isfinite(v) for v in [chi1,chi2,strength]) or max(abs(chi1),abs(chi2))>=1 or strength<0:
            raise DomainError('finite sub-Kerr aligned spins and nonnegative RR strength required')
        self.delta=sqrt(1-4*eta) if delta is None else delta
        if not isfinite(self.delta) or abs(self.delta**2-(1-4*eta))>1e-12:
            raise DomainError('signed mass difference inconsistent with eta')
        self.chi1,self.chi2,self.strength=chi1,chi2,strength
        x1,x2=(1+self.delta)/2,(1-self.delta)/2
        self.so=.5*(chi1*(2+2*self.delta-eta)+chi2*(2-2*self.delta-eta))
        self.spin_square=(x1*x1*chi1+x2*x2*chi2)**2
        self.ss_constant=self.spin_square-(x1*chi1+x2*chi2)**2

    def spin_terms(self,r,p,j):
        a,ar,_=self.potentials(r)
        q=1+6*self.eta/r**2+2*self.eta*(26-3*self.eta)/r**3
        qr=-12*self.eta/r**3-6*self.eta*(26-3*self.eta)/r**4
        xi=a*sqrt(q);log_xi_r=ar/a+.5*qr/q
        k=self.spin_square/(2*r*r*xi*xi)
        potential=self.so*j/r**3+self.ss_constant/(2*r**3)-self.spin_square*j*j/(2*r**4)
        radial=-3*self.so*j/r**4-3*self.ss_constant/(2*r**4)+2*self.spin_square*j*j/r**5
        radial+=k*(-2/r-2*log_xi_r)*p*p
        omega=self.so/r**3-self.spin_square*j/r**4
        return potential,k,radial,omega

    def hamiltonian(self,r,p,j):
        v,k,_,_=self.spin_terms(r,p,j)
        return super().hamiltonian(r,p,j)+v+k*p*p

    def h_r(self,r,p,j):return super().h_r(r,p,j)+self.spin_terms(r,p,j)[2]
    def h_prstar(self,r,p,j):return super().h_prstar(r,p,j)+2*self.spin_terms(r,p,j)[1]*p
    def h_prstar_prstar(self,r,p,j):return super().h_prstar_prstar(r,p,j)+2*self.spin_terms(r,p,j)[1]
    def omega(self,r,p,j):return super().omega(r,p,j)+self.spin_terms(r,p,j)[3]

    def momentum_from_energy(self,r,energy,j,branch):
        if branch not in (-1,1):raise DomainError('explicit radial branch required')
        potential,k,_,_=self.spin_terms(r,0.,j)
        target=energy-potential
        he0=1+target+.5*self.eta*target*target
        a=self.potentials(r)[0];quartic=a*self.z3/r**2
        gap=he0*he0-a*(1+j*j/r**2)
        if gap<0:raise DomainError('forbidden spinning radial shell')
        upper=2*gap/(1+sqrt(1+4*quartic*gap))
        if k==0:return branch*sqrt(upper)
        def residual(w):
            dhe=(1+self.eta*target)*k*w-.5*self.eta*k*k*w*w
            return w+quartic*w*w-gap+dhe*(2*he0-dhe)
        w=upper/(1+2*he0*k*(1+self.eta*target))
        for _ in range(5):
            dhe=(1+self.eta*target)*k*w-.5*self.eta*k*k*w*w
            derivative=1+2*quartic*w+2*k*(1+self.eta*target-self.eta*k*w)*(he0-dhe)
            w-=residual(w)/derivative
        if not 0<=w<=upper or abs(residual(w))>1e-12*max(gap,1e-6):
            w=brentq(residual,0.,upper,xtol=1e-15,rtol=1e-12)
        return branch*sqrt(w)

    def forces(self,r,p,j):
        fr,fj=public_forces_2pn(r,p/self.xi(r),j,self.eta,self.chi1,self.chi2,self.delta)
        return self.strength*fr,self.strength*fj

    def fluxes(self,r,p,j):
        fr,fj=self.forces(r,p,j)
        return -(self.xi(r)*self.h_prstar(r,p,j)*fr+self.omega(r,p,j)*fj),-fj

    def rhs(self,t,state):
        r,p,phi,j=state;result=np.array(super().rhs(t,state));fr,fj=self.forces(r,p,j)
        result[1]+=self.xi(r)*fr;result[3]+=fj
        return result

    def bound_orbit(self,rp,ra):
        if not 0<rp<ra:raise DomainError('distinct ordered turning points required')
        def difference(j):return self.hamiltonian(rp,0.,j)-self.hamiltonian(ra,0.,j)
        guess=sqrt(2*rp*ra/(rp+ra))
        lo,hi=.4*guess,1.8*guess
        if difference(lo)*difference(hi)>=0:raise DomainError('no bracketed physical turning-root angular momentum')
        j=brentq(difference,lo,hi,xtol=1e-12,rtol=1e-12)
        energy=self.hamiltonian(rp,0.,j)
        if energy>=0:raise DomainError('selected spinning orbit is not bound')
        return FrozenOrbit(self,energy,j,rp,ra)
