"""Independent, nonspinning 3PN EOB conservative research model.

Public equations only: Buonanno et al., PRD76,104049 (2007), SecII,
doi:10.1103/PhysRevD.76.104049; Yunes et al., PRL104,091102 (2010),
Eqs(1)-(2), doi:10.1103/PhysRevLett.104.091102. A=Pade(1,3), D=Pade(0,3).
The tortoise-momentum quartic form is a specified resummation, equivalent
through the retained PN order. No calibration, spins, RR, or waveform model.
No pyseobnr code/imports; this is not SEOBNRv6EHM.

Units match radial_core except the rest energy is subtracted:
h=(H_real-M)/mu, tau=t/M, r=R/M, p=p_rstar/mu, j=J/(mu*M).
"""
from math import isfinite, pi, sqrt
import numpy as np
from .radial_core import DomainError, FrozenOrbit


class AnalyticEOB:
    def __init__(self, eta):
        if not isfinite(eta) or not 0 <= eta <= .25:
            raise DomainError('require 0<=eta<=1/4; zero denotes the test-mass limit')
        self.eta = eta
        self.z3 = 2*eta*(4-3*eta)
        c = np.array([1., -2., 0., 2*eta, (94/3-41*pi*pi/32)*eta])
        matrix = np.array([[c[1], c[0], 0], [c[2], c[1], c[0]], [c[3], c[2], c[1]]])
        self.denominator = np.r_[1., np.linalg.solve(matrix, -c[2:5])]
        self.numerator = np.array([1., c[1]+self.denominator[1]])
        self.horizon = -self.numerator[1]

    def potentials(self, r):
        if not isfinite(r) or r <= max(0, self.horizon):
            raise DomainError('radius outside the positive exterior A domain')
        u = 1/r
        q = np.polynomial.polynomial.polyval(u, self.denominator)
        n = 1+self.numerator[1]*u
        dq = np.polynomial.polynomial.polyval(u, np.arange(1, 4)*self.denominator[1:])
        if q <= 0:
            raise DomainError('Pade pole or wrong exterior branch')
        a = n/q
        ar = -(self.numerator[1]*q-n*dq)/(q*q*r*r)
        d = 1/(1+6*self.eta*u*u+2*self.eta*(26-3*self.eta)*u**3)
        if a <= 0 or d <= 0:
            raise DomainError('nonpositive metric potential')
        return a, ar, d

    def xi(self, r):
        a, _, d = self.potentials(r)
        return a/sqrt(d)

    def _terms(self, r, p, j):
        if not all(isfinite(v) for v in (p, j)):
            raise DomainError('nonfinite momentum')
        a, ar, _ = self.potentials(r)
        k = self.z3/r**2
        b = 1+j*j/r**2+k*p**4
        he = sqrt(p*p+a*b)
        real = sqrt(1+2*self.eta*(he-1))
        sp = 2*p+4*a*k*p**3
        sr = ar*b-2*a*(j*j+self.z3*p**4)/r**3
        return a, he, real, sp, sr, k

    def hamiltonian(self, r, p, j):
        _, he, real, *_ = self._terms(r, p, j)
        return 2*(he-1)/(real+1)

    def h_prstar(self, r, p, j):
        _, he, real, sp, _, _ = self._terms(r, p, j)
        return sp/(2*he*real)

    def h_prstar_prstar(self, r, p, j):
        a, he, real, sp, _, k = self._terms(r, p, j)
        hep = sp/(2*he)
        hepp = (2+12*a*k*p*p)/(2*he)-sp*sp/(4*he**3)
        return hepp/real-self.eta*hep*hep/real**3

    def h_r(self, r, p, j):
        _, he, real, _, sr, _ = self._terms(r, p, j)
        return sr/(2*he*real)

    def omega(self, r, p, j):
        a, he, real, *_ = self._terms(r, p, j)
        return a*j/(r*r*he*real)

    def fluxes(self, r, p, j):
        # An exactly conservative model, not a prescription for radiative losses.
        return 0., 0.

    def momentum_from_energy(self, r, energy, j, branch):
        if branch not in (-1, 1):
            raise DomainError('radial branch must be explicit')
        a, _, _ = self.potentials(r)
        he = 1+energy+.5*self.eta*energy*energy
        if he <= 0 or 1+self.eta*energy <= 0:
            raise DomainError('nonphysical real-to-effective energy map')
        gap = he*he-a*(1+j*j/r**2)
        if gap < 0:
            raise DomainError('forbidden radial state')
        k = a*self.z3/r**2
        w = 2*gap/(1+sqrt(1+4*k*gap))
        return branch*sqrt(w)

    def rhs(self, time, state):
        r, p, phi, j = state
        xi = self.xi(r)
        return [xi*self.h_prstar(r, p, j), -xi*self.h_r(r, p, j),
                self.omega(r, p, j), 0.]


def bound_orbit(eta, r_peri, r_apo, options=None):
    """Specify actual turning radii, not a backend-specific native eccentricity."""
    if not 0 < r_peri < r_apo:
        raise DomainError('require distinct ordered turning radii')
    adapter = AnalyticEOB(eta)
    ap = adapter.potentials(r_peri)[0]
    aa = adapter.potentials(r_apo)[0]
    denominator = ap/r_peri**2-aa/r_apo**2
    if denominator <= 0 or aa <= ap:
        raise DomainError('no positive angular momentum for the selected roots')
    j = sqrt((aa-ap)/denominator)
    energy = adapter.hamiltonian(r_peri, 0., j)
    if energy >= 0:
        raise DomainError('selected orbit is not bound')
    return FrozenOrbit(adapter, energy, j, r_peri, r_apo, options)
