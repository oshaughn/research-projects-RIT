"""Independent leading (2.5PN) minimal-gauge radiation reaction.

Bini & Damour PRD86,124012 (2012), Eqs3.58,3.66,3.68. Canonical
p_r is p_rstar/xi, NOT p_rstar. Units r/M,t/M,H/mu,J/(mu*M).
Leading-PN balance includes E_S=16 eta p_r j^2/(5r^4), J_S=0.
Combined with a resummed 3PN H, balance with the leading radiative flux
holds to the retained dissipative order only. The EXACT mechanical work
of these forces is used consistently for orbit losses and direct dynamics.
No private code, fitted parameters, spin terms or v6 implementation.
"""
import numpy as np
from .analytic_eob import AnalyticEOB


def minimal_forces(r, pr, j, eta):
    transverse = j*j/r**2
    p2 = pr*pr+transverse
    fr = (8/15)*eta*pr/r**3*(21*transverse-1/r)
    fj = -(8/5)*eta*j/r**3*(2*p2-3*pr*pr+2/r)
    return fr, fj


def schott_energy(r, pr, j, eta):
    return (16/5)*eta*pr*j*j/r**4


def schott_gradient(r, pr, j, eta):
    c = (16/5)*eta
    return np.array([-4*c*pr*j*j/r**5, c*j*j/r**4, 2*c*pr*j/r**4])


def radiative_fluxes_lo(r, pr, j, eta):
    p2 = pr*pr+j*j/r**2
    fe = (8/15)*eta/r**4*(12*p2-11*pr*pr)
    fj = (8/5)*eta*j/r**3*(2*p2-3*pr*pr+2/r)
    return fe, fj


class RadiatingEOB(AnalyticEOB):
    def __init__(self, eta, strength=1.):
        super().__init__(eta)
        if not np.isfinite(strength) or strength < 0:
            raise ValueError('RR strength must be finite and nonnegative')
        self.strength = strength

    def forces(self, r, pstar, j):
        fr, fj = minimal_forces(r, pstar/self.xi(r), j, self.eta)
        return self.strength*fr, self.strength*fj

    def fluxes(self, r, pstar, j):
        fr, fj = self.forces(r,pstar,j)
        dr = self.xi(r)*self.h_prstar(r,pstar,j)
        return -(dr*fr+self.omega(r,pstar,j)*fj), -fj

    def rhs(self, time, state):
        r,pstar,phi,j = state
        conservative = np.array(super().rhs(time,state))
        fr,fj = self.forces(r,pstar,j)
        conservative[1] += self.xi(r)*fr
        conservative[3] += fj
        return conservative
