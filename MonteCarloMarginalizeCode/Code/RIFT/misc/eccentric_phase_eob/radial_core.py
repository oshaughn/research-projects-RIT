"""Research-only frozen radial clocks and leading adiabatic loss transport.

No native waveform parameters, samples, or priors are modified. The inverse
below is ONLY at fixed conservative (E,j); no unit Jacobian is asserted.

Adapter units: r/M, p_r*/mu, j=J_orb/(mu M), tau=t/M, H=H_real/mu
(an additive rest-energy constant is harmless). Thus dr/dtau=xi H_p*.
fluxes returns (-dH/dtau,-dj/dtau) from one consistent mechanical/balance
prescription, including its Schott convention; instantaneous signs may vary.

References: Buonanno--Damour gr-qc/9811091; Glampedakis--Kennefick
gr-qc/0203086; O'Shaughnessy gr-qc/0211023, II A--C and Appendix A;
Bini--Damour 1210.2834. In particular a whirl frequency does not determine
the final barrier gap or transition duration. All accuracy claims here are
quadrature/ODE tolerances, not a bound on the adiabatic modeling error.
"""

from dataclasses import dataclass
from math import acosh, asinh, cos, exp, expm1, isfinite, log, log1p, pi, sin, sqrt
from typing import Callable, Optional, Protocol, Tuple

import numpy as np
from scipy.integrate import quad, quad_vec, solve_ivp
from scipy.optimize import brentq
from scipy.special import hyp2f1

TAU = 2.0 * pi


class DomainError(ValueError):
    """The requested orbit/chart does not have the required topology."""


class AccuracyError(RuntimeError):
    """Numerical conditioning or error estimates fail the requested contract."""


class AdiabaticError(RuntimeError):
    """A supplied physical validity check rejects frozen/averaged transport."""


class PlanarAdapter(Protocol):
    def hamiltonian(self, r: float, prstar: float, j: float) -> float: ...
    def xi(self, r: float) -> float: ...
    def h_prstar(self, r: float, prstar: float, j: float) -> float: ...
    def h_prstar_prstar(self, r: float, prstar: float, j: float) -> float: ...
    def h_r(self, r: float, prstar: float, j: float) -> float: ...
    def omega(self, r: float, prstar: float, j: float) -> float: ...
    def fluxes(self, r: float, prstar: float, j: float) -> Tuple[float, float]: ...


@dataclass(frozen=True)
class QuadratureOptions:
    rtol: float = 2e-8
    atol: float = 2e-10
    root_rtol: float = 2e-12
    root_atol: float = 2e-13
    # Use the analytic limiting density in this tiny endpoint interval.
    # Its asymptotic truncation is not included in QUADPACK's error estimate;
    # convergence under decreasing endpoint_u is a required numerical check.
    endpoint_u: float = 2e-6
    limit: int = 400
    max_momentum: float = 1e4


@dataclass(frozen=True)
class CycleIntegrals:
    period: float
    azimuth: float
    energy_loss: float
    angular_momentum_loss: float
    error_estimates: Tuple[float, float, float, float]
    outgoing_time: float

    @property
    def omega_r(self):
        return TAU / self.period

    @property
    def omega_phi(self):
        return self.azimuth / self.period


@dataclass(frozen=True)
class RadialState:
    r: float
    prstar: float
    branch: int  # +1 outgoing, -1 incoming; deterministic at endpoints
    phase: float


def _branch(branch):
    if branch not in (-1, 1):
        raise DomainError("radial branch must be +1 (outgoing) or -1 (incoming)")
    return branch


def _finite(*values):
    if not all(isfinite(float(v)) for v in values):
        raise DomainError("nonfinite orbit/model input")


class FrozenOrbit:
    """A declared pair of simple outer turning points and a fixed energy shell.

    The caller must choose the physical turning roots and backend/version.
    H_p*(r,0,j)=0 and positive radial kinetic curvature are required. Both
    branches are integrated independently: time/flux symmetry is not assumed.
    A generic momentum solve assumes one monotone physical root per sign;
    adapters with more complicated momentum branches must provide
    momentum_from_energy(r,E,j,branch) with explicit branch selection.
    """

    def __init__(self, adapter: PlanarAdapter, energy: float, angular_momentum: float,
                 r_peri: float, r_apo: float,
                 options: Optional[QuadratureOptions] = None):
        self.adapter, self.energy, self.j = adapter, float(energy), float(angular_momentum)
        self.r_peri, self.r_apo = float(r_peri), float(r_apo)
        self.options = options or QuadratureOptions()
        _finite(energy, angular_momentum, r_peri, r_apo)
        if not 0 < r_peri < r_apo:
            raise DomainError("a noncircular bound orbit needs 0 < r_peri < r_apo")
        self.width = self.r_apo - self.r_peri
        if self.width <= 128*np.finfo(float).eps*max(1.0, self.r_apo):
            raise AccuracyError("turning points are numerically coincident")
        self._end_density = []
        for r, sign in ((self.r_peri, -1), (self.r_apo, 1)):
            h = adapter.hamiltonian(r, 0.0, self.j)
            hp = adapter.h_prstar(r, 0.0, self.j)
            hr = adapter.h_r(r, 0.0, self.j)
            hpp = adapter.h_prstar_prstar(r, 0.0, self.j)
            xi = adapter.xi(r)
            _finite(h, hp, hr, hpp, xi)
            tol = self.options.root_atol + self.options.root_rtol*max(1.0, abs(h), abs(energy))
            if abs(h-energy) > tol or abs(hp) > tol:
                raise DomainError("declared endpoint is not a zero-radial-velocity turning point")
            if xi <= 0 or hpp <= 0 or sign*hr <= 0:
                raise DomainError("non-simple or wrong-branch turning point; separatrix/circular chart rejected")
            slope = 2*xi*xi*hpp*sign*hr
            if slope <= 128*np.finfo(float).eps/max(1.0, self.r_apo):
                raise AccuracyError("turning-root slope is numerically unresolved")
            self._end_density.append(2*sqrt(self.width/slope))
        self._cycle = None

    def radius(self, u):
        if not 0 <= u <= pi/2:
            raise DomainError("regularized radial coordinate outside [0,pi/2]")
        # Use distance to the closer endpoint to avoid loss of precision there.
        if u <= pi/4:
            return self.r_peri + self.width*sin(u)**2
        return self.r_apo - self.width*cos(u)**2

    def momentum(self, r, branch):
        _branch(branch)
        if not self.r_peri <= r <= self.r_apo:
            raise DomainError("radius outside the selected radial interval")
        if r in (self.r_peri, self.r_apo):
            return 0.0
        direct = getattr(self.adapter, "momentum_from_energy", None)
        if direct is not None:
            p = float(direct(r, self.energy, self.j, branch))
        else:
            h0 = self.adapter.hamiltonian(r, 0.0, self.j)
            gap = self.energy-h0
            if gap <= 0:
                raise AccuracyError("energy shell is forbidden or unresolved inside the selected roots")
            hpp = self.adapter.h_prstar_prstar(r, 0.0, self.j)
            if hpp <= 0:
                raise DomainError("nonpositive radial kinetic curvature")
            hi = max(2*sqrt(2*gap/hpp), 1e-7)
            def residual(q):
                return self.adapter.hamiltonian(r, branch*q, self.j)-self.energy
            while residual(hi) < 0 and hi < self.options.max_momentum:
                hi *= 2
            if hi > self.options.max_momentum or residual(hi) < 0:
                raise DomainError("no bracketed physical radial momentum root")
            p = branch*brentq(residual, 0.0, hi, xtol=self.options.root_atol,
                              rtol=self.options.root_rtol)
        _finite(p)
        residual = abs(self.adapter.hamiltonian(r, p, self.j)-self.energy)
        tolerance = (self.options.root_atol
                     +self.options.root_rtol*max(1.0, abs(self.energy)))
        if residual > tolerance:
            raise AccuracyError("momentum solver did not recover the declared energy shell")
        v = self.adapter.xi(r)*self.adapter.h_prstar(r, p, self.j)
        if branch*p <= 0 or branch*v <= 0:
            raise DomainError("momentum root has the wrong radial branch")
        return p

    def _time_density(self, u, branch):
        _branch(branch)
        r = self.radius(u)
        if u <= self.options.endpoint_u:
            return self._end_density[0], r, 0.0
        if pi/2-u <= self.options.endpoint_u:
            return self._end_density[1], r, 0.0
        p = self.momentum(r, branch)
        v = self.adapter.xi(r)*self.adapter.h_prstar(r, p, self.j)
        density = 2*self.width*sin(u)*cos(u)/abs(v)
        _finite(density)
        if density <= 0:
            raise DomainError("nonmonotone conservative radial clock")
        return density, r, p

    def _integrand(self, u, branch):
        density, r, p = self._time_density(u, branch)
        omega = self.adapter.omega(r, p, self.j)
        fe, fj = self.adapter.fluxes(r, p, self.j)
        values = density*np.array([1.0, omega, fe, fj], dtype=float)
        if not np.all(np.isfinite(values)):
            raise DomainError("nonfinite frequency/flux on selected energy shell")
        return values

    def cycle(self):
        if self._cycle is not None:
            return self._cycle
        total, errors, times = np.zeros(4), np.zeros(4), []
        for branch in (1, -1):
            samples = np.array([self._integrand(u, branch)
                                for u in (0.0, pi/8, pi/4, 3*pi/8, pi/2)])
            scale = np.max(np.abs(samples), axis=0)
            scale[scale == 0] = 1.0
            value, error, info = quad_vec(lambda u: self._integrand(u, branch)/scale,
                0.0, pi/2, epsabs=self.options.atol, epsrel=self.options.rtol,
                norm="max", limit=self.options.limit, full_output=True)
            if not info.success:
                raise AccuracyError("radial cycle quadrature failed: " + str(getattr(info, "message", info.status)))
            value = value*scale
            total += value
            errors += float(error)*scale
            times.append(float(value[0]))
        if total[0] <= 0:
            raise DomainError("nonpositive radial period")
        self._cycle = CycleIntegrals(*map(float, total), tuple(map(float, errors)), times[0])
        return self._cycle

    def _partial_time(self, lo, hi, branch):
        if hi == lo:
            return 0.0
        if lo == 0.0 and hi == pi/2:
            cyc = self.cycle()
            return cyc.outgoing_time if branch == 1 else cyc.period-cyc.outgoing_time
        value, error = quad(lambda u: self._time_density(u, branch)[0], lo, hi,
                            epsabs=self.options.atol, epsrel=self.options.rtol,
                            limit=self.options.limit)
        if error > 10*(self.options.atol+self.options.rtol*abs(value)):
            raise AccuracyError("partial radial time quadrature did not meet tolerance")
        return value

    def phase(self, r, branch, *, wrapped=True):
        """Uniform radial time angle at fixed E,j, with branch retained."""
        _branch(branch)
        if not self.r_peri <= r <= self.r_apo:
            raise DomainError("radius outside selected bound orbit")
        u = float(np.arcsin(sqrt((r-self.r_peri)/self.width)))
        cyc = self.cycle()
        elapsed = (self._partial_time(0.0, u, 1) if branch == 1 else
                   cyc.outgoing_time+self._partial_time(u, pi/2, -1))
        q = TAU*elapsed/cyc.period
        return q % TAU if wrapped else q

    def state_at_phase(self, angle):
        """Inverse on this frozen orbit, periodic in angle; not a dissipative inverse."""
        _finite(angle)
        q = float(angle) % TAU
        cyc = self.cycle()
        t = q/TAU*cyc.period
        if q == 0:
            return RadialState(self.r_peri, 0.0, 1, q)
        if abs(t-cyc.outgoing_time) <= self.options.root_atol:
            return RadialState(self.r_apo, 0.0, 1, q)
        if t <= cyc.outgoing_time:
            branch = 1
            residual = lambda u: self._partial_time(0.0, u, 1)-t
        else:
            branch = -1
            residual = lambda u: cyc.outgoing_time+self._partial_time(u, pi/2, -1)-t
        u = brentq(residual, 0.0, pi/2, xtol=self.options.root_atol,
                   rtol=self.options.root_rtol)
        r = self.radius(u)
        return RadialState(r, self.momentum(r, branch), branch, q)

    def time_to_next_periapsis(self, r, branch):
        """Future event time, including the chosen radial branch (zero at periapsis)."""
        q = self.phase(r, branch)
        return 0.0 if q == 0 else self.cycle().period*(1-q/TAU)


@dataclass(frozen=True)
class PhaseAdvance:
    advance: float
    phase_unwrapped: float
    phase_wrapped: float
    angular_momentum_final: float
    evaluated_states: int


def loss_ratio_advance(energy0, angular_momentum0, energy1,
                       cycle_provider: Callable[[float, float], CycleIntegrals], *,
                       validity_check: Callable[[float, float, CycleIntegrals], None],
                       phase0=0.0, rtol=1e-7, atol=1e-9, max_energy_step=None):
    """Leading averaged dq/dE=-2pi/DE and dj/dE=DJ/DE, forward only.

    validity_check is mandatory and must raise AdiabaticError when its chosen
    physical domain fails. Numerical ODE accuracy cannot certify averaging.
    Only states visited by the solver are checked; this is not a proof that
    a narrow intervening invalid region cannot exist. The endpoint energy is
    a declared slow-state section, not a periapsis hitting event.
    """
    _finite(energy0, angular_momentum0, energy1, phase0)
    if not callable(validity_check):
        raise ValueError("an explicit adiabatic validity check is required")
    if energy1 > energy0:
        raise DomainError("loss-ratio transport requires a forward decreasing-energy section")
    if energy1 == energy0:
        return PhaseAdvance(0.0, phase0, phase0 % TAU, angular_momentum0, 0)
    count = 0
    def rhs(energy, state):
        nonlocal count
        cyc = cycle_provider(float(energy), float(state[0]))
        count += 1
        validity_check(float(energy), float(state[0]), cyc)
        if not isfinite(cyc.energy_loss) or cyc.energy_loss <= 0:
            raise AdiabaticError("cycle energy loss must be positive for energy-parametrized transport")
        _finite(cyc.angular_momentum_loss)
        return [cyc.angular_momentum_loss/cyc.energy_loss, -TAU/cyc.energy_loss]
    step = (energy0-energy1)/16 if max_energy_step is None else max_energy_step
    if step <= 0:
        raise ValueError("max_energy_step must be positive")
    sol = solve_ivp(rhs, (energy0, energy1), [angular_momentum0, 0.0],
                    rtol=rtol, atol=atol, max_step=step)
    if not sol.success:
        raise AccuracyError("loss-ratio transport failed: " + sol.message)
    advance = float(sol.y[1, -1])
    return PhaseAdvance(advance, phase0+advance, (phase0+advance) % TAU,
                        float(sol.y[0, -1]), count)


@dataclass(frozen=True)
class Saddle:
    """Local quadratic saddle. Gap and initial branch are never inferred from omega."""
    a: float
    b: float
    omega: float

    def __post_init__(self):
        _finite(self.a, self.b, self.omega)
        if self.a <= 0 or self.b <= 0:
            raise DomainError("nondegenerate hyperbolic saddle required; ISCO/fold excluded")

    @property
    def rate(self):
        return sqrt(self.a*self.b)

    def gap_scale(self, detuning):
        _finite(detuning)
        if detuning == 0:
            raise DomainError("zero gap has no finite conservative encounter duration")
        return sqrt(2*abs(detuning)/self.b)

    def duration(self, detuning, matching_radius):
        """Full +Rm->+Rm reflected or +Rm->-Rm inward crossing duration."""
        _finite(matching_radius)
        scale = self.gap_scale(detuning)
        if matching_radius <= 0:
            raise DomainError("matching radius must be positive")
        x = matching_radius/scale
        if detuning < 0:
            if x < 1:
                raise DomainError("returning turning point lies outside the local section")
            return 2*acosh(x)/self.rate
        return 2*asinh(x)/self.rate

    def time_to_exit(self, detuning, radial_offset, radial_branch, matching_radius):
        """Remaining duration from a specified local position and velocity branch.

        For a reflected exterior orbit the exit is +Rm; for a crossing orbit
        it is radial_branch*Rm. This retains the phase/last-gap sensitivity
        emphasized by O'Shaughnessy (2003), rather than assigning a unique
        duration to a leading circular-whirl frequency.
        """
        _finite(radial_offset, matching_radius)
        _branch(radial_branch)
        scale = self.gap_scale(detuning)
        self.duration(detuning, matching_radius)  # validates section topology
        if abs(radial_offset) > matching_radius:
            raise DomainError("initial point lies outside local matching section")
        if detuning < 0:
            if radial_offset < scale:
                raise DomainError("initial point is not on the exterior returning branch")
            return (acosh(matching_radius/scale)
                    -radial_branch*acosh(radial_offset/scale))/self.rate
        return (asinh(matching_radius/scale)
                -radial_branch*asinh(radial_offset/scale))/self.rate


@dataclass(frozen=True)
class SaddleValidity:
    ratios: dict
    failures: Tuple[str, ...]

    @property
    def valid(self):
        return not self.failures

    def require(self):
        if self.failures:
            raise AdiabaticError("; ".join(self.failures))


def assess_saddle(saddle, detuning, *, maximum_normal_change, saddle_speed,
                  maximum_saddle_displacement, radial_force,
                  well_depth=None, cycle_normal_change=None,
                  relative_coefficient_drift=0.0, threshold=0.1):
    """Sufficient local diagnostics, with an explicit engineering tolerance.

    maximum_normal_change is max_t |Delta(t)-Delta(entry)|, NOT a net loss
    or an instantaneous derivative. saddle_speed is the maximum magnitude
    of r_u'(j)*jdot in tau time during the selected encounter, not one sample.
    maximum_saddle_displacement=max_t |r_u(t)-r_u(entry)| over that ENTIRE
    encounter is mandatory: the conservative frozen-saddle contract also
    requires accumulated motion to stay small compared with the gap scale.
    This is deliberately stronger than a local moving-saddle approximation.
    radial_force is the force on canonical p_r, NOT unconverted p_r*.
    Supplying cycle_normal_change also requires the conservative well depth
    (barrier minus stable minimum), following the flat-potential guard in
    O'Shaughnessy Eq. (15). Omitting both gives only a LOCAL assessment.
    """
    scale = saddle.gap_scale(detuning)
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    _finite(maximum_normal_change, saddle_speed, maximum_saddle_displacement,
            radial_force, relative_coefficient_drift)
    ratios = {
        "normal_gap_change": abs(maximum_normal_change/detuning),
        "saddle_motion": abs(saddle_speed)/(saddle.rate*scale),
        "accumulated_saddle_motion": abs(maximum_saddle_displacement)/scale,
        "radial_forcing": abs(radial_force)/(saddle.b*scale),
        "coefficient_drift": abs(relative_coefficient_drift),
    }
    if (well_depth is None) != (cycle_normal_change is None):
        raise ValueError("whole-cycle validity requires both well_depth and cycle_normal_change")
    if well_depth is not None:
        _finite(well_depth, cycle_normal_change)
        if well_depth <= 0:
            raise DomainError("positive well depth required; flat/circular transition excluded")
        ratios["well_depth_change"] = abs(cycle_normal_change)/well_depth
    failures = tuple(key + " exceeds tolerance" for key, ratio in ratios.items()
                     if ratio >= threshold)
    return SaddleValidity(ratios, failures)


def peters_phase_advance(e0, ea, p0_over_M, eta):
    """Full-e leading quadrupole mean-anomaly advance between given e endpoints.

    p0 is the Newtonian semilatus rectum/M. This stable form analytically
    cancels the divergent Peters invariant as e0 tends to zero. Exactly
    circular orbits need mean-frequency endpoints instead.
    """
    _finite(e0, ea, p0_over_M, eta)
    if not 0 <= ea <= e0 < 1 or e0 == 0 or p0_over_M <= 0 or not 0 < eta <= .25:
        raise DomainError("require 0<=ea<=e0<1, e0>0, p0/M>0 and 0<eta<=1/4")
    if ea == e0:
        return 0.0
    c = 121/304
    def hyper(e):
        return float(hyp2f1(124/2299, 15/19, 34/19, -c*e*e))
    bracket = hyper(e0)-(ea/e0)**(30/19)*hyper(ea)
    return p0_over_M**2.5*bracket/(32*eta*(1+c*e0*e0)**(2175/2299))


def peters_phase_from_mean_frequency(x0, xa, e0, eta):
    """Control using x=(M*n)^(2/3), i.e. Newtonian MEAN frequency.

    Returns (advance, final_e). A pericenter-frequency anchor is a different
    convention and must not be inserted here under the same name.
    """
    _finite(x0, xa, e0, eta)
    if not 0 < x0 <= xa or not 0 <= e0 < 1 or not 0 < eta <= .25:
        raise DomainError("require positive forward x endpoints, 0<=e0<1, 0<eta<=1/4")
    if xa == x0:
        return 0.0, e0
    log_ratio = log(xa/x0)
    if e0 == 0:
        return x0**-2.5*(-expm1(-2.5*log_ratio))/(32*eta), 0.0
    c, b, le0 = 121/304, 870/2299, log(e0)
    def equation(le):
        e = exp(le)
        return (log1p(-e*e)-log1p(-e0*e0)+(12/19)*(le0-le)
                +b*(log1p(c*e0*e0)-log1p(c*e*e))-log_ratio)
    lo = le0-max(2.0, (19/12)*log_ratio+2.0)
    while equation(lo) < 0:
        lo -= 2.0
    lea = brentq(equation, lo, le0, xtol=2e-13, rtol=2e-13)
    ea = exp(lea)
    return peters_phase_advance(e0, ea, (1-e0*e0)/x0, eta), ea


class SchwarzschildAdapter:
    """Exact geodesic control, NOT a finite-mass waveform or flux model.

    Default losses are zero. An optional explicit loss callback is useful
    for mathematics tests; it must not be described as a GR flux model.
    """
    def __init__(self, loss_function=None, factored_orbit=None):
        self.loss_function = loss_function
        self.factored_orbit = factored_orbit

    def xi(self, r):
        return 1-2/r

    def _v(self, r, j):
        return self.xi(r)*(1+j*j/r**2)

    def hamiltonian(self, r, prstar, j):
        return sqrt(prstar*prstar+self._v(r, j))

    def h_prstar(self, r, prstar, j):
        return prstar/self.hamiltonian(r, prstar, j)

    def h_prstar_prstar(self, r, prstar, j):
        return self._v(r, j)/self.hamiltonian(r, prstar, j)**3

    def h_r(self, r, prstar, j):
        vp = 2/r**2-2*j*j/r**3+6*j*j/r**4
        return vp/(2*self.hamiltonian(r, prstar, j))

    def omega(self, r, prstar, j):
        return self.xi(r)*j/(r*r*self.hamiltonian(r, prstar, j))

    def fluxes(self, r, prstar, j):
        return (0.0, 0.0) if self.loss_function is None else self.loss_function(r, prstar, j)

    def momentum_from_energy(self, r, energy, j, branch):
        if self.factored_orbit is not None:
            e_ref, j_ref, rp, ra, r3 = self.factored_orbit
            if energy != e_ref or j != j_ref:
                raise DomainError("factored Schwarzschild adapter belongs to another energy shell")
            p2 = (1-energy*energy)*(ra-r)*(r-rp)*(r-r3)/r**3
        else:
            p2 = energy*energy-self._v(r, j)
        if p2 < 0:
            raise DomainError("forbidden Schwarzschild radial state")
        return _branch(branch)*sqrt(p2)


def schwarzschild_orbit(p, eccentricity, *, loss_function=None, options=None):
    """Stable bound geodesic using Darwin p,e. The circular torus is excluded."""
    e = eccentricity
    _finite(p, e)
    if not 0 < e < 1 or p <= 6+2*e:
        raise DomainError("require 0<e<1 and p>6+2e; circular/separatrix/plunge excluded")
    energy = sqrt(((p-2)**2-4*e*e)/(p*(p-3-e*e)))
    j = p/sqrt(p-3-e*e)
    rp, ra, r3 = p/(1+e), p/(1-e), 2*p/(p-4)
    adapter = SchwarzschildAdapter(loss_function, (energy, j, rp, ra, r3))
    return FrozenOrbit(adapter, energy, j, rp, ra, options)


def schwarzschild_saddle(radius):
    """Coordinate-time hyperbolic coefficients at a Schwarzschild circular orbit."""
    if not 3 < radius < 6:
        raise DomainError("Schwarzschild timelike unstable circle requires 3<r/M<6")
    r = radius
    e = (r-2)/sqrt(r*(r-3))
    a = (1-2/r)**2/e
    b = (6-r)/(e*r**3*(r-3))
    return Saddle(a, b, r**-1.5)
