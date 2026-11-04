"""Cached distance maps; native-prior and evaluation cosmologies stay separate."""
import numpy as np
from scipy.interpolate import PchipInterpolator


class CosmologyContext:
    def __init__(self, specification="Planck15", distance_max=10000., nodes=4097):
        from astropy import cosmology as ac
        from astropy import units as u
        if isinstance(specification, str):
            if not hasattr(ac, specification):
                raise ValueError("Unknown Astropy cosmology: " + specification)
            cosmology = getattr(ac, specification)
        elif isinstance(specification, dict):
            cosmology = ac.Cosmology.from_format(specification, format="mapping")
        else:
            cosmology = specification
        if not isinstance(cosmology, ac.Cosmology):
            raise ValueError("Expected an Astropy cosmology")
        if not np.isfinite(distance_max) or distance_max <= 0 or nodes < 100:
            raise ValueError("Invalid cosmology table domain/resolution")
        self.cosmology = cosmology
        self.specification = cosmology.to_format("mapping")
        # Astropy's YAML roundtrip preserves nonflat and neutrino parameters.
        self.serialized = cosmology.to_format("yaml")
        zmax = float(ac.z_at_value(cosmology.luminosity_distance,
                                   distance_max * (1 + 1e-8) * u.Mpc))
        # Resolve the dVc/dz ~ z^2 limit as well as cosmological distances.
        self.z = np.r_[0., np.geomspace(min(1e-9, zmax/1e6), zmax, nodes-1)]
        self.dl = cosmology.luminosity_distance(self.z).to_value(u.Mpc)
        if not np.all(np.diff(self.dl) > 0):
            raise ValueError("Distance map is not monotone on the requested domain")
        self._forward = PchipInterpolator(self.z, self.dl, extrapolate=False)
        self._inverse = PchipInterpolator(self.dl, self.z, extrapolate=False)
        self._volume = PchipInterpolator(self.z,
            cosmology.differential_comoving_volume(self.z).to_value(u.Mpc**3/u.sr),
            extrapolate=False)
        probes = (self.z[:-1]+self.z[1:])/2
        probes = probes[np.linspace(0, len(probes)-1, min(129, len(probes)), dtype=int)]
        exact = cosmology.luminosity_distance(probes).to_value(u.Mpc)
        if np.max(abs(self._forward(probes)-exact)/np.maximum(exact, 1e-12)) > 1e-5:
            raise ValueError("Cosmology interpolation is unresolved; increase nodes")

    @classmethod
    def from_yaml(cls, serialized, distance_max):
        from astropy.cosmology import Cosmology
        return cls(Cosmology.from_format(serialized, format="yaml"), distance_max)

    @classmethod
    def from_astropy(cls, cosmology, distance_max=10000.):
        return cls(cosmology, distance_max)

    def luminosity_distance(self, z):
        return self._forward(z)

    def redshift(self, distance):
        return self._inverse(distance)

    def distance_derivative(self, z):
        # Differentiate the actual distance map, including spatial curvature.
        return self._forward.derivative()(z)

    def distance_logprior(self, distance, kind):
        distance = np.asarray(distance, float)
        if kind == "Euclidean":
            return 2*np.log(distance)
        if kind == "uniform":
            return np.zeros_like(distance)
        if kind == "pseudo_cosmo":
            coefficients = [0.000387, 0.016732, 0.262462, 1.136740, 1.012306]
            return 2*np.log(distance)-np.log(np.polyval(coefficients, distance/1000.))
        if kind not in ("cosmo", "cosmo_sourceframe"):
            raise ValueError("Unsupported distance prior: " + kind)
        z = self.redshift(distance)
        result = np.log(self._volume(z))-np.log(self.distance_derivative(z))
        if kind == "cosmo_sourceframe":
            result -= np.log1p(z)
        return result
