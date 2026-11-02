"""Raw distance-export evaluation, separate from posterior likelihood shapes.

Predictors implement ``predict(rows)``. The core predicts raw log likelihood on
the declared nine-coordinate chart; an optional tail predicts inverse-distance
coefficients on its eight intrinsic coordinates. No priors are divided out here.
"""
import copy
import json
import numbers
from pathlib import Path

import numpy as np

SCHEMA = "rift-raw-distance-v1"
FEATURES = ("ln_mc_detector_msun", "delta", "s1x", "s1y", "s1z",
            "s2x", "s2y", "s2z", "inverse_distance")


def _finite(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(name + " must be real")
    if raw.dtype.kind not in "iuf":
        raise ValueError(name + " must be numeric")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def intrinsic_coordinates(m1_msun, m2_msun, spins, *, mass_frame, redshift=None):
    """Construct the native chart; source masses require explicit redshifts.

    Spins are (...,6) Cartesian components in the declared frame. Cosmology is
    supplied by the analysis: this scalar likelihood transform adds no Jacobian.
    """
    m1, m2 = np.broadcast_arrays(_finite(m1_msun, "Mass"), _finite(m2_msun, "Mass"))
    if np.any(m2 <= 0) or np.any(m1 < m2):
        raise ValueError("Require m1 >= m2 > 0")
    if mass_frame == "source":
        if redshift is None:
            raise ValueError("Source masses require explicit redshift")
        z = _finite(redshift, "Redshift")
        if np.any(z < 0):
            raise ValueError("Invalid redshift")
        m1, m2, z = np.broadcast_arrays(m1, m2, z)
        m1, m2 = _finite(m1*(1+z), "Detector mass"), _finite(m2*(1+z), "Detector mass")
    elif mass_frame != "detector" or redshift is not None:
        raise ValueError("Use detector masses without a redshift or source masses with one")
    spin = _finite(spins, "Spins")
    if spin.ndim < 1 or spin.shape[-1] != 6:
        raise ValueError("Expected six Cartesian spin components")
    spin = np.broadcast_to(spin, m1.shape+(6,))
    log_mc = .6*(np.log(m1)+np.log(m2))-.2*np.logaddexp(np.log(m1), np.log(m2))
    q = m2/m1
    return _finite(np.concatenate((log_mc[..., None], ((1-q)/(1+q))[..., None], spin), axis=-1),
                   "Intrinsic coordinates")


def validate_contract(contract):
    """Validate metadata without inventing a baseline or analysis prior."""
    c = copy.deepcopy(contract)
    if c.get("schema") != SCHEMA or c.get("quantity") != "raw_log_likelihood":
        raise ValueError("Expected a versioned raw likelihood artifact")
    if tuple(c.get("features", ())) != FEATURES:
        raise ValueError("Unsupported feature chart")
    if not c.get("spin_frame") or not c.get("source_provenance"):
        raise ValueError("Spin frame and source provenance are required")
    c["reference_frequency_hz"] = float(_finite(c["reference_frequency_hz"], "Reference frequency"))
    if c["reference_frequency_hz"] <= 0:
        raise ValueError("Invalid spin reference frequency")
    for name, shape in (("distance_support_mpc", (2,)),
                        ("intrinsic_bounds", (2, 8)), ("spin_caps", (2,))):
        a = _finite(c[name], name)
        if a.shape != shape:
            raise ValueError("Invalid " + name + " shape")
        c[name] = a.tolist()  # plain floats: JSON-safe and numeric downstream
    bounds = np.asarray(c["intrinsic_bounds"])
    low, high = c["distance_support_mpc"]
    if low <= 0 or high <= low or np.any(bounds[1] < bounds[0]):
        raise ValueError("Invalid support bounds")
    if np.any(np.asarray(c["spin_caps"]) <= 0) or np.any(np.asarray(c["spin_caps"]) > 1):
        raise ValueError("Invalid spin caps")
    c["distance_scale_mpc"] = float(_finite(c["distance_scale_mpc"], "Distance scale"))
    if c["distance_scale_mpc"] <= 0:
        raise ValueError("Invalid distance scale")
    norm = c.get("normalization")
    if not isinstance(norm, dict):
        raise ValueError("Normalization metadata is required")
    if norm.get("overflow_offset_restored") is not True:
        raise ValueError("Raw overflow offset must already be restored")
    if not norm.get("measure") or not norm.get("provenance"):
        raise ValueError("Native measure and baseline provenance are required")
    if norm.get("kind") == "constant":
        if "raw_zero_signal_log_likelihood" not in norm:
            raise ValueError("Constant baseline value is required")
        baseline = _finite(norm["raw_zero_signal_log_likelihood"], "Noise baseline")
        if baseline.ndim != 0:
            raise ValueError("Constant baseline must be scalar")
        norm["raw_zero_signal_log_likelihood"] = float(baseline)
    elif norm.get("kind") == "state_evaluator":
        if not norm.get("evaluator_id"):
            raise ValueError("State baseline requires an evaluator identity")
    else:
        raise ValueError("Unknown noise baseline")
    if "tail" in c:
        tail = c["tail"]
        if not isinstance(tail, dict):
            raise ValueError("Tail metadata must be a mapping")
        overlap = _finite(tail.get("overlap_mpc"), "Overlap")
        if overlap.shape != (2,) or not low <= overlap[0] < overlap[1] <= high:
            raise ValueError("Tail overlap must lie inside accepted distance support")
        tail["overlap_mpc"] = overlap.tolist()
        if not tail.get("validation_provenance"):
            raise ValueError("Tail validation provenance is required")
        tail["distance_scale_mpc"] = float(_finite(tail.get("distance_scale_mpc"), "Tail distance scale"))
        if tail["distance_scale_mpc"] <= 0:
            raise ValueError("Invalid tail distance scale")
        degree = tail.get("degree")
        if not isinstance(degree, numbers.Integral) or isinstance(degree, bool) or degree < 1:
            raise ValueError("Tail degree must be a positive integer")
        tail["degree"] = int(degree)
    return c


class DistanceLikelihood:
    """An explicitly normalized raw predictor, with RF-only evaluation by default.

    Predictor persistence is owned by its backend. JSON contract persistence
    does not deserialize code or relabel posterior-compression artifacts as raw.
    """
    def __init__(self, contract, core, *, tail=None, enable_tail=False,
                 baseline=None, baseline_id=None):
        self.contract = validate_contract(contract)
        self.core = core
        self.tail = tail
        self.enable_tail = bool(enable_tail)
        if self.enable_tail and (tail is None or "tail" not in self.contract):
            raise ValueError("Enabled tail requires a predictor and overlap contract")
        norm = self.contract["normalization"]
        if norm["kind"] == "state_evaluator":
            if baseline is None or baseline_id != norm["evaluator_id"]:
                raise ValueError("Provide the declared native baseline evaluator")
        elif baseline is not None:
            raise ValueError("A constant contract cannot use a different baseline")
        self.baseline = baseline

    def save_contract(self, path):
        Path(path).write_text(json.dumps(self.contract, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load_contract(cls, path, core, *, allow_experimental=False, **kwargs):
        contract = json.loads(Path(path).read_text())
        if not allow_experimental and contract.get("validation_status") != "accepted":
            raise ValueError("Experimental artifact requires explicit diagnostic access")
        return cls(contract, core, **kwargs)

    def prepare(self, intrinsic):
        """Prepare full-spin detector-frame states once for repeated distance grids.

        Input is (N,8), including all six Cartesian spins. Convert source masses
        using the caller's cosmology/redshift before constructing this chart.
        A scalar likelihood coordinate transformation has no density Jacobian.
        """
        x = _finite(intrinsic, "Intrinsic states")
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != 8 or not len(x):
            raise ValueError("Expected nonempty (N,8) intrinsic states")
        bounds = np.asarray(self.contract["intrinsic_bounds"])
        if (np.any(x < bounds[0]) or np.any(x > bounds[1]) or
                np.any(x[:, 1] < 0) or np.any(x[:, 1] >= 1)):
            raise ValueError("Intrinsic states outside declared support")
        for sl, cap in zip((slice(2, 5), slice(5, 8)), self.contract["spin_caps"]):
            if np.any(np.linalg.norm(x[:, sl], axis=1) > cap):
                raise ValueError("Spin outside declared support")
        unique, inverse = np.unique(x, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)  # numpy 2.0.0 returned a 2-D inverse for axis=0
        norm = self.contract["normalization"]
        if self.baseline is None:
            baseline = np.full(len(unique), norm["raw_zero_signal_log_likelihood"])
        else:
            baseline = _finite(self.baseline.predict(unique.copy()), "Noise baseline")
            if baseline.shape != (len(unique),):
                raise ValueError("Baseline evaluator must return one value per state")
        coefficients = None
        if self.enable_tail:
            coefficients = _finite(self.tail.predict(unique.copy()), "Tail coefficients")
            if coefficients.ndim == 1:
                coefficients = coefficients[:, None]
            if coefficients.shape != (len(unique), self.contract["tail"]["degree"]):
                raise ValueError("Coefficient vectors must match the declared tail degree")
        return PreparedDistanceLikelihood(self, unique, inverse, baseline, coefficients)


class PreparedDistanceLikelihood:
    """Explicit coefficient cache owned by one prepared full-spin state batch."""
    def __init__(self, model, unique, inverse, baseline, coefficients):
        self.contract = copy.deepcopy(model.contract)
        self.core = model.core
        self.intrinsic = unique.copy()
        self.inverse = inverse.copy()
        self.baseline = baseline.copy()
        self.coefficients = None if coefficients is None else coefficients.copy()

    def raw_log_likelihood(self, distances_mpc):
        """Return (N,K) for a shared (K,) grid or per-state (N,K) grids.

        A 1-D input is always a shared grid, even when K equals N. Pass one
        distance per state as an (N,1) array.
        """
        d = _finite(distances_mpc, "Distances")
        n = len(self.inverse)
        if d.ndim == 0:
            d = d.reshape(1)
        if d.ndim == 1:
            d = np.broadcast_to(d, (n, len(d)))
        if d.ndim != 2 or d.shape[0] != n:
            raise ValueError("Expected a shared grid or an (N,K) distance array")
        low, high = self.contract["distance_support_mpc"]
        if np.any(d < low) or np.any(d > high):
            raise ValueError("Distance outside accepted support")
        u = _finite(self.contract["distance_scale_mpc"] / d, "Inverse distance")
        weight = np.zeros_like(d)
        if self.coefficients is not None:
            lo, hi = self.contract["tail"]["overlap_mpc"]
            t = np.clip((d-lo)/(hi-lo), 0., 1.)
            weight = t*t*(3-2*t)
        result = np.zeros_like(d)
        core = weight < 1
        if core.any():
            states = np.broadcast_to(self.intrinsic[self.inverse, None, :], d.shape+(8,))
            rows = np.column_stack((states[core], u[core]))
            values = _finite(self.core.predict(rows), "Core likelihood")
            if values.shape != (core.sum(),):
                raise ValueError("Core predictor must return one raw log likelihood per row")
            result[core] = (1-weight[core])*values
        if self.coefficients is not None:
            # Horner evaluation has no constant term: the raw zero-signal value
            # is the native baseline, not a fitted event/state offset.
            tail = np.zeros_like(d)
            tail_u = _finite(self.contract["tail"]["distance_scale_mpc"] / d, "Tail inverse distance")
            for coefficient in self.coefficients[self.inverse].T[::-1]:
                tail = (tail + coefficient[:, None])*tail_u
            tail += self.baseline[self.inverse, None]
            active = weight > 0
            result[active] += weight[active]*tail[active]
        return _finite(result, "Raw likelihood")

    def log_noise_ratio(self, distances_mpc):
        return _finite(self.raw_log_likelihood(distances_mpc) - self.baseline[self.inverse, None],
                       "Noise ratio")

    def raw_zero_signal_log_likelihood(self):
        """Return the declared baseline; this does not extend accepted support."""
        return self.baseline[self.inverse].copy()
