"""
RIFT.hyperpipe.coords
=====================

Coordinate-transformation framework for the hyperpipeline.

This module mirrors the conventions already established by
``util_ConstructEOSPosterior.py`` (and the broader CIP family) so that
hyperpipe configurations can reuse existing coord modules without a
parallel ecosystem.  Specifically:

  * A "coord module" is just an *importable Python module name*. When
    passed to the post-stage executable via ``--supplementary-coordinate-code``,
    that executable will ``__import__`` it at runtime and call into it for
    coordinate conversion / Jacobian / prior factors.

  * Each coordinate appears in the post stage via:
        ``--parameter <name>``  (fitting & MC parameter)
        ``--integration-parameter-range <name>:[a,b]``  (sampling bound)
    plus optionally
        ``--parameter-implied <name>``  (used in fit, not independently sampled)
        ``--parameter-nofit <name>``   (sampled but not a fit coordinate)

  * For *heterogeneous* analyses (multiple likelihood drivers contributing
    to the same hyperparameter inference), each driver may specify its own
    coord module --- this is just passed through as an extra argument to
    that driver's args line (typically as ``--supplementary-coordinate-code``
    if that driver supports it, else as a custom flag the driver consumes).

The :class:`HyperCoordSpec` dataclass below bundles the four pieces a
hyperpipe configuration needs to know:

    name             # what gets passed to --supplementary-coordinate-code
    parameters       # list of fitting parameters (== --parameter X ...)
    parameter_ranges # dict[str, (a, b)]  for --integration-parameter-range
    implied          # list of parameter-implied entries (optional)
    nofit            # list of parameter-nofit entries (optional)
    likelihood_factor # optional (module, function, ini) trio for
                      # --supplementary-likelihood-factor-{code,function,ini}

and provides helpers that emit the argument strings the post stage (and
optional per-driver) consume.
"""

from __future__ import annotations

import importlib
import logging
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

_RANGE_RE = re.compile(
    r"""^\s*
        (?P<name>[A-Za-z_][\w]*)
        \s*[:=]\s*
        \[\s*
            (?P<lo>[-+0-9eE.naif]+)
            \s*,\s*
            (?P<hi>[-+0-9eE.naif]+)
        \s*\]
        \s*$""",
    re.VERBOSE,
)


def parse_range_block(block: str) -> Tuple[str, Tuple[float, float]]:
    """Parse a single ``name:[lo,hi]`` block into ``(name, (lo, hi))``.

    The form ``name=[lo,hi]`` is also accepted as a convenience.
    """
    m = _RANGE_RE.match(block)
    if not m:
        raise ValueError(
            f"Could not parse coord-range block {block!r}; "
            "expected 'name:[lo,hi]'."
        )
    lo = float(m.group("lo"))
    hi = float(m.group("hi"))
    if not lo < hi:
        raise ValueError(
            f"Range for {m.group('name')!r} must be increasing; got [{lo},{hi}]."
        )
    return m.group("name"), (lo, hi)


def parse_range_string(s: str) -> Dict[str, Tuple[float, float]]:
    """Parse a space-separated string of ``name:[lo,hi]`` blocks.

    Example
    -------
    >>> parse_range_string("x:[-8,8] y:[-1,1]")
    {'x': (-8.0, 8.0), 'y': (-1.0, 1.0)}
    """
    out: Dict[str, Tuple[float, float]] = {}
    if not s:
        return out
    for block in shlex.split(s):
        name, rng = parse_range_block(block)
        if name in out:
            raise ValueError(f"Duplicate range for parameter {name!r} in {s!r}.")
        out[name] = rng
    return out


def parse_parameter_list(s: str) -> List[str]:
    """Split a space-separated parameter-name string into a list, preserving order."""
    if not s:
        return []
    return shlex.split(s)


# --------------------------------------------------------------------------
# HyperCoordSpec
# --------------------------------------------------------------------------


@dataclass
class HyperCoordSpec:
    """Bundle describing the coordinates a hyperpipe stage operates in.

    Parameters
    ----------
    name
        Name of an *importable* Python module used by the consumer for
        coord conversion / Jacobian / priors. If ``None``, no
        ``--supplementary-coordinate-code`` is emitted.
    parameters
        Fitting / MC parameter names. Emitted as ``--parameter X`` flags.
    parameter_ranges
        Map from parameter name to ``(lo, hi)``. Emitted as
        ``--integration-parameter-range X:[lo,hi]``.
    implied
        Parameters used in the fit but not independently sampled.
        Emitted as ``--parameter-implied X``.
    nofit
        Parameters sampled but not in the fit.  Emitted as
        ``--parameter-nofit X``.
    likelihood_factor
        Optional ``(module, function, ini)`` triple wiring a
        supplementary external-prior / likelihood factor through the
        post stage (``--supplementary-likelihood-factor-{code,function,ini}``).
    """

    name: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    implied: List[str] = field(default_factory=list)
    nofit: List[str] = field(default_factory=list)
    likelihood_factor: Optional[Tuple[str, Optional[str], Optional[str]]] = None

    # ----- construction --------------------------------------------------
    @classmethod
    def from_strings(
        cls,
        *,
        name: Optional[str] = None,
        coords_fit: str = "",
        coords_sample: str = "",
        coords_implied: str = "",
        coords_nofit: str = "",
        likelihood_factor: Optional[Sequence[Optional[str]]] = None,
    ) -> "HyperCoordSpec":
        """Build a spec from the string-shaped fields a Hydra config gives us.

        ``coords_fit``     : "x y z"
        ``coords_sample``  : "x:[-8,8] y:[-8,8] z:[-8,8]"
        ``coords_implied`` : "R1.4 Mmax"   (optional)
        ``coords_nofit``   : "delta_mc s1z s2z"   (optional)
        ``likelihood_factor``: (module, function, ini)  (any element may be None)
        """
        params = parse_parameter_list(coords_fit)
        ranges = parse_range_string(coords_sample)
        unknown = set(ranges) - set(params)
        if unknown:
            raise ValueError(
                f"coords-sample names a parameter not in coords-fit: {sorted(unknown)!r}"
            )
        lf: Optional[Tuple[str, Optional[str], Optional[str]]] = None
        if likelihood_factor:
            # Pad to length 3 and coerce empties to None
            seq = list(likelihood_factor) + [None] * (3 - len(likelihood_factor))
            seq = [s if s else None for s in seq[:3]]
            if seq[0] is not None:
                lf = (seq[0], seq[1], seq[2])  # type: ignore[assignment]
        return cls(
            name=name or None,
            parameters=params,
            parameter_ranges=ranges,
            implied=parse_parameter_list(coords_implied),
            nofit=parse_parameter_list(coords_nofit),
            likelihood_factor=lf,
        )

    # ----- validation ----------------------------------------------------
    def validate(self, strict_import: bool = False) -> None:
        """Sanity-check the spec; optionally verify the coord module imports.

        ``strict_import=True`` will attempt ``importlib.import_module(self.name)``
        and raise on failure; with ``False`` we only warn, since coord
        modules are often only importable inside the downstream runtime
        environment (singularity image, OSG worker) and not necessarily
        on the submit host.
        """
        if not self.parameters:
            raise ValueError("HyperCoordSpec requires at least one fitting parameter.")
        missing = [p for p in self.parameters if p not in self.parameter_ranges]
        if missing:
            raise ValueError(
                f"No integration range supplied for parameter(s): {missing!r}. "
                "Every entry in coords-fit must appear in coords-sample."
            )
        for p, (lo, hi) in self.parameter_ranges.items():
            if not lo < hi:
                raise ValueError(f"Range for {p!r} must be increasing; got [{lo},{hi}].")
        if self.name:
            try:
                importlib.import_module(self.name)
            except Exception as exc:  # noqa: BLE001 -- broad on purpose; cf. docstring
                msg = (
                    f"HyperCoordSpec: coord module {self.name!r} did not import "
                    f"on the submit host ({type(exc).__name__}: {exc}). "
                    "If it is only present on the worker, this is expected."
                )
                if strict_import:
                    raise ImportError(msg) from exc
                logger.warning(msg)

    # ----- emission ------------------------------------------------------
    @staticmethod
    def _fmt_num(x: float) -> str:
        """Format a coord bound, preserving the integer form when applicable.

        e.g. -8.0 -> '-8'   (matches the existing hyperpipe demos)
              -1.6 -> '-1.6'
              1e-05 -> '1e-05'
        """
        try:
            xi = int(x)
        except (OverflowError, ValueError):
            return repr(x)
        if xi == x:
            return str(xi)
        # general format strips trailing zeros while keeping decimal precision
        return format(x, "g")

    def to_parameter_args(self) -> str:
        """Emit ``--parameter X`` / ``--integration-parameter-range X:[a,b]`` flags.

        Includes implied / nofit and any required-by-CIP ordering. Returns a
        single space-joined string, ready to be appended to a hyperpipe
        args_*.txt file.
        """
        bits: List[str] = []
        for p in self.parameters:
            bits.append(f"--parameter {p}")
        for p in self.implied:
            bits.append(f"--parameter-implied {p}")
        for p in self.nofit:
            bits.append(f"--parameter-nofit {p}")
        for p in self.parameters:
            lo, hi = self.parameter_ranges[p]
            bits.append(
                f"--integration-parameter-range {p}:[{self._fmt_num(lo)},{self._fmt_num(hi)}]"
            )
        return " ".join(bits)

    def to_post_args(self) -> str:
        """Emit the post-stage arg block (parameters + coord-module + lf trio)."""
        bits = [self.to_parameter_args()]
        if self.name:
            bits.append(f"--supplementary-coordinate-code {self.name}")
        if self.likelihood_factor is not None:
            mod, fn, ini = self.likelihood_factor
            bits.append(f"--supplementary-likelihood-factor-code {mod}")
            if fn:
                bits.append(f"--supplementary-likelihood-factor-function {fn}")
            if ini:
                bits.append(f"--supplementary-likelihood-factor-ini {ini}")
        return " ".join(b for b in bits if b)

    def to_puff_args(self, force_away: float = 0.03, puff_factor: float = 0.5) -> str:
        """Emit the puff-stage arg block.

        By default we puff in every fitting parameter; this is what every
        existing hyperpipe example does. Extra flags can be appended by the
        caller.
        """
        bits = [f"--force-away {force_away}", f"--puff-factor {puff_factor}"]
        for p in self.parameters:
            bits.append(f"--parameter {p}")
        return " ".join(bits)

    def to_test_args(self, method: str = "JS", threshold: float = 0.05) -> str:
        """Emit the convergence-test arg block.

        Mirrors the args_test.txt pattern from the Gaussian demo:
            ``--parameter x --parameter y --parameter z --method JS --threshold 0.05``
        """
        bits = [f"--parameter {p}" for p in self.parameters]
        bits.append(f"--method {method}")
        bits.append(f"--threshold {threshold}")
        return " ".join(bits)

    # ----- per-driver hook ---------------------------------------------------
    def to_driver_coord_flag(self) -> str:
        """Emit only the ``--supplementary-coordinate-code`` flag, if any.

        Useful for composing into a marg-driver's per-event args line when
        that driver also consumes the same coord-module convention (e.g.
        CIP-as-marg in the GW+NICER NS-EOS pipeline).
        """
        return f"--supplementary-coordinate-code {self.name}" if self.name else ""


# --------------------------------------------------------------------------
# Convenience constructors
# --------------------------------------------------------------------------


def coord_spec_from_config_section(section) -> HyperCoordSpec:
    """Build a :class:`HyperCoordSpec` from a Hydra ``post`` (or analogous) section.

    The section is expected to provide the keys:
        ``coord-module`` (str, optional)
        ``coords-fit``  (str)
        ``coords-sample`` (str)
        ``coords-implied`` (str, optional)
        ``coords-nofit`` (str, optional)
        ``likelihood-factor-module`` / ``likelihood-factor-function`` /
        ``likelihood-factor-ini`` (str, optional)
    """
    def _get(key, default=None):
        # tolerate both DictConfig and plain dict
        try:
            return section.get(key, default)  # type: ignore[union-attr]
        except AttributeError:
            return section[key] if key in section else default

    lf_mod = _get("likelihood-factor-module")
    lf = None
    if lf_mod:
        lf = (lf_mod, _get("likelihood-factor-function"), _get("likelihood-factor-ini"))

    return HyperCoordSpec.from_strings(
        name=_get("coord-module"),
        coords_fit=_get("coords-fit", "") or "",
        coords_sample=_get("coords-sample", "") or "",
        coords_implied=_get("coords-implied", "") or "",
        coords_nofit=_get("coords-nofit", "") or "",
        likelihood_factor=lf,
    )
