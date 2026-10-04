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
    function, ini, chart
        Optional ``--supplementary-coordinate-{function,ini,chart}`` values.
        They go to every stage that loads the coord module (post and, in the
        plugin basis, puff).
    input_parameters
        Data-file columns the coord module maps from (the plugin's input
        basis).  Grid and posterior files are always written in these
        columns.  Needed when the sampling basis is the plugin's output
        basis, so the puff stage can round-trip through the plugin and the
        test stage can read the file columns.
    """

    name: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    implied: List[str] = field(default_factory=list)
    nofit: List[str] = field(default_factory=list)
    likelihood_factor: Optional[Tuple[str, Optional[str], Optional[str]]] = None
    function: Optional[str] = None
    ini: Optional[str] = None
    chart: Optional[str] = None
    input_parameters: List[str] = field(default_factory=list)
    # True when function/ini/chart came from post.extra-args, which the post
    # stage already receives verbatim; they are then forwarded only to puff.
    plugin_flags_in_extra_args: bool = False

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
        coord_function: Optional[str] = None,
        coord_ini: Optional[str] = None,
        coord_chart: Optional[str] = None,
        coord_input_parameters: str = "",
    ) -> "HyperCoordSpec":
        """Build a spec from the string-shaped fields a Hydra config gives us.

        ``coords_fit``     : "x y z"
        ``coords_sample``  : "x:[-8,8] y:[-8,8] z:[-8,8]"
        ``coords_implied`` : "R1.4 Mmax"   (optional)
        ``coords_nofit``   : "delta_mc s1z s2z"   (optional)
        ``likelihood_factor``: (module, function, ini)  (any element may be None)
        """
        params  = parse_parameter_list(coords_fit)
        implied = parse_parameter_list(coords_implied)
        nofit   = parse_parameter_list(coords_nofit)
        ranges  = parse_range_string(coords_sample)
        # coords-sample provides INTEGRATION ranges, so it has to cover every
        # name in the MC SAMPLING basis -- which is coords-fit + coords-nofit.
        # (implied names are fit-only and don't need a sample range.)  Pre-
        # decoupling this was just coords-fit because nofit/implied were
        # rarely used and the sampling basis was forced to equal the fit
        # basis; now we have to allow the nofit names too.
        sampling_basis = set(params) | set(nofit)
        unknown = set(ranges) - sampling_basis
        if unknown:
            raise ValueError(
                f"coords-sample names a parameter not in coords-fit or "
                f"coords-nofit: {sorted(unknown)!r}"
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
            implied=implied,
            nofit=nofit,
            likelihood_factor=lf,
            function=coord_function or None,
            ini=coord_ini or None,
            chart=coord_chart or None,
            input_parameters=parse_parameter_list(coord_input_parameters),
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
        # The fit basis is coords-fit + coords-implied; the sampling basis is
        # coords-fit + coords-nofit.  Both must be non-empty for the run to
        # make sense.  Pre-decoupling this only required coords-fit -- now
        # an "EOS-style fit in a transformed basis" config can legally have
        # empty coords-fit (everything goes through implied + nofit).
        if not self.parameters and not self.implied:
            raise ValueError(
                "HyperCoordSpec requires at least one fit dimension "
                "(coords-fit or coords-implied)."
            )
        if not self.parameters and not self.nofit:
            raise ValueError(
                "HyperCoordSpec requires at least one MC sampling dimension "
                "(coords-fit or coords-nofit)."
            )
        # Every name in the SAMPLING basis must have an integration range
        # (the integrator reads prior_range_map[p] for p in low_level_coord_names).
        sampling_names = list(self.parameters) + list(self.nofit)
        missing = [p for p in sampling_names if p not in self.parameter_ranges]
        if missing:
            raise ValueError(
                f"No integration range supplied for sampling parameter(s): "
                f"{missing!r}. Every entry in coords-fit and coords-nofit must "
                "appear in coords-sample."
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
        # Integration ranges cover the MC SAMPLING basis (parameters + nofit).
        # Implied coordinates are fit-only and don't have a sampling range.
        for p in list(self.parameters) + list(self.nofit):
            lo, hi = self.parameter_ranges[p]
            bits.append(
                f"--integration-parameter-range {p}:[{self._fmt_num(lo)},{self._fmt_num(hi)}]"
            )
        return " ".join(bits)

    # ----- bases ---------------------------------------------------------
    def sampling_basis(self) -> List[str]:
        return list(self.parameters) + list(self.nofit)

    def samples_in_plugin_basis(self) -> bool:
        """True when the MC samples in the coord module's OUTPUT basis.

        The post stage then writes its samples back in the module's input
        (data-file) columns, so the puff and test stages cannot use the
        sampling-basis names as file columns.  Decidable only when
        ``input_parameters`` is declared; a sampling basis that mixes
        plugin-output names and data-file columns is refused.
        """
        if not (self.name and self.input_parameters):
            return False
        sampling = self.sampling_basis()
        in_file = [p for p in sampling if p in self.input_parameters]
        if in_file and len(in_file) != len(sampling):
            raise ValueError(
                f"Sampling basis {sampling!r} mixes coord-module outputs and "
                f"data-file columns {in_file!r}; the puff and test stages "
                "need one basis or the other."
            )
        return not in_file

    def puff_basis(self, mode: str = "auto") -> Tuple[List[str], bool]:
        """Return (names, use_plugin) for the puff stage.

        ``mode`` is ``auto`` (plugin basis exactly when the sampling basis is
        the plugin's output basis), ``file`` (data-file columns, no plugin) or
        ``plugin`` (puff in the plugin's output basis even when the MC samples
        data-file columns: the fit-basis names that are not file columns).
        """
        if mode not in ("auto", "file", "plugin"):
            raise ValueError(f"puff coord-basis must be auto, file or plugin; got {mode!r}")
        if mode == "file" or (mode == "auto" and not self.samples_in_plugin_basis()):
            if self.samples_in_plugin_basis():
                raise ValueError(
                    "puff coord-basis 'file' cannot work: the sampling basis "
                    f"{self.sampling_basis()!r} is not data-file columns."
                )
            return self.sampling_basis(), False
        if not (self.name and self.input_parameters):
            raise ValueError(
                "puff coord-basis 'plugin' needs post.coord-module and "
                "post.coord-input-parameters (the data-file columns the module maps from)."
            )
        if self.samples_in_plugin_basis():
            return self.sampling_basis(), True
        names = [p for p in dict.fromkeys(list(self.parameters) + list(self.implied))
                 if p not in self.input_parameters]
        if not names:
            raise ValueError("puff coord-basis 'plugin': no fit-basis name is a coord-module output.")
        return names, True

    def test_basis(self) -> List[str]:
        """Columns of the grid / posterior files the convergence test reads."""
        if self.samples_in_plugin_basis():
            return list(self.input_parameters)
        return self.sampling_basis()

    def _plugin_flags(self, with_input_parameters: bool) -> List[str]:
        bits = [f"--supplementary-coordinate-code {self.name}"]
        if self.function:
            bits.append(f"--supplementary-coordinate-function {self.function}")
        if self.ini:
            bits.append(f"--supplementary-coordinate-ini {self.ini}")
        if self.chart:
            bits.append(f"--supplementary-coordinate-chart {self.chart}")
        if with_input_parameters:
            bits += [f"--supplementary-coordinate-input-parameter {p}" for p in self.input_parameters]
        return bits

    def to_post_args(self) -> str:
        """Emit the post-stage arg block (parameters + coord-module + lf trio)."""
        bits = [self.to_parameter_args()]
        if self.name:
            if self.plugin_flags_in_extra_args:
                bits.append(f"--supplementary-coordinate-code {self.name}")
            else:
                bits += self._plugin_flags(with_input_parameters=False)
        if self.likelihood_factor is not None:
            mod, fn, ini = self.likelihood_factor
            bits.append(f"--supplementary-likelihood-factor-code {mod}")
            if fn:
                bits.append(f"--supplementary-likelihood-factor-function {fn}")
            if ini:
                bits.append(f"--supplementary-likelihood-factor-ini {ini}")
        return " ".join(b for b in bits if b)

    def to_puff_args(self, force_away: float = 0.03, puff_factor: float = 0.5,
                     coord_basis: str = "auto") -> str:
        """Emit the puff-stage arg block.

        The puff lane reads / writes grid files in the data-file column
        basis, which is the MC sampling basis (coords-fit + coords-nofit).
        Pre-decoupling this only emitted --parameter for coords-fit because
        the sampling basis was forced to equal the fit basis; once those
        diverge (EOSPosterior with --parameter-implied for a transformed
        fit basis), the puff lane must continue to operate on the data-
        file columns -- i.e. coords-fit + coords-nofit.

        When the MC samples in the coord module's output basis (see
        :meth:`samples_in_plugin_basis`), or ``coord_basis='plugin'``, the
        puff runs in that basis and gets the coord module, so it can map the
        file columns in and back out.
        """
        names, use_plugin = self.puff_basis(coord_basis)
        bits = [f"--force-away {force_away}", f"--puff-factor {puff_factor}"]
        for p in names:
            bits.append(f"--parameter {p}")
        if use_plugin:
            bits += self._plugin_flags(with_input_parameters=True)
        return " ".join(bits)

    def to_test_args(self, method: str = "JS", threshold: float = 0.05) -> str:
        """Emit the convergence-test arg block.

        Mirrors the args_test.txt pattern from the Gaussian demo:
            ``--parameter x --parameter y --parameter z --method JS --threshold 0.05``

        Like the puff lane, this operates on the SAMPLING basis (coords-fit
        + coords-nofit) -- the convergence-test driver reads grid / posterior
        files whose columns are in the sampling basis.
        """
        bits = [f"--parameter {p}" for p in self.test_basis()]
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

    # Coord-module options may be structured keys or, as in older configs,
    # flags inside post.extra-args.  Either way the puff stage needs them.
    from_extra = _plugin_flags_from_args(_get("extra-args", "") or "")
    if "input-parameter" in from_extra:
        # extra-args reaches util_ConstructEOSPosterior.py verbatim, and it has no such flag.
        raise ValueError(
            "post.extra-args: --supplementary-coordinate-input-parameter is a puff-stage flag; "
            "set post.coord-input-parameters instead."
        )
    spec = HyperCoordSpec.from_strings(
        name=_get("coord-module"),
        coords_fit=_get("coords-fit", "") or "",
        coords_sample=_get("coords-sample", "") or "",
        coords_implied=_get("coords-implied", "") or "",
        coords_nofit=_get("coords-nofit", "") or "",
        likelihood_factor=lf,
        coord_function=_get("coord-function") or from_extra.get("function"),
        coord_ini=_get("coord-ini") or from_extra.get("ini"),
        coord_chart=_get("coord-chart") or from_extra.get("chart"),
        coord_input_parameters=_get("coord-input-parameters", "") or "",
    )
    structured = any(_get(k) for k in ("coord-function", "coord-ini", "coord-chart"))
    if from_extra and structured:
        raise ValueError(
            "post: give coord-function / coord-ini / coord-chart either as keys "
            "or as --supplementary-coordinate-* flags in extra-args, not both."
        )
    spec.plugin_flags_in_extra_args = bool(from_extra)
    return spec


def _plugin_flags_from_args(args: str) -> Dict[str, object]:
    """Pull --supplementary-coordinate-{function,ini,chart,input-parameter} out of an args string."""
    out: Dict[str, object] = {}
    toks = shlex.split(args)
    for i, tok in enumerate(toks):
        key, _, val = tok.partition("=")
        if not key.startswith("--supplementary-coordinate-"):
            continue
        sub = key[len("--supplementary-coordinate-"):]
        if sub not in ("function", "ini", "chart", "input-parameter"):
            continue
        if not val:
            if i + 1 >= len(toks):
                raise ValueError(f"{key} needs a value in extra-args")
            val = toks[i + 1]
        if sub == "input-parameter":
            out.setdefault(sub, []).append(val)  # type: ignore[union-attr]
        else:
            out[sub] = val
    return out
