"""Generate a RIFT follow-up for the Rimsky online-PE orchestrator."""

import argparse
import copy
import json
from pathlib import Path


class RimskyIntegrationError(ValueError):
    """Raised when a Rimsky configuration cannot define a RIFT follow-up."""


def _deep_update(target, updates):
    """Recursively apply mapping ``updates`` without sharing mutable values."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_rimsky_config(path):
    """Load a Rimsky YAML configuration and return it as a mapping."""
    path = Path(path)
    try:
        import yaml
    except ImportError as exc:  # Rimsky itself depends on PyYAML.
        raise RimskyIntegrationError(
            "PyYAML is required to read a Rimsky configuration"
        ) from exc

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise RimskyIntegrationError("Rimsky configuration must be a mapping")
    return config


def _detectors(config):
    detectors = config.get("detectors", ["H1", "L1"])
    if isinstance(detectors, str):
        detectors = [item.strip() for item in detectors.split(",") if item.strip()]
    if not isinstance(detectors, (list, tuple)) or not detectors:
        raise RimskyIntegrationError("Rimsky 'detectors' must be a non-empty list")
    if not all(isinstance(detector, str) and detector for detector in detectors):
        raise RimskyIntegrationError("Each Rimsky detector must be a non-empty string")
    return list(detectors)


def _frequency_dict(value, detectors, field):
    if isinstance(value, (int, float)):
        return {detector: value for detector in detectors}
    if not isinstance(value, dict):
        raise RimskyIntegrationError(
            "{} must be a number or detector mapping".format(field)
        )
    missing = [detector for detector in detectors if detector not in value]
    if missing:
        raise RimskyIntegrationError(
            "{} is missing detector(s): {}".format(field, ", ".join(missing))
        )
    selected = {detector: value[detector] for detector in detectors}
    if not all(isinstance(item, (int, float)) for item in selected.values()):
        raise RimskyIntegrationError("{} values must be numeric".format(field))
    return selected


def _resolve_output_dir(config, config_path=None):
    output_dir = Path(config.get("output_dir", "output")).expanduser()
    if not output_dir.is_absolute():
        base = Path(config_path).resolve().parent if config_path else Path.cwd()
        output_dir = base / output_dir
    return output_dir.resolve()


def _resolve_config_path(value, *, config_path=None):
    """Resolve a Rimsky path using the source configuration as its anchor."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        base = Path(config_path).resolve().parent if config_path else Path.cwd()
        path = base / path
    return path.resolve()


def build_analysis(config, *, config_path=None, overrides=None):
    """Build the Asimov analysis document consumed by Rimsky's follow-up hook.

    Parameters
    ----------
    config : mapping
        Parsed Rimsky configuration.  An optional top-level ``rift`` mapping is
        ignored by Rimsky and accepted here as user overrides.
    config_path : path-like, optional
        Location of the Rimsky YAML file.  Relative ``output_dir`` values are
        resolved relative to this file, matching Rimsky's current behaviour.
    overrides : mapping, optional
        Programmatic overrides applied after the top-level ``rift`` mapping.

    Returns
    -------
    dict
        One Asimov analysis document suitable for
        ``sample_sink.asimov_configuration``.
    """
    if not isinstance(config, dict):
        raise RimskyIntegrationError("Rimsky configuration must be a mapping")

    detectors = _detectors(config)
    event_sink = config.get("event_sink") or {}
    bilby = event_sink.get("bilby_pipe_defaults") or {}
    minimum = _frequency_dict(
        bilby.get("minimum_frequency", 20), detectors, "minimum_frequency"
    )
    maximum = _frequency_dict(
        bilby.get("maximum_frequency", 1024), detectors, "maximum_frequency"
    )
    output_dir = _resolve_output_dir(config, config_path=config_path)

    # Rimsky stores each event under output_dir/YYMM/DD/SID and writes this
    # metafile immediately before invoking the configured Asimov follow-ups.
    bootstrap = output_dir / "*" / "*" / "{event}" / "results_page" / "metafile.hdf5"

    analysis = {
        "kind": "analysis",
        "name": "rift-online",
        "status": "Ready",
        "pipeline": "RIFT",
        "orchestrator": "rimsky",
        "comment": "RIFT follow-up launched by Rimsky after online Bilby PE",
        "dataset": "bilby-online",
        "likelihood": {
            "start frequency": min(minimum.values()),
            "minimum frequency": minimum,
            "assume": {"precessing": True},
            "marginalization": {"distance": True},
        },
        "quality": {
            "minimum frequency": minimum,
            "maximum frequency": maximum,
        },
        "waveform": {
            "approximant": "IMRPhenomXPHM",
            "pn amplitude order": 5,
            "maximum mode": 4,
        },
        "priors": {
            "mass 1": {"minimum": 1, "maximum": 1000},
        },
        "sampler": {"cip": {}, "ile": {}},
        "scheduler": {
            "accounting group": "ligo.dev.o4.cbc.pe.rift",
            "bootstrap coinc": True,
            "bootstrap file": str(bootstrap),
            "osg": False,
        },
    }

    configured = config.get("rift") or {}
    if not isinstance(configured, dict):
        raise RimskyIntegrationError(
            "Optional Rimsky 'rift' settings must be a mapping"
        )
    _deep_update(analysis, configured)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise RimskyIntegrationError("RIFT overrides must be a mapping")
        _deep_update(analysis, overrides)
    return analysis


def configure_rimsky(config, analysis_path, *, config_path=None):
    """Return a runnable Rimsky configuration wired to the RIFT follow-up.

    Rimsky interprets paths relative to its launch directory, rather than the
    YAML file.  Emit absolute paths so the online output observed by Rimsky is
    the same output searched by RIFT's bootstrap glob.  Missing orchestration
    settings default to a local Asimov project and a submitted Bilby run; an
    operator's explicit values are retained.
    """
    if not isinstance(config, dict):
        raise RimskyIntegrationError("Rimsky configuration must be a mapping")

    configured = copy.deepcopy(config)
    configured["output_dir"] = str(
        _resolve_output_dir(configured, config_path=config_path)
    )
    configured["asimovdir"] = str(
        _resolve_config_path(
            configured.get("asimovdir", "asimov"), config_path=config_path
        )
    )
    event_sink = configured.setdefault("event_sink", {})
    if not isinstance(event_sink, dict):
        raise RimskyIntegrationError("Rimsky 'event_sink' must be a mapping")
    event_sink.setdefault("bilby_pipe_format", "full-submit")

    sample_sink = configured.setdefault("sample_sink", {})
    if not isinstance(sample_sink, dict):
        raise RimskyIntegrationError("Rimsky 'sample_sink' must be a mapping")
    sample_sink["asimov_configuration"] = str(
        _resolve_config_path(analysis_path)
    )
    return configured


def normalize_event_metadata(metadata):
    """Return RIFT-compatible metadata from a Rimsky-created event mapping.

    Rimsky 0.1 emits Bilby parameter names with underscores.  RIFT's Asimov
    template predates that convention and uses names containing spaces.  Keep
    both spellings so other analyses in the same ledger are unaffected.
    """
    normalized = copy.deepcopy(metadata)
    priors = normalized.setdefault("priors", {})
    aliases = {
        "chirp_mass": "chirp mass",
        "mass_ratio": "mass ratio",
        "luminosity_distance": "luminosity distance",
        "mass_1": "mass 1",
    }
    for source, destination in aliases.items():
        if destination not in priors and source in priors:
            priors[destination] = copy.deepcopy(priors[source])

    for source, destination in (("a_1", "spin 1"), ("a_2", "spin 2")):
        if destination not in priors and source in priors:
            prior = priors[source]
            if isinstance(prior, dict) and "maximum" in prior:
                priors[destination] = {"maximum": prior["maximum"]}
    for source, destination in (("chi_1", "spin 1"), ("chi_2", "spin 2")):
        if destination not in priors and source in priors:
            priors[destination] = {"maximum": 0.99}

    # Asimov's RIFT PSD convention is sample-rate -> detector -> path, while
    # Rimsky records detector -> path. This copy belongs only to the RIFT
    # production, so the event document retained by Rimsky remains unchanged.
    psds = normalized.get("psds")
    sample_rate = normalized.get("likelihood", {}).get("sample rate")
    detectors = normalized.get("interferometers", [])
    if (
        isinstance(psds, dict)
        and sample_rate is not None
        and detectors
        and all(detector in psds for detector in detectors)
    ):
        normalized["psds"] = {
            sample_rate: {
                detector: copy.deepcopy(psds[detector]) for detector in detectors
            }
        }
    return normalized


def write_analysis(analysis, path):
    """Write one analysis document as YAML (or JSON, which is valid YAML)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(analysis, stream, indent=2)
            stream.write("\n")
    else:
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(analysis, stream, sort_keys=False)
    return path


def write_rimsky_config(config, path):
    """Write a Rimsky configuration containing the automatic RIFT hook."""
    return write_analysis(config, path)


def main(argv=None):
    """Create the RIFT analysis and a Rimsky config that invokes it."""
    parser = argparse.ArgumentParser(
        description="Generate a RIFT follow-up analysis for a Rimsky configuration"
    )
    parser.add_argument("rimsky_config", help="Rimsky YAML configuration")
    parser.add_argument("output", help="Destination analysis YAML")
    parser.add_argument(
        "--configured-rimsky",
        help=(
            "Destination for the runnable Rimsky YAML "
            "(default: <input stem>-rift.yaml)"
        ),
    )
    args = parser.parse_args(argv)

    config = load_rimsky_config(args.rimsky_config)
    analysis_path = write_analysis(
        build_analysis(config, config_path=args.rimsky_config), args.output
    )
    source = Path(args.rimsky_config)
    configured_path = Path(args.configured_rimsky) if args.configured_rimsky else (
        source.with_name(source.stem + "-rift" + source.suffix)
    )
    write_rimsky_config(
        configure_rimsky(
            config, analysis_path.resolve(), config_path=args.rimsky_config
        ),
        configured_path,
    )
    print(analysis_path)
    print(configured_path)


if __name__ == "__main__":
    main()
