"""Helpers for matching bilby-pipe calibration conventions."""


def correction_type_for_ifo(setting, ifo_name, parse_dict=None):
    """Resolve bilby-pipe's calibration correction type for one detector."""
    if setting is None or setting == "None":
        return "template" if ifo_name == "V1" else "data"

    if isinstance(setting, str):
        if setting in ("data", "template"):
            return setting
        if parse_dict is None:
            # Accept the compact CLI form H1:data,L1:data,V1:template as well as
            # bilby-pipe's optionally quoted {H1: data, ...} representation.
            pairs = setting.strip().strip("{}").split(",")
            try:
                setting = dict(tuple(part.strip().strip("\"'") for part in pair.split(":"))
                               for pair in pairs)
            except ValueError as exc:
                raise ValueError("Expected data, template, or IFO:convention pairs") from exc
        else:
            setting = parse_dict(setting)

    try:
        correction_type = setting[ifo_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"No calibration correction type specified for {ifo_name}"
        ) from exc
    if correction_type not in ("data", "template"):
        raise ValueError(
            f"Invalid calibration correction type for {ifo_name}: {correction_type}"
        )
    return correction_type


def correction_types_for_ifos(setting, ifos):
    """Resolve and validate the envelope convention before constructing any draws."""
    return {ifo: correction_type_for_ifo(setting, ifo) for ifo in ifos}


def correction_types_argument(setting, ifos):
    """Serialize conventions as a single shell-safe command-line token."""
    resolved = correction_types_for_ifos(setting, ifos)
    return ",".join(f"{ifo}:{value}" for ifo, value in resolved.items())
