"""Helpers for matching bilby-pipe calibration conventions."""


def correction_type_for_ifo(setting, ifo_name, parse_dict=None):
    """Resolve bilby-pipe's calibration correction type for one detector."""
    if setting is None or setting == "None":
        return "template" if ifo_name == "V1" else "data"

    if isinstance(setting, str):
        if setting in ("data", "template"):
            return setting
        if parse_dict is None:
            raise ValueError("parse_dict is required for detector-specific settings")
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
