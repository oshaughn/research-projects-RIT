"""Frozen generator for GW PE synthetic-targeted backend.

This generator is captured into the archive's `code/generator.py` at
archive creation time. It is NOT what runs the actual PE — that's
done by `util_RIFT_pseudo_pipe.py` invoked by the run queue's
subdag_factory. This file exists for the standard archive contract
(every archive freezes a generator) and as a safety net for backends
that ever want to switch to a non-SUBDAG inline path.

Params schema (a dict):
    mc:               chirp mass (M_sun)
    eta:              symmetric mass ratio
    distance:         luminosity distance (Mpc)
    inclination:      radians
    polarization:     radians
    ra:               right ascension (radians)
    dec:              declination (radians)
    geocent_time:     GPS seconds
    s1z, s2z:         aligned spins (optional; default 0)
    eccentricity:     optional; default 0
    approximant:      string (e.g. "SEOBNRv4PHM"); default "IMRPhenomXPHM"
    flow:             dict per-IFO low-frequency cutoff (default {H1: 20, L1: 20, V1: 20})
    psd_source:       "design_aligo" | "user" (default "design_aligo")
    noise_source:     "gwpy" | "user" (default "gwpy")
    user_noise_frames: list of pre-existing .gwf paths (used when
                       noise_source=="user")
    user_psd_path:    path to a custom psd.xml.gz (used when psd_source=="user")
"""

from __future__ import annotations

import json
import os


def run(params, sim_dir, level, prev_levels):
    """Stub `generator.run` that records a marker and exits.

    The actual PE is computed by the SUBDAG produced by the run queue's
    subdag_factory, NOT by this function. This is here only so an
    archive frozen against this backend has a defined `generator:run`
    entrypoint (the framework freezes one regardless of whether a
    subdag_factory is in play).

    A well-formed level_<N>.json is written so refresh_status_from_disk
    treats the sim as complete; the actual posterior content lives
    elsewhere (set by the pseudo_pipe-built sub-DAG)."""
    out = os.path.join(sim_dir, "level_{}.json".format(level))
    with open(out, "w") as f:
        json.dump({
            "level": int(level),
            "_note": ("This file is a stub written by the inline "
                      "generator. The actual PE outputs live in the "
                      "sub-DAG's rundir under sim_dir."),
            "prev_count": len(prev_levels),
        }, f)
