#!/usr/bin/env python3
"""Thin shim over RIFT.simulation_manager.cli.admin — archive admin
operations (verify, rebuild-index, resummarize, unstick).

Subcommands:
    verify           cross-check the index, status.json, and on-disk files
    rebuild-index    reconstruct index.jsonl from per-sim status.json
    resummarize      re-run the manifest's summarizer over every sim
    unstick          clear stuck status; --bump-memory to raise per-sim mem

Run `sim_archive_admin.py <subcmd> --help` for per-subcommand options."""

import sys
from RIFT.simulation_manager.cli.admin import main

if __name__ == "__main__":
    sys.exit(main())
