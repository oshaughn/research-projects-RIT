#!/usr/bin/env python3
"""Thin shim over RIFT.simulation_manager.cli.request_sim — register a
simulation in an archive and (with --ensure) block until it's at the
requested level. Suitable as the executable for a glue.pipeline DAG
node embedded in an existing workflow.

See `python -m RIFT.simulation_manager.cli.request_sim --help` for full
option descriptions."""

import sys
from RIFT.simulation_manager.cli.request_sim import main

if __name__ == "__main__":
    sys.exit(main())
