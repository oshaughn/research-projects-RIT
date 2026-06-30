#!/usr/bin/env python
"""Contract test for the LISA production-ini path (util_RIFT_pseudo_pipe --use-ini).

The LISA known-sky workflow can be driven either from --lisa-* CLI flags or from
an .ini file (production form).  This test renders BOTH ways with matched config
and asserts the generated workflow is identical -- so the ini path stays a thin
front-end over the validated CLI machinery and never silently diverges.

The ini sources per-channel data products from the conventional sections
([data] channels, [lalinference] psds); everything else flows through the generic
[rift-pseudo-pipe] parser by CLI-arg name.
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "MonteCarloMarginalizeCode", "Code")
PSEUDO_PIPE = os.path.join(CODE_DIR, "bin", "util_RIFT_pseudo_pipe.py")


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = os.path.join(CODE_DIR, "bin") + os.pathsep + env.get("PATH", "")
    return env


# matched (CLI flag, ini key, value) for the scalar/algorithm options
_COMMON = dict(
    approx="IMRPhenomD",
    event_time="0",
    ecliptic_longitude="1.0",
    ecliptic_latitude="0.3",
    srate="0.25",
    fmin="0.0001",
    fmax="0.125",
    fref="0.005",
    window="300",
    grid_size="16",
    n_iter="4",
    n_samp="16",
)


def _write_ini(path, cache, psd):
    path.write_text(
        "[data]\n"
        "channels = {{'A': 'fake_strain', 'E': 'fake_strain', 'T': 'fake_strain'}}\n"
        "\n[lalinference]\n"
        "psds = {{'A': '{A}', 'E': '{E}', 'T': '{T}'}}\n"
        "\n[rift-pseudo-pipe]\n"
        "lisa-known-sky=True\n"
        "lisa-vary-sky=True\n"
        "lisa-search-reflected-sky-mode=True\n"
        "lisa-reference-time=0\n"
        'approx="{approx}"\n'
        "event-time={event_time}\n"
        "ecliptic-longitude={ecliptic_longitude}\n"
        "ecliptic-latitude={ecliptic_latitude}\n"
        'lisa-cache-file="{cache}"\n'
        "lisa-srate={srate}\n"
        "lisa-fmin-template={fmin}\n"
        "lisa-fmax={fmax}\n"
        "lisa-reference-freq={fref}\n"
        "lisa-data-integration-window-half={window}\n"
        "lisa-grid-size={grid_size}\n"
        "lisa-n-iterations={n_iter}\n"
        "lisa-n-samples-per-job={n_samp}\n"
        "internal-ile-request-memory=2048\n"
        "internal-cip-request-memory=2048\n".format(
            A=psd["A"], E=psd["E"], T=psd["T"], cache=cache, **_COMMON
        )
    )


def _render_cli(rundir, cache, psd):
    cmd = [
        sys.executable, PSEUDO_PIPE,
        "--lisa-known-sky", "--lisa-vary-sky",
        "--lisa-search-reflected-sky-mode", "--lisa-reference-time", "0",
        "--use-rundir", os.fspath(rundir),
        "--approx", _COMMON["approx"],
        "--event-time", _COMMON["event_time"],
        "--ecliptic-longitude", _COMMON["ecliptic_longitude"],
        "--ecliptic-latitude", _COMMON["ecliptic_latitude"],
        "--lisa-cache-file", cache,
        "--lisa-channel-name", "A=fake_strain",
        "--lisa-channel-name", "E=fake_strain",
        "--lisa-channel-name", "T=fake_strain",
        "--lisa-psd-file", "A={}".format(psd["A"]),
        "--lisa-psd-file", "E={}".format(psd["E"]),
        "--lisa-psd-file", "T={}".format(psd["T"]),
        "--lisa-srate", _COMMON["srate"],
        "--lisa-fmin-template", _COMMON["fmin"],
        "--lisa-fmax", _COMMON["fmax"],
        "--lisa-reference-freq", _COMMON["fref"],
        "--lisa-data-integration-window-half", _COMMON["window"],
        "--lisa-grid-size", _COMMON["grid_size"],
        "--lisa-n-iterations", _COMMON["n_iter"],
        "--lisa-n-samples-per-job", _COMMON["n_samp"],
        "--internal-ile-request-memory", "2048",
        "--internal-cip-request-memory", "2048",
    ]
    subprocess.run(cmd, check=True, env=_env())


def test_lisa_ini_path_matches_cli(tmp_path):
    cache = os.fspath(tmp_path / "lisa.cache")
    psd = {c: os.fspath(tmp_path / "{}_psd.xml.gz".format(c)) for c in ("A", "E", "T")}

    cli_dir = tmp_path / "cli"
    ini_dir = tmp_path / "ini"
    ini_file = tmp_path / "demo.ini"
    _write_ini(ini_file, cache, psd)

    _render_cli(cli_dir, cache, psd)
    subprocess.run(
        [sys.executable, PSEUDO_PIPE, "--use-ini", os.fspath(ini_file),
         "--use-rundir", os.fspath(ini_dir)],
        check=True, env=_env(),
    )

    def norm(rundir, name):
        return (rundir / name).read_text().replace(os.fspath(rundir), "RUNDIR")

    # the rendered ILE / CIP / test argument files must be byte-identical
    for name in ("args_ile.txt", "args_cip_list.txt", "args_test.txt"):
        assert norm(cli_dir, name) == norm(ini_dir, name), name

    # the ini path must produce the same workflow (incl. the reflect node)
    cli_dag = next(cli_dir.glob("*.dag")).read_text()
    ini_dag = next(ini_dir.glob("*.dag")).read_text()
    assert cli_dag.count("\nJOB ") == ini_dag.count("\nJOB ")
    assert "convert_primary_sky_mode_to_secondary" in ini_dag
    # data products sourced from [data]/[lalinference] reach the ILE args
    ile_args = (ini_dir / "args_ile.txt").read_text()
    assert "--channel-name A=fake_strain" in ile_args
    assert psd["A"] in ile_args
