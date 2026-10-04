"""Keep narrow BNS masses distinct through both standard RIFT cleaner passes."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
BIN = CODE / "bin"
CLEAN = BIN / "util_CleanILE.py"
BUILDER = BIN / "create_event_parameter_pipeline_BasicIteration"


def env():
    result = dict(os.environ)
    result["PYTHONPATH"] = str(CODE) + os.pathsep + result.get("PYTHONPATH", "")
    result["PATH"] = str(BIN) + os.pathsep + result.get("PATH", "")
    result["GW_SURROGATE"] = ""
    return result


def test_cleaner_keeps_narrow_mass_grid_when_requested(tmp_path):
    rows = np.array([
        [-1, 1.2292711, 1.25, 0, 0, 0, 0, 0, 0, 10, .01, 100, 50],
        [-1, 1.2292722, 1.25, 0, 0, 0, 0, 0, 0, 11, .01, 100, 50],
    ])
    path = tmp_path / "ile.dat"
    np.savetxt(path, rows, fmt="%.17g")

    def run(*options):
        proc = subprocess.run([sys.executable, str(CLEAN), *options, str(path)],
                              cwd=tmp_path, env=env(), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode == 0, proc.stderr
        return [line.split() for line in proc.stdout.splitlines() if line.strip()]

    assert len(run()) == 1  # historical five-decimal coalescing
    precise = run("--intrinsic-digits", "12")
    assert len(precise) == 2
    assert sorted(float(row[1]) for row in precise) == [1.2292711, 1.2292722]


def test_basic_builder_applies_precision_to_join_and_unify(tmp_path):
    import lal
    import RIFT.lalsimutils as lsu

    point = lsu.ChooseWaveformParams()
    point.m1, point.m2 = 35 * lal.MSUN_SI, 30 * lal.MSUN_SI
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        lsu.ChooseWaveformParams_array_to_xml([point, point], "grid")
    finally:
        os.chdir(previous)
    (tmp_path / "args_ile.txt").write_text(
        "integrate_likelihood_extrinsic_batchmode --time-marginalization "
        "--vectorized --gpu --srate 4096 --n-eff 50\n")
    (tmp_path / "args_cip_list.txt").write_text(
        "2 --parameter mc --parameter delta_mc --n-output-samples 5000\n")
    (tmp_path / "args_test.txt").write_text("X --always-succeed\n")
    command = [
        sys.executable, str(BUILDER),
        "--ile-n-events-to-analyze", "1", "--input-grid", str(tmp_path / "grid.xml.gz"),
        "--ile-exe", str(BIN / "integrate_likelihood_extrinsic_batchmode"),
        "--ile-args", str(tmp_path / "args_ile.txt"),
        "--cip-args-list", str(tmp_path / "args_cip_list.txt"),
        "--test-args", str(tmp_path / "args_test.txt"),
        "--working-directory", str(tmp_path), "--n-iterations", "2",
        "--n-samples-per-job", "500", "--last-iteration-extrinsic",
        "--last-iteration-extrinsic-samples-per-ile", "200",
        "--clean-ile-intrinsic-digits", "12",
    ]
    proc = subprocess.run(command, cwd=tmp_path, env=env(), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0, proc.stdout[-4000:]
    assert "--intrinsic-digits 12" in (tmp_path / "join.sub").read_text()
    assert "--intrinsic-digits 12" in (tmp_path / "unify.sh").read_text()


# ---------------------------------------------------------------------------
# Hyperpipeline format.  RIFT_HYPERPIPELINE_FORMAT swaps BOTH cleaner passes
# for util_CleanILE_hyperpipeline.py, which spells the same setting --digits.
# Until this landed the join dropped the flag list entirely and the unify
# handed --intrinsic-digits to parse_known_args, which discarded the option and
# left "12" behind as a positional; read_many SKIPS a shard that does not
# exist, so that combination coalesced at five decimals and said nothing.
# ---------------------------------------------------------------------------

HPIP_CLEAN = BIN / "util_CleanILE_hyperpipeline.py"
POSTPROCESS = BIN / "util_ILEdagPostprocess.sh"
MULTI_BUILDER = BIN / "create_event_parameter_pipeline_BasicMultiApproxIteration"

# Two chirp masses that differ in the 7th decimal: distinct points on a narrow
# 3G BNS grid, one point after five-decimal rounding.
MC_A = 1.2292711
MC_B = 1.2292722


def _write_shard(path):
    from RIFT.misc import hyperpipeline_io as hpio
    columns = ("mc", "eta", "lnL", "sigma_lnL")
    hpio.write_table(str(path), columns,
                     [[MC_A, 0.2490, 10.0, 0.01],
                      [MC_B, 0.2490, 11.0, 0.01]])


def _mc_values(path):
    from RIFT.misc import hyperpipeline_io as hpio
    arr, _ = hpio.read_table(str(path))
    return sorted(float(x) for x in arr["mc"])


def test_hyperpipeline_cleaner_accepts_the_builders_spelling(tmp_path):
    """--intrinsic-digits and --digits must mean the same thing here.

    The builders emit --intrinsic-digits because on the legacy path it applies
    to the intrinsic columns only.  Every column of a hyperpipeline shard bar
    lnL/sigma_lnL is intrinsic, so the distinction has no content -- but the
    NAME still has to be accepted, because one argument string is built before
    the format is known.
    """
    shard = tmp_path / "CME_out-0.dat"
    _write_shard(shard)

    def run(*options):
        out = tmp_path / ("out_%d.dat" % len(options))
        proc = subprocess.run([sys.executable, str(HPIP_CLEAN),
                               "--output", str(out), *options, str(shard)],
                              cwd=tmp_path, env=env(), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode == 0, proc.stderr
        return _mc_values(out)

    assert run() == [pytest.approx(1.22927, abs=1e-12)]     # default coalesces
    assert run("--digits", "12") == [MC_A, MC_B]
    assert run("--intrinsic-digits", "12") == [MC_A, MC_B]


def test_hyperpipeline_cleaner_refuses_a_stray_option_value(tmp_path):
    """An unknown VALUED option must not have its value read as a shard.

    parse_known_args is deliberate -- the DAG forwards advanced-physics flags
    this script has no use for.  Its cost is that `--foo 12` drops --foo and
    leaves "12" as a positional, and read_many skips a nonexistent shard
    without raising, so the run finishes with the WRONG precision and a zero
    exit.  That is the failure mode this refusal exists to convert into a stop.
    """
    shard = tmp_path / "CME_out-0.dat"
    _write_shard(shard)
    proc = subprocess.run([sys.executable, str(HPIP_CLEAN),
                           "--output", str(tmp_path / "out.dat"),
                           "--no-such-option", "12", str(shard)],
                          cwd=tmp_path, env=env(), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode != 0, proc.stdout
    assert "'12'" in proc.stderr, proc.stderr
    assert "--no-such-option" in proc.stderr, proc.stderr


def test_postprocess_forwards_precision_on_the_hyperpipeline_join(tmp_path):
    """The join is a shell script, and it selects the cleaner by env var.

    Asserted by RUNNING it both ways rather than by reading its source: the
    legacy branch forwarded the flag list and the hyperpipeline branch did not,
    which no test of util_CleanILE.py alone can see.
    """
    shards = tmp_path / "iteration_0_ile"
    shards.mkdir()
    _write_shard(shards / "CME_out-0.dat")
    (shards / "command-single.sh").write_text("# nothing to record\n")

    hp_env = env()
    hp_env["RIFT_HYPERPIPELINE_FORMAT"] = "1"
    # The cleaner's shebang is `env python`.  CI symlinks python3 to
    # /usr/bin/python; an IGWN conda interpreter has no `python` on PATH at
    # all, and without this the join fails.  Point `python` at the interpreter
    # running the test.
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "python").symlink_to(sys.executable)
    hp_env["PATH"] = str(shim) + os.pathsep + hp_env["PATH"]

    def join(base, *options):
        proc = subprocess.run(["bash", str(POSTPROCESS), str(shards), base, *options],
                              cwd=tmp_path, env=hp_env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode == 0, proc.stderr
        return _mc_values(tmp_path / (base + ".composite"))

    assert join("default") == [pytest.approx(1.22927, abs=1e-12)]
    assert join("precise", "--intrinsic-digits", "12") == [MC_A, MC_B]


def test_multi_builder_applies_precision_to_all_three_cleaner_passes(tmp_path):
    """The multi-approximant builder cleans THREE times, and they must agree.

    join, the pooled unify (all.net) and the per-model unify all key on the
    intrinsic coordinates.  A pass that rounds harder than its neighbours
    merges points the others kept apart -- and here those coordinates are also
    what pairs one model's row with another's, so a disagreement changes the
    model marginalization, not just the grid.
    """
    import lal
    import RIFT.lalsimutils as lsu

    points = []
    for i in range(4):
        p = lsu.ChooseWaveformParams()
        p.m1, p.m2 = (10 + i) * lal.MSUN_SI, 8 * lal.MSUN_SI
        points.append(p)
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        lsu.ChooseWaveformParams_array_to_xml(points, "proposed-grid")
    finally:
        os.chdir(previous)
    (tmp_path / "args_ile.txt").write_text(
        "integrate_likelihood_extrinsic_batchmode --fmin-template 20.0 "
        "--n-max 100 --n-eff 17 --time-marginalization --approx placeholder\n")
    # One CIP version per iteration: the builder indexes this list by iteration
    # and raises IndexError with fewer.
    (tmp_path / "args_cip_list.txt").write_text(
        "1 --no-plots --fit-method rf --parameter mc --parameter delta_mc "
        "--n-output-samples 5\n"
        "1 --no-plots --fit-method rf --parameter mc --parameter delta_mc "
        "--n-output-samples 5 --posterior-unique-draw\n")
    (tmp_path / "args_test.txt").write_text("X --method lame --parameter mc\n")

    proc = subprocess.run(
        [sys.executable, str(MULTI_BUILDER),
         "--approx", "IMRPhenomXPHM", "--approx", "SEOBNRv4PHM",
         "--input-grid", "proposed-grid.xml.gz",
         "--ile-exe", str(BIN / "integrate_likelihood_extrinsic_batchmode"),
         "--ile-args", str(tmp_path / "args_ile.txt"),
         "--cip-args-list", "args_cip_list.txt", "--test-args", "args_test.txt",
         "--ile-n-events-to-analyze", "2", "--n-samples-per-job", "5",
         "--working-directory", str(tmp_path),
         "--n-iterations", "2", "--n-copies", "1",
         "--clean-ile-intrinsic-digits", "12"],
        cwd=tmp_path, env=env(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0, proc.stdout[-4000:]

    for name in ("join.sub", "unify.sh", "unify_model.sh"):
        assert "--intrinsic-digits 12" in (tmp_path / name).read_text(), name


def test_both_passes_must_be_told_or_the_chain_loses_the_grid(tmp_path):
    """Two passes run over the same points: the coarser one decides.

    util_CleanILE writes with Python's repr, so the intermediate .composite
    carries every digit the join kept.  That is what makes a 5-decimal unify
    silently undo a 12-decimal join -- the precision is present in the file and
    thrown away at read.  Pinned as an end-to-end chain rather than per pass,
    because either pass alone looks correct.
    """
    rows = np.array([
        [-1, MC_A, 1.25, 0, 0, 0, 0, 0, 0, 10, .01, 100, 50],
        [-1, MC_B, 1.25, 0, 0, 0, 0, 0, 0, 11, .01, 100, 50],
    ])
    ile = tmp_path / "ile.dat"
    np.savetxt(ile, rows, fmt="%.17g")

    def clean(source, *options):
        proc = subprocess.run([sys.executable, str(CLEAN), *options, str(source)],
                              cwd=tmp_path, env=env(), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode == 0, proc.stderr
        return [line.split() for line in proc.stdout.splitlines() if line.strip()]

    joined = clean(ile, "--intrinsic-digits", "12")
    assert len(joined) == 2
    composite = tmp_path / "joined.composite"
    composite.write_text("".join(" ".join(row) + "\n" for row in joined))

    unified = clean(composite, "--intrinsic-digits", "12")
    assert sorted(float(r[1]) for r in unified) == [MC_A, MC_B]

    # and the negative half: the join alone does not save the grid.
    collapsed = clean(composite)
    assert len(collapsed) == 1, "a default unify should still coalesce"
