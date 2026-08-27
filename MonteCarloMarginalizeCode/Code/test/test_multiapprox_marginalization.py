"""Cross-model marginalization in the multi-approximant workflow.

Two kinds of check, deliberately separated:

* the ARITHMETIC of the combination, against hand-computed values.  This is the
  science: L_marg(lambda) = sum_m p(m) L_m(lambda), linear in L.
* the SHAPE of the emitted DAG.  The workflow merges in the iteration loop (one
  shared grid, one marginalized fit) and forks at the terminal stage (per-model
  posterior and evidence, recombined by p(m) Z_m).

Neither validates the inference on data; see
RIFT/misc/DESIGN_multiapprox_marginalization.md.

The DAG tests build with TWO approximants on purpose.  With one, every
cross-model path is vacuously satisfied -- which is how this builder kept a
severed grid handoff, an approximant-blind extrinsic stage, and a submit file
naming a directory that was never created.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
BIN = CODE / "bin"
CLEANILE = BIN / "util_CleanILE.py"
COMBINE = BIN / "util_CombineApproximantPosteriors.py"
BUILDER = BIN / "create_event_parameter_pipeline_BasicMultiApproxIteration"
MODEL_RX = r"approx_(.+?)_consolidated"


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CODE) + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = str(BIN) + os.pathsep + env.get("PATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["GW_SURROGATE"] = ""
    return env


def _run(args, cwd):
    return subprocess.run([sys.executable] + [str(a) for a in args], cwd=str(cwd),
                          env=_env(), text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


def _row(m1, m2, lnL, sigma=0.01, ntot=1000.0):
    """One 13-column ILE row: indx m1 m2 s1x..s2z lnL sigmaOverL ntot neff."""
    return [-1, m1, m2, 0., 0., 0.1, 0., 0., 0.2, lnL, sigma, ntot, 100.]


def _composite(path, rows):
    np.savetxt(str(path), np.array(rows))


def _lnL_column(stdout):
    return [float(line.split()[9]) for line in stdout.strip().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------

@pytest.fixture
def two_models(tmp_path):
    """One shared intrinsic point; model A has TWO replicas, model B has one.

    This is the case that separates a correct marginalization from flat
    pooling: flat pooling weights by replica COUNT, so A gets 2/3 instead of
    1/2.  With one replica each the two agree, which is why that arrangement
    would not have caught the defect.
    """
    _composite(tmp_path / "approx_MODELA_consolidated_0.composite",
               [_row(10., 8., 100.0), _row(10., 8., 100.0)])
    _composite(tmp_path / "approx_MODELB_consolidated_0.composite",
               [_row(10., 8., 104.0)])
    return tmp_path


def test_flat_pooling_uses_replica_counts_as_model_weights(two_models):
    """Without --model-group-regex the models are weighted by how many times
    each was evaluated.  Pinned because it is the wrong answer we are moving
    away from, and its size is the reason the change matters."""
    out = _run([CLEANILE] + sorted(str(p) for p in two_models.glob("*.composite")), two_models)
    assert out.returncode == 0, out.stderr
    flat = _lnL_column(out.stdout)[0]
    expected = np.log((2 * np.exp(100.0) + np.exp(104.0)) / 3.0)
    assert flat == pytest.approx(expected, abs=1e-9)


def test_model_aware_combination_is_marginalization(two_models):
    """L_marg = sum_m p(m) L_m, uniform p(m) -- linear in L, not in lnL."""
    out = _run([CLEANILE, "--model-group-regex", MODEL_RX]
               + sorted(str(p) for p in two_models.glob("*.composite")), two_models)
    assert out.returncode == 0, out.stderr
    got = _lnL_column(out.stdout)[0]
    expected = np.log(0.5 * np.exp(100.0) + 0.5 * np.exp(104.0))
    assert got == pytest.approx(expected, abs=1e-9)

    # and it must differ from flat pooling by a real amount, or the flag is
    # cosmetic
    flat = np.log((2 * np.exp(100.0) + np.exp(104.0)) / 3.0)
    assert got - flat > 0.3


def test_geometric_mean_is_not_what_we_compute(two_models):
    """Averaging lnL (a logarithmic opinion pool) is NOT model marginalization.

    Guards against someone 'simplifying' the combination to a mean of lnL.
    """
    out = _run([CLEANILE, "--model-group-regex", MODEL_RX]
               + sorted(str(p) for p in two_models.glob("*.composite")), two_models)
    got = _lnL_column(out.stdout)[0]
    geometric = 0.5 * 100.0 + 0.5 * 104.0
    assert abs(got - geometric) > 0.2


def test_model_prior_reweights_the_mixture(two_models):
    out = _run([CLEANILE, "--model-group-regex", MODEL_RX,
                "--model-prior", "MODELA=0.3", "--model-prior", "MODELB=0.7"]
               + sorted(str(p) for p in two_models.glob("*.composite")), two_models)
    assert out.returncode == 0, out.stderr
    got = _lnL_column(out.stdout)[0]
    expected = np.log(0.3 * np.exp(100.0) + 0.7 * np.exp(104.0))
    assert got == pytest.approx(expected, abs=1e-9)


def test_partial_model_prior_is_refused(two_models):
    """Half-specified weights would silently default the rest to 1.0."""
    out = _run([CLEANILE, "--model-group-regex", MODEL_RX, "--model-prior", "MODELA=0.3"]
               + sorted(str(p) for p in two_models.glob("*.composite")), two_models)
    assert out.returncode != 0
    assert "missing weights" in out.stderr


def test_one_model_reduces_to_the_flat_pool(two_models):
    """Enabling the flag must never change a single-model result."""
    only_a = [str(two_models / "approx_MODELA_consolidated_0.composite")]
    plain = _run([CLEANILE] + only_a, two_models)
    aware = _run([CLEANILE, "--model-group-regex", MODEL_RX] + only_a, two_models)
    assert plain.returncode == 0 and aware.returncode == 0, aware.stderr
    assert plain.stdout == aware.stdout


def test_partial_coverage_is_reported_and_can_be_dropped(tmp_path):
    """A point missing one model is marginalized over the subset, so the
    estimator changes point to point.  That must never be silent."""
    _composite(tmp_path / "approx_MODELA_consolidated_0.composite",
               [_row(10., 8., 100.), _row(12., 9., 101.), _row(14., 7., 99.)])
    _composite(tmp_path / "approx_MODELB_consolidated_0.composite",
               [_row(10., 8., 104.), _row(14., 7., 98.)])
    files = sorted(str(p) for p in tmp_path.glob("*.composite"))

    warned = _run([CLEANILE, "--model-group-regex", MODEL_RX] + files, tmp_path)
    assert warned.returncode == 0, warned.stderr
    assert len(_lnL_column(warned.stdout)) == 3
    assert "WARNING" in warned.stderr and "SUBSET" in warned.stderr

    dropped = _run([CLEANILE, "--model-group-regex", MODEL_RX,
                    "--require-all-models"] + files, tmp_path)
    assert dropped.returncode == 0, dropped.stderr
    assert len(_lnL_column(dropped.stdout)) == 2
    assert "DROPPED 1" in dropped.stderr


def test_unlabelled_input_is_refused(tmp_path):
    """An unmatched filename would be pooled as a nameless extra model."""
    _composite(tmp_path / "stray.composite", [_row(10., 8., 100.)])
    out = _run([CLEANILE, "--model-group-regex", MODEL_RX,
                str(tmp_path / "stray.composite")], tmp_path)
    assert out.returncode != 0
    assert "does not match" in out.stderr


# --------------------------------------------------------------------------
# the final mixture
# --------------------------------------------------------------------------

def _posterior(path, mean, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    dat = np.column_stack([rng.normal(mean, 1.0, n), rng.normal(0.0, 1.0, n)])
    with open(str(path), "w") as f:
        f.write("# m1 m2\n")
        np.savetxt(f, dat)


def _annotation(path, ln_z):
    with open(str(path), "w") as f:
        f.write("# lnL sigma_lnL neff\n{!r} 0.01 100\n".format(ln_z))


@pytest.fixture
def two_posteriors(tmp_path):
    _posterior(tmp_path / "post_A.dat", 10.0, seed=7)
    _posterior(tmp_path / "post_B.dat", 20.0, seed=8)
    _annotation(tmp_path / "annot_A.dat", 100.0)
    _annotation(tmp_path / "annot_B.dat", 102.0)
    return tmp_path


def _combine(tmp_path, extra=()):
    return _run([COMBINE,
                 "--model", "A:{}/post_A.dat:{}/annot_A.dat".format(tmp_path, tmp_path),
                 "--model", "B:{}/post_B.dat:{}/annot_B.dat".format(tmp_path, tmp_path),
                 "--output", "{}/out.dat".format(tmp_path), "--seed", "1"] + list(extra),
                tmp_path)


def test_mixture_is_weighted_by_evidence(two_posteriors):
    out = _combine(two_posteriors)
    assert out.returncode == 0, out.stderr
    w = np.exp(np.array([100.0, 102.0]) - 102.0)
    w = w / w.sum()
    combined = np.genfromtxt(str(two_posteriors / "out.dat"), comments="#")
    # the mixture mean is the weighted mean of the two component means
    assert combined[:, 0].mean() == pytest.approx(w[0] * 10.0 + w[1] * 20.0, abs=0.15)


def test_mixture_prior_multiplies_the_evidence(two_posteriors):
    """p(m) is a prior; Z_m still multiplies it.  Passing --model-prior does
    NOT give a prior-weighted mixture, and that is intended."""
    out = _combine(two_posteriors, ["--model-prior", "A=0.9", "--model-prior", "B=0.1"])
    assert out.returncode == 0, out.stderr
    w = np.exp(np.array([100.0, 102.0]) - 102.0) * np.array([0.9, 0.1])
    w = w / w.sum()
    combined = np.genfromtxt(str(two_posteriors / "out.dat"), comments="#")
    assert combined[:, 0].mean() == pytest.approx(w[0] * 10.0 + w[1] * 20.0, abs=0.15)


def test_mismatched_columns_are_refused(two_posteriors):
    """Mixing posteriors with different columns would produce garbage."""
    with open(str(two_posteriors / "post_B.dat")) as f:
        body = f.read().split("\n", 1)[1]
    with open(str(two_posteriors / "post_B.dat"), "w") as f:
        f.write("# m1 chi_eff\n" + body)
    out = _combine(two_posteriors)
    assert out.returncode != 0
    assert "different column header" in out.stderr


# --------------------------------------------------------------------------
# the emitted DAG
# --------------------------------------------------------------------------

def _dag_facts(rundir):
    dag = next(Path(rundir).glob("*.dag"))
    jobs, macros, parents = {}, {}, {}
    for line in dag.read_text().splitlines():
        parts = line.split()
        if line.startswith("JOB"):
            jobs[parts[1]] = parts[2]
        elif line.startswith("VARS"):
            macros[parts[1]] = dict(re.findall(r'(\w+)="([^"]*)"', line))
        elif line.startswith("PARENT"):
            cut = parts.index("CHILD")
            for child in parts[cut + 1:]:
                parents.setdefault(child, set()).update(parts[1:cut])
    return jobs, macros, parents


@pytest.fixture(scope="module")
def multiapprox_rundir(tmp_path_factory):
    """Build a two-approximant DAG from synthetic inputs.

    Self-contained on purpose: this builder has no caller (pseudo_pipe's
    --pipeline-builder does not offer it), so the test must supply the
    args_*.txt and grid a user would.
    """
    pytest.importorskip("RIFT.lalsimutils")
    rundir = tmp_path_factory.mktemp("multiapprox")
    (rundir / "args_ile.txt").write_text(
        "--fmin-template 20.0 --n-max 100 --approx placeholder\n")
    (rundir / "args_cip_list.txt").write_text(
        "1   --no-plots --fit-method rf --parameter mc --parameter delta_mc --n-output-samples 5\n"
        "1   --no-plots --fit-method rf --parameter mc --parameter delta_mc --n-output-samples 5\n")
    (rundir / "args_test.txt").write_text("--method lame --parameter mc --always-succeed\n")
    # Plotting ON.  It is off by default, which is why an unresolved
    # $(macroapprox) sat in the plot job's log paths through a whole review
    # cycle: the plot job is model-independent and its node binds only the
    # iteration macros, so a model-tagged log path can never resolve.  A stage
    # that is not built is a stage no assertion can check.
    (rundir / "args_plot.txt").write_text("--parameter mc --parameter eta\n")

    grid = _run(["-c",
                 "import RIFT.lalsimutils as u;"
                 "P=[];\n"
                 "import numpy as np\n"
                 "for i in range(4):\n"
                 "    p=u.ChooseWaveformParams(); p.m1=(10+i)*u.lsu_MSUN; p.m2=8*u.lsu_MSUN; P.append(p)\n"
                 "u.ChooseWaveformParams_array_to_xml(P,'proposed-grid')\n"], rundir)
    if grid.returncode:
        pytest.skip("cannot build a seed grid: {}".format(grid.stderr[-400:]))

    build = _run([BUILDER,
                  "--approx", "IMRPhenomXPHM", "--approx", "SEOBNRv4PHM",
                  "--input-grid", "proposed-grid.xml.gz",
                  "--ile-exe", str(BIN / "integrate_likelihood_extrinsic_batchmode"),
                  "--ile-args", str(rundir / "args_ile.txt"),
                  "--cip-args-list", "args_cip_list.txt",
                  "--test-args", "args_test.txt",
                  "--ile-n-events-to-analyze", "2", "--n-samples-per-job", "2",
                  "--request-memory-CIP", "4096", "--request-memory-ILE", "4096",
                  "--working-directory", str(rundir),
                  "--n-iterations", "2", "--n-copies", "1",
                  "--last-iteration-extrinsic",
                  "--last-iteration-extrinsic-nsamples", "4",
                  "--plot-args", str(rundir / "args_plot.txt")], rundir)
    if build.returncode:
        pytest.fail("builder failed:\n{}".format(build.stdout[-3000:]))
    return rundir


def test_every_model_reads_one_shared_grid(multiapprox_rundir):
    """The loop ILE grid must NOT be per model, or nothing is marginalized."""
    sub = (multiapprox_rundir / "ILE.sub").read_text()
    grid = re.search(r"--sim-xml\s+(\S+)", sub)
    assert grid, "no --sim-xml in ILE.sub"
    assert "$(macroapprox)" not in grid.group(1), grid.group(1)

    unify = (multiapprox_rundir / "unify.sh").read_text()
    assert "--model-group-regex" in unify, (
        "unify.sh pools composites without --model-group-regex, so replica "
        "counts act as model weights")


def test_the_loop_fits_once_per_iteration(multiapprox_rundir):
    jobs, macros, parents = _dag_facts(multiapprox_rundir)
    models = {macros.get(n, {}).get("macroapprox") for n, s in jobs.items()
              if s.endswith("ILE.sub")}
    models.discard(None)
    assert len(models) == 2, sorted(models)

    for node, sub in jobs.items():
        if sub.startswith("CIP") and not sub.startswith("CIP_terminal"):
            assert "macroapprox" not in macros.get(node, {}), (
                "in-loop CIP is per model; it must fit the marginalized net once")

    for node, sub in jobs.items():
        if sub.endswith("unify.sub"):
            waited = {macros.get(p, {}).get("macroapprox") for p in parents.get(node, ())
                      if jobs.get(p, "").endswith("join.sub")}
            waited.discard(None)
            assert waited == models, (
                "unify waits on {}, not every model {}".format(sorted(waited), sorted(models)))


def test_the_terminal_stage_forks_and_recombines(multiapprox_rundir):
    jobs, macros, parents = _dag_facts(multiapprox_rundir)
    models = {macros.get(n, {}).get("macroapprox") for n, s in jobs.items()
              if s.endswith("ILE.sub")}
    models.discard(None)

    for suffix in ("CIP_terminal.sub", "cat.sub", "ILE_extr.sub"):
        got = {macros.get(n, {}).get("macroapprox") for n, s in jobs.items()
               if s.endswith(suffix)}
        got.discard(None)
        assert got == models, "{} covers {}, not {}".format(suffix, sorted(got), sorted(models))

    combine = [n for n, s in jobs.items() if s.endswith("combine_models.sub")]
    assert len(combine) == 1
    cats = {macros.get(p, {}).get("macroapprox") for p in parents.get(combine[0], ())
            if jobs.get(p, "").endswith("cat.sub")}
    cats.discard(None)
    assert cats == models, "the mixture is built from {}, not {}".format(sorted(cats), sorted(models))


def test_extrinsic_stage_reads_the_grid_the_run_finished_on(multiapprox_rundir):
    """The off-by-one this builder shipped: the terminal ILE read
    overlap-grid-<n-1> while the final CIP wrote overlap-grid-<n>."""
    jobs, macros, _ = _dag_facts(multiapprox_rundir)
    extrinsic = {macros.get(n, {}).get("macroiteration") for n, s in jobs.items()
                 if s.endswith("ILE_extr.sub")}
    written = {macros.get(n, {}).get("macroiterationnext") for n, s in jobs.items()
               if s.endswith("join_grids.sub") or s.startswith("CIP")}
    extrinsic.discard(None)
    written.discard(None)
    assert len(extrinsic) == 1, sorted(extrinsic)
    assert next(iter(extrinsic)) == max(written, key=int)


def test_every_job_directory_exists(multiapprox_rundir):
    """A submit file naming a directory the builder never created holds the job
    on the execute node, and no DAG-shape assertion sees it.  An unresolved
    $(macro) is a failure too: that is how ILE_extr came to interpolate an
    empty approximant into both --approx and its initialdir.

    initialdir IS a directory; output/error/log are files whose directory must
    exist.  Taking dirname of both silently checks initialdir's parent.
    """
    jobs, macros, _ = _dag_facts(multiapprox_rundir)
    cache, problems = {}, []
    for node, submit in jobs.items():
        if submit not in cache:
            text = (multiapprox_rundir / submit).read_text()
            cache[submit] = (
                [(v, True) for v in re.findall(r"^initialdir\s*=\s*(\S+)", text, re.M)]
                + [(v, False) for v in re.findall(r"^(?:output|error|log)\s*=\s*(\S+)", text, re.M)])
        for raw, is_dir in cache[submit]:
            resolved = raw
            for key, value in macros.get(node, {}).items():
                resolved = resolved.replace("$({})".format(key), value)
            resolved = re.sub(r"\$\((?:cluster|process|macromassid)\)", "X", resolved)
            if "$(" in resolved:
                problems.append("{}: unresolved macro in {}".format(submit, resolved))
                continue
            target = resolved if is_dir else os.path.dirname(resolved)
            if target and not os.path.isdir(target):
                problems.append("{}: missing directory {}".format(submit, target))
    assert not problems, "\n  ".join(sorted(set(problems)))
