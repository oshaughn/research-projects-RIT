"""Known-answer mixtures, hostile contracts and the existing-run CLI."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scipy.special import logsumexp

CODE = Path(__file__).resolve().parents[1]
# Keep numerical tests independent of RIFT's optional waveform imports.
spec = importlib.util.spec_from_file_location("compression", CODE/"RIFT/compression/__init__.py", submodule_search_locations=[str(CODE/"RIFT/compression")])
c = importlib.util.module_from_spec(spec)
sys.modules["compression"] = c
spec.loader.exec_module(c)
from compression.config import resolve
from compression.cli import run
from compression.model import CompressedFit, fit_posterior


def configuration(tmp_path, distance="Euclidean", cosmology="Planck15"):
    ini = tmp_path/"production.ini"
    ini.write_text(f"""[engine]
chirpmass-min = 10
chirpmass-max = 60
q-min = .1
q-max = 1
distance-min = 100
distance-max = 2500
cosmology = {cosmology}
[rift-pseudo-pipe]
assume-precessing = True
ile-distance-prior = {distance}
spin-magnitude-prior = volumetric
[rift-auto-fit]
enabled = true
targets = aligned_distance, precessing_source_redshift
component_counts = 1,2
weight_column = posterior_weight
group_column = intrinsic_group
validation_ks = .08
""")
    (tmp_path/"args_ile.txt").write_text(f"X --d-min 100 --d-max 2500 --d-prior {distance} --reference-freq 20")
    (tmp_path/"args_cip_list.txt").write_text("2 X --parameter mc --parameter eta --parameter s1x --parameter s1y --parameter s1z --parameter s2x --parameter s2y --parameter s2z --mc-range [10,60] --eta-range [.08,.25] --chi-max .99")
    return ini


def known_likelihood(n=18000, dimension=9, seed=91):
    """Draw the known likelihood proposal, not the fitted posterior model."""
    rng = np.random.default_rng(seed)
    mean = np.array([30., .5, .12, -.1, .2, -.1, .15, -.15, 1000.])
    widths = np.array([1.5, .025, .04, .04, .05, .04, .04, .05, 80.])
    if dimension == 5:
        ix = [0, 1, 4, 7, 8]
        mean, widths = mean[ix], widths[ix]
    d = len(mean)
    correlation = np.eye(d)
    correlation[0, -1] = correlation[-1, 0] = .45
    correlation[1, 2] = correlation[2, 1] = -.25
    cov = correlation*widths[:, None]*widths[None, :]
    means = np.stack([mean.copy(), mean.copy()])
    means[1, 0] += 5.
    means[1, -1] += 160.
    means[1, 2] -= .12
    labels = rng.choice(2, n, p=[.65, .35])
    chart = means[labels]+rng.multivariate_normal(np.zeros(d), cov, n)
    return chart, means, cov


def truth_logpdf(x, means, cov):
    from scipy.stats import multivariate_normal
    return logsumexp(np.stack([np.log(w)+multivariate_normal.logpdf(x, m, cov) for w, m in zip([.65, .35], means)]), axis=0)


@pytest.mark.parametrize("dimension,distance_prior", [(5, "Euclidean"), (9, "Euclidean"), (5, "cosmo_sourceframe")])
def test_known_gmm_likelihood_and_source_frames(tmp_path, dimension, distance_prior):
    ini = configuration(tmp_path, distance_prior)
    _, contract, _ = resolve(ini, tmp_path)
    target = "aligned_distance" if dimension == 5 else "precessing_source_redshift"
    x, means, cov = known_likelihood(dimension=dimension)
    template = dict(schema="rift-posterior-compression-v1", target=target, contract=contract,
        cosmology_yaml=c.CosmologyContext().serialized,
        mixture=dict(weights=[1.], means=[means[0].tolist()], covariances=[cov.tolist()]))
    prior = CompressedFit(template)
    w = np.exp(prior.log_native_prior_chart(x)-np.max(prior.log_native_prior_chart(x)))
    points = x.copy()
    points[:, 1] = (1-x[:, 1]**2)/4
    model, report = fit_posterior(points, w, np.arange(len(w)), contract, target, components=[1, 2], seed=33, maxiter=300)
    assert report["status"] == "pass_native_recovery", report
    probes, _, _ = known_likelihood(4000, dimension, seed=123456)
    physical = probes.copy()
    physical[:, 1] = (1-probes[:, 1]**2)/4
    estimated = model.log_likelihood(physical, mass_frame="detector", radial_coordinate="luminosity_distance")
    exact = truth_logpdf(probes, means, cov)
    residual = estimated-exact
    residual -= np.median(residual)
    assert np.sqrt(np.mean(residual**2)) < .25
    for cosmology in ("Planck15", "Planck18"):
        ctx = c.CosmologyContext(cosmology, 2500)
        source = physical.copy()
        z = ctx.redshift(source[:, -1])
        source[:, 0] /= 1+z
        source[:, -1] = z
        np.testing.assert_allclose(model.log_likelihood(source, mass_frame="source", radial_coordinate="redshift", cosmology=ctx), estimated, atol=1e-6)
    # Counterexample: fitting one broad component must fail the declared shape gate.
    _, bad_report = fit_posterior(points, w, np.arange(len(w)), contract, target, components=[1], seed=33, tolerance=.025)
    assert bad_report["status"] == "failed_validation"


def test_inheritance_and_conflicts(tmp_path):
    ini = configuration(tmp_path, "cosmo_sourceframe")
    _, contract, _ = resolve(ini, tmp_path)
    assert contract["cosmology"] == "Planck15"
    assert contract["distance_prior"] == "cosmo_sourceframe"
    ini.write_text(ini.read_text()+"distance_prior = Euclidean\n")
    with pytest.raises(ValueError, match="conflicts"):
        resolve(ini, tmp_path)


def test_wrong_cosmology_and_custom_priors_refused(tmp_path):
    ini = configuration(tmp_path, "cosmo_sourceframe", "Planck18")
    with pytest.raises(ValueError, match="Planck15"):
        resolve(ini, tmp_path)


@pytest.mark.parametrize("coordinates,extra,prior", [
    ("--parameter s1z --parameter s2z", "", "uniform_aligned"),
    ("--parameter s1z --parameter s2z", "--aligned-prior alignedspin-zprior", "zprior_aligned"),
    ("--parameter chi1 --parameter chi2 --parameter cos_theta1 --parameter cos_theta2 --parameter phi1 --parameter phi2", "", "uniform_mag_prec"),
    ("--parameter s1z_bar --parameter s2z_bar --parameter chi1_perp_u --parameter chi2_perp_u --parameter phi1 --parameter phi2", "--prior-in-integrand-correction uniform_over_rbar_singular", "uniform_mag_prec"),
])
def test_actual_sampling_spin_priors(tmp_path, coordinates, extra, prior):
    ini = configuration(tmp_path)
    (tmp_path/"args_cip_list.txt").write_text("1 X --parameter mc --parameter eta "+coordinates+" "+extra)
    requested = {"uniform_aligned":"volumetric", "zprior_aligned":"zprior_aligned", "uniform_mag_prec":"uniform_mag_prec"}[prior]
    ini.write_text(ini.read_text().replace("spin-magnitude-prior = volumetric", 'spin-magnitude-prior = "'+requested+'"'))
    if prior in ("uniform_aligned", "zprior_aligned"):
        ini.write_text(ini.read_text().replace("assume-precessing = True", "assume-precessing = False"))
    _, contract, _ = resolve(ini, tmp_path)
    assert contract["spin_prior"] == prior


def test_native_mass_support_and_missing_records(tmp_path):
    ini = configuration(tmp_path)
    ini.write_text(ini.read_text().replace("assume-precessing = True", "assume-precessing = False"))
    (tmp_path/"args_cip_list.txt").write_text("1 X --parameter mtot --parameter q --mtot-range [40,80] --parameter s1z --parameter s2z")
    _, contract, _ = resolve(ini, tmp_path)
    assert contract["cuts"]["mtot"] == [40, 80]
    (tmp_path/"args_cip_list.txt").write_text("1 X --parameter m1 --parameter m2 --parameter s1z --parameter s2z")
    assert resolve(ini, tmp_path)[1]["cuts"]["m2"] == [.9,1000]
    (tmp_path/"args_ile.txt").unlink()
    with pytest.raises(ValueError, match="executed"):
        resolve(ini, tmp_path)


def test_real_ile_pin_and_reference_frequency(tmp_path):
    ini = configuration(tmp_path)
    with (tmp_path/"args_ile.txt").open("a") as stream:
        stream.write(" --reference-freq 43")
    assert resolve(ini, tmp_path)[1]["fref"] == 43
    with (tmp_path/"args_ile.txt").open("a") as stream:
        stream.write(" --pin-distance-to-sim")
    with pytest.raises(ValueError, match="Pinned"):
        resolve(ini, tmp_path)
    ini = configuration(tmp_path)
    with (tmp_path/"args_cip_list.txt").open("a") as f:
        f.write(" --supplementary-prior-code unknown")
    with pytest.raises(ValueError, match="Unsupported retained-prior"):
        resolve(ini, tmp_path)


def test_nonflat_cosmology_derivative():
    from astropy.cosmology import LambdaCDM
    ctx = c.CosmologyContext(LambdaCDM(H0=71, Om0=.25, Ode0=.6, Tcmb0=2.7), 15000)
    z = np.linspace(.01, 1, 25)
    numerical = (ctx.cosmology.luminosity_distance(z+1e-5).value-ctx.cosmology.luminosity_distance(z-1e-5).value)/2e-5
    np.testing.assert_allclose(ctx.distance_derivative(z), numerical, rtol=2e-6)
    np.testing.assert_allclose(ctx.redshift(ctx.luminosity_distance(z)), z, atol=1e-8)


def test_native_distance_prior_matches_rift(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(CODE)+os.pathsep+os.environ.get("PYTHONPATH", ""))
    script = """
import numpy as np
import importlib.util
from pathlib import Path
import RIFT
from RIFT.compression import CosmologyContext
spec = importlib.util.spec_from_file_location('distance_prior_reference', Path(RIFT.__file__).parent/'likelihood/jax_ile/distance_prior.py')
reference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_module)
cosmo_distance_prior = reference_module.cosmo_distance_prior
from RIFT.likelihood.priors_utils import dist_prior_pseudo_cosmo
context = CosmologyContext('Planck15', 10000)
distance = np.geomspace(1., 9000., 100)
for kind in ('cosmo', 'cosmo_sourceframe'):
    actual = context.distance_logprior(distance, kind)
    reference = cosmo_distance_prior(kind).log_density_unnormalized(distance)
    np.testing.assert_allclose(actual-actual[0], reference-reference[0], atol=2e-5)
actual = context.distance_logprior(distance, 'pseudo_cosmo')
reference = np.log(dist_prior_pseudo_cosmo(distance))
np.testing.assert_allclose(actual-actual[0], reference-reference[0], atol=1e-12)
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stdout+proc.stderr


def test_fake_ile_existing_run_cli(tmp_path):
    ini = configuration(tmp_path)
    fake = CODE/"demo/auto_fit/fake_ile.py"
    proc = subprocess.run([sys.executable, str(fake), "--output", str(tmp_path/"extrinsic_posterior_samples.dat"), "--rows", "18000"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not list(tmp_path.glob("posterior_samples-*.dat"))
    assert not list(tmp_path.glob("overlap-grid-*"))
    ini.write_text(ini.read_text()+"build-diagnostics = true\n")
    result = run(ini, tmp_path)
    assert not list((tmp_path/"auto_fit").glob("*/diagnostics"))
    assert result["status"] == "complete", result
    for target in ("aligned_distance", "precessing_source_redshift"):
        loaded = CompressedFit.load(tmp_path/"auto_fit"/target/"model.json")
        assert loaded.dimension == (5 if target == "aligned_distance" else 9)
    probes, means, covariance = known_likelihood(4000, 9, seed=37372)
    points = probes.copy()
    points[:, 1] = (1-probes[:, 1]**2)/4
    recovered = loaded.log_likelihood(points, mass_frame="detector", radial_coordinate="luminosity_distance")
    residual = recovered-truth_logpdf(probes, means, covariance)
    residual -= np.median(residual)
    rms = float(np.sqrt(np.mean(residual**2)))
    assert rms < .3
    (tmp_path/"known_answer_cli.json").write_text(json.dumps(dict(log_likelihood_shape_rms=rms,
        maximum_ks={k:v["maximum_ks"] for k,v in result["reports"].items()})))
    env = dict(os.environ, PYTHONPATH=str(CODE)+os.pathsep+os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, str(CODE/"bin/util_RIFT_compress_posterior.py"),
        "--config", str(ini), "--rundir", str(tmp_path), "--output", "standalone"],
        capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    assert json.loads((tmp_path/"standalone/manifest.json").read_text())["status"] == "complete"
    from compression.diagnostics import compare
    from compression.config import file_hash
    for target in result["settings"]["targets"]:
        directory = tmp_path/"auto_fit"/target/"diagnostics"
        samples = directory/"posterior_samples.dat"
        proc = subprocess.run([sys.executable, str(CODE/"bin/util_RIFT_sample_compressed.py"),
            "--model", str(directory.parent/"model.json"), "--output", str(samples),
            "--n-samples", "5000"], capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0, proc.stdout+proc.stderr
        proc = subprocess.run([sys.executable, str(CODE/"bin/util_RIFT_compare_compressed.py"),
            "--fit-directory", str(directory.parent.parent), "--target", target, "--samples", str(samples),
            "--output", str(directory/"js.json"), "--bins", "32"],
            capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0, proc.stdout+proc.stderr
        report = json.loads((directory/"js.json").read_text())
        assert report["maximum_js_bits"] < .04, report
        assert report["reference_rows"] < 18000
        assert report["posterior_rows"] == 5000
        data_columns = np.genfromtxt(samples, names=True).dtype.names
        assert {"a1z", "a2z", "chi_eff", "distance"} <= set(data_columns)
        assert {"mc", "eta", "xi", "dist", "mc_source", "redshift"} <= report["parameters"].keys()
        assert ("chi1_perp" in report["parameters"]) == (target == "precessing_source_redshift")
        assert (directory/"js.tsv").is_file()
        with pytest.raises(FileExistsError):
            compare(directory.parent.parent, target, samples, directory/"js.json")
        # A distorted source-mass marginal must be visible, even if file hashes agree.
        data = np.genfromtxt(samples, names=True)
        data["mc_source"] *= 3
        np.savetxt(samples, np.column_stack([data[k] for k in data.dtype.names]),
                   header=" ".join(data.dtype.names), comments="")
        with pytest.raises(ValueError, match="hashes"):
            compare(directory.parent.parent, target, samples, directory/"bad_hash.json")
        metadata_path = Path(str(samples)+".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["samples_sha256"] = file_hash(samples)
        metadata_path.write_text(json.dumps(metadata))
        bad = compare(directory.parent.parent, target, samples, directory/"distorted.json", bins=32)
        assert bad["parameters"]["mc_source"]["js_bits"] > .8
    with pytest.raises(FileExistsError):
        run(ini, tmp_path)
    # An already exported redshift from a different cosmology must not be trusted.
    table = tmp_path/"extrinsic_posterior_samples.dat"
    text = table.read_text().splitlines()
    table.write_text(text[0]+" redshift\n"+"\n".join(t+" 2.0" for t in text[1:])+"\n")
    with pytest.raises(ValueError, match="cosmology"):
        run(ini, tmp_path, output="wrong_cosmology")
    assert json.loads((tmp_path/"wrong_cosmology/manifest.json").read_text())["status"] == "failed"


def test_group_information_cannot_be_manufactured(tmp_path):
    ini = configuration(tmp_path)
    _, contract, _ = resolve(ini, tmp_path)
    x, _, _ = known_likelihood(2000, 5)
    x[:, 1] = (1-x[:, 1]**2)/4
    with pytest.raises(ValueError, match="100 original"):
        fit_posterior(x, np.ones(len(x)), np.zeros(len(x)), contract, "aligned_distance")


def test_extrinsic_rows_weights_and_groups_are_used_directly(tmp_path, monkeypatch):
    import compression.cli as cli
    ini = configuration(tmp_path)
    fake = CODE/"demo/auto_fit/fake_ile.py"
    table = tmp_path/"extrinsic_posterior_samples.dat"
    subprocess.run([sys.executable, str(fake), "--output", str(table), "--rows", "3000"], check=True)
    names = table.read_text().splitlines()[0]
    data = np.loadtxt(table, skiprows=1)
    data[:, -2] = np.linspace(.2, 2., len(data))
    np.savetxt(table, data, header=names, comments="")
    # An unrelated CIP table must never be opened or used to alter these weights.
    (tmp_path/"posterior_samples-999.dat").write_text("invalid unrelated CIP table\n")
    aligned, full, weights, groups, _ = cli.read_posterior(table, "posterior_weight", "intrinsic_group")
    seen = []
    def capture(points, row_weights, row_groups, contract, target, **kwargs):
        np.testing.assert_array_equal(points, full if target == "precessing_source_redshift" else aligned)
        np.testing.assert_array_equal(row_weights, weights)
        np.testing.assert_array_equal(row_groups, groups)
        seen.append(target)
        class Model:
            artifact = {"test": "direct input contract"}
        return Model(), {"status": "pass_native_recovery"}
    monkeypatch.setattr(cli, "fit_posterior", capture)
    result = cli.run(ini, tmp_path)
    assert result["status"] == "complete"
    assert len(seen) == 2
    assert "final_cip_posterior" not in result


def test_final_cip_spin_prior_not_early_iteration_prior(tmp_path):
    ini = configuration(tmp_path)
    args = tmp_path/"args_cip_list.txt"
    final = args.read_text()
    args.write_text("2 X --parameter mc --parameter eta --transverse-prior taper-down\n"+final)
    (tmp_path/"CIP_prior.sub").write_text('arguments = "'+final.strip()+'"\n')
    assert resolve(ini, tmp_path)[1]["spin_prior"] == "volumetric"


@pytest.mark.parametrize("extra", ["--no-downselect", "--trust-sample-parameter-box"])
def test_unresolved_native_support_refused(tmp_path, extra):
    ini = configuration(tmp_path)
    with (tmp_path/"args_cip_list.txt").open("a") as stream:
        stream.write(" "+extra)
    with pytest.raises(ValueError, match="support"):
        resolve(ini, tmp_path)


def test_inactive_ranges_and_partial_spins(tmp_path):
    ini = configuration(tmp_path)
    ini.write_text(ini.read_text().replace("assume-precessing = True", "assume-precessing = False"))
    cip = tmp_path/"args_cip_list.txt"
    cip.write_text("1 X --parameter m1 --parameter m2 --mc-range [10, 60] --parameter s1z --parameter s2z")
    contract = resolve(ini, tmp_path)[1]
    assert contract["bounds"]["mc"][1] > 800
    assert contract["bounds"]["eta"][0] < .01
    cip.write_text("1 X --parameter mtot --parameter q --mtot-range [40, 800] --parameter s1z --parameter s2z")
    contract = resolve(ini, tmp_path)[1]
    assert contract["bounds"]["mc"][1] > 300
    cip.write_text("1 X --parameter mc --parameter eta --parameter s1x --parameter s1y --parameter s1z --parameter s2z")
    with pytest.raises(ValueError, match="incomplete"):
        resolve(ini, tmp_path)


def test_alternate_sampling_flag_alone_does_not_change_prior(tmp_path):
    ini = configuration(tmp_path)
    with (tmp_path/"args_cip_list.txt").open("a") as stream:
        stream.write(" --pseudo-uniform-magnitude-prior-alternate-sampling")
    assert resolve(ini, tmp_path)[1]["spin_prior"] == "volumetric"


def test_framework_default_boolean_and_overwritten_cuts(tmp_path):
    ini = configuration(tmp_path)
    ini.write_text(ini.read_text().replace("spin-magnitude-prior = volumetric", 'spin-magnitude-prior = "default"').replace("assume-precessing = True", "assume-precessing ="))
    with (tmp_path/"args_cip_list.txt").open("a") as stream:
        stream.write(" --downselect-parameter s1z --downselect-parameter-range [-.1,.1]")
    contract = resolve(ini, tmp_path)[1]
    assert contract["spin_prior"] == "volumetric"
    assert contract["cuts"]["s1z"][1] > .9
    ini.write_text(ini.read_text().replace("assume-precessing =\n", "assume-precessing = False\n"))
    (tmp_path/"args_cip_list.txt").write_text("1 X --parameter mc --parameter eta --parameter s1z --parameter s2z")
    assert resolve(ini, tmp_path)[1]["spin_prior"] == "uniform_aligned"


def test_reverse_spin_caps_respect_native_cartesian_clip(tmp_path):
    ini = configuration(tmp_path)
    with (tmp_path/"args_cip_list.txt").open("a") as stream:
        stream.write(" --chi-max .5 --chi-small-max .9")
    with pytest.raises(ValueError, match="anisotropic"):
        resolve(ini, tmp_path)
    ini.write_text(ini.read_text().replace("assume-precessing = True", "assume-precessing = False"))
    (tmp_path/"args_cip_list.txt").write_text("1 X --parameter mc --parameter eta --parameter s1z --parameter s2z --chi-max .5 --chi-small-max .9")
    assert resolve(ini, tmp_path)[1]["cuts"]["s2z"] == [-.5,.5]


@pytest.mark.parametrize("diagnostics", [False, True])
def test_real_pseudo_pipe_terminal_job(tmp_path, diagnostics):
    pytest.importorskip("lal")
    from test_rf_transverse_pipeline_build import _build, _opt_in_ini
    ini = _opt_in_ini(tmp_path)
    ini.write_text(ini.read_text()+"\n[rift-auto-fit]\nenabled = true\ncomponent_counts = 1,2\n")
    if diagnostics:
        ini.write_text(ini.read_text()+"build-diagnostics = true\n")
    proc, rundir = _build(tmp_path, ini)
    assert proc.returncode == 0, proc.stdout[-6000:]
    submit = (rundir/"AUTO_FIT.sub").read_text()
    assert "universe = local" in submit.lower()
    assert "auto_fit.sh" in submit
    assert "request_disk = 2097152" in submit.lower()
    assert (rundir/"auto_fit.final_cip_args.json").is_file()
    assert (rundir/"auto_fit.production.ini").read_text() == ini.read_text()
    wrapper = (rundir/"auto_fit.sh").read_text()
    assert "--posterior extrinsic_posterior_samples.dat" in wrapper
    assert "--intrinsic-posterior" not in wrapper
    assert "overlap-grid" not in wrapper
    dag = (rundir/"marginalize_intrinsic_parameters_BasicIterationWorkflow.dag").read_text()
    assert "AUTO_FIT.sub" in dag
    lines = dag.splitlines()
    job = next(line.split()[1] for line in lines if line.startswith("JOB ") and "AUTO_FIT.sub" in line)
    assert any(line.startswith("PARENT ") and " CHILD "+job in line for line in lines)
    if diagnostics:
        for category, stem in (("AUTO_FIT_SAMPLE", "auto_fit_sample"), ("AUTO_FIT_COMPARE", "auto_fit_compare")):
            child = next(line.split()[1] for line in lines if line.startswith("JOB ") and category+".sub" in line)
            assert any(line == "PARENT "+job+" CHILD "+child for line in lines)
            assert "universe = local" in (rundir/(category+".sub")).read_text().lower()
            assert "util_RIFT_compress_posterior.py" not in (rundir/(stem+".sh")).read_text()
            job = child
    else:
        assert "AUTO_FIT_SAMPLE.sub" not in dag
        assert "AUTO_FIT_COMPARE.sub" not in dag



@pytest.mark.parametrize("aligned,flags", [
    (True, "assume-nonprecessing = True"),
    (True, ""),
    (False, ""),
    (False, "assume-precessing = False"),
    (True, "assume-precessing = True\nassume-nonprecessing = True"),
])
def test_default_targets_follow_resolved_spin_coordinates(tmp_path, aligned, flags):
    ini = configuration(tmp_path)
    ini.write_text(ini.read_text().replace("assume-precessing = True", flags)
                   .replace("targets = aligned_distance, precessing_source_redshift\n", ""))
    if aligned:
        path = tmp_path/"args_cip_list.txt"
        text = path.read_text()
        for name in ("s1x", "s1y", "s2x", "s2y"):
            text = text.replace("--parameter "+name+" ", "")
        path.write_text(text)
    _, contract, settings = resolve(ini, tmp_path)
    assert contract["precessing"] == (not aligned)
    assert settings["targets"] == (["aligned_distance"] if aligned else
        ["precessing_source_redshift", "aligned_distance"])


def test_nonprecessing_override_conflict_refused(tmp_path):
    ini = configuration(tmp_path)
    ini.write_text(ini.read_text().replace("assume-precessing = True", "assume-nonprecessing = True"))
    with pytest.raises(ValueError, match="nonprecession choice conflicts"):
        resolve(ini, tmp_path)


def test_js_weighted_and_degenerate_marginals():
    from compression.diagnostics import histogram_js
    edges = np.arange(5.)
    assert histogram_js([.5, 1.5], [1, 3], [.5, 1.5, 1.5, 1.5], [1]*4, edges) == 0
    assert histogram_js([.5]*3, [1]*3, [3.5]*3, [1]*3, edges) == pytest.approx(1.)
    with pytest.raises(ValueError, match="lost probability"):
        histogram_js([-.5], [1], [.5], [1], edges)


def test_diagnostic_parameter_conventions():
    from compression.diagnostics import standard_parameters
    class Context:
        def redshift(self, d):
            return np.full(len(d), .5)
    # m1=40,m2=20, chi1_perp=.3, chi2_perp=.4.
    eta = 2/9
    points = np.array([[60*eta**.6, eta, .3, 0, .2, 0, .4, -.1, 1000.]])
    values = standard_parameters(points, Context())
    assert values["m1"][0] == pytest.approx(40)
    assert values["q"][0] == pytest.approx(.5)
    assert values["xi"][0] == pytest.approx(.1)
    assert values["m1_source"][0] == pytest.approx(40/1.5)
    assert values["chi_p"][0] == pytest.approx(.3)
    assert values["phi2"][0] == pytest.approx(np.pi/2)


def test_diagnostic_relative_azimuth_convention():
    from compression.diagnostics import standard_parameters
    class Context:
        def redshift(self, d):
            return np.zeros(len(d))
    points = np.array([[30., .2, 0., .3, .2, .4, 0., -.1, 1000.],
                       [30., .2, 0., 0., .2, 0., .4, -.1, 1000.],
                       [30., .2, 0., 0., 0., 0., .4, -.1, 1000.]])
    fields = standard_parameters(points, Context())
    np.testing.assert_allclose(fields['phi12'], [1.5*np.pi, 0., 0.])
    assert np.isnan(fields['theta1'][-1])
    np.testing.assert_allclose(fields['cos_theta2'], -.1/np.hypot(.4, .1))


def test_diagnostic_output_suffix_checked_before_writing(tmp_path):
    from compression.diagnostics import compare
    with pytest.raises(ValueError, match="end in .json"):
        compare(tmp_path/'missing', 'aligned_distance', tmp_path/'missing.dat', tmp_path/'results.tsv')
    assert not (tmp_path/'results.tsv').exists()


def test_constant_heldout_coordinate_cannot_false_pass(tmp_path):
    from compression.model import split_groups, weighted_ks
    _, contract, _ = resolve(configuration(tmp_path), tmp_path)
    rng = np.random.default_rng(31)
    n = 5000
    points = np.column_stack([rng.normal(30, 1, n), rng.normal(.19, .005, n),
        np.full(n, .25), rng.normal(.1, .02, n), rng.normal(1000, 20, n)])
    groups = np.arange(n)
    _, train, _, held = split_groups(groups, 31)
    points[np.flatnonzero(train)[0], 2] = .375
    model, report = fit_posterior(points, np.ones(n), groups, contract, 'aligned_distance',
        components=[1], seed=31, maxiter=100)
    assert np.std(points[held, 2]) == 0
    assert report['status'] == 'failed_validation'
    assert report['maximum_ks'] > .4
    assert weighted_ks(points[held, 2], np.ones(held.sum()), model.sample_native(10000)[:, 2], np.ones(10000)) > .4
    with pytest.raises(ValueError, match='finite'):
        weighted_ks([np.nan], [1], [1], [1])


def test_native_projected_prior_zero_and_clamp(tmp_path):
    _, contract, _ = resolve(configuration(tmp_path), tmp_path)
    def model(prior, prec, cuts):
        return CompressedFit(dict(schema='rift-posterior-compression-v1', target='aligned_distance',
            contract=dict(contract, spin_prior=prior, precessing=prec, cuts=cuts),
            cosmology_yaml=c.CosmologyContext('Planck15', 2500).serialized,
            mixture=dict(weights=[1.], means=[np.zeros(5).tolist()], covariances=[np.eye(5).tolist()])))
    points = np.array([[30., .5, 0., .2, 1000.], [30., .5, 1e-12, .2, 1000.], [30., .5, 1e-8, .2, 1000.]])
    aligned = model('zprior_aligned', False, {})
    values = aligned.log_native_prior_chart(points)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, values[0], atol=1e-12)
    shell = model('uniform_mag_prec', True, dict(chi1=[.2,.5], chi2=[.1,.5]))
    assert np.isfinite(shell.log_native_prior_chart(points[:1])).all()
    cusp = model('uniform_mag_prec', True, {})
    with pytest.raises(ValueError, match='cusp'):
        cusp.log_native_prior_chart(points[:1])


def test_unsupported_spin_support_changes_refused(tmp_path):
    ini = configuration(tmp_path)
    path = tmp_path/'args_cip_list.txt'
    original = path.read_text()
    path.write_text(original+' --chi-max .5 --downselect-parameter chi1 --downselect-parameter-range [0,.8]')
    with pytest.raises(ValueError, match='cube-clipped'):
        resolve(ini, tmp_path)
    path.write_text('1 X --parameter m1 --parameter m2 --parameter s1z --parameter s2z --chi-max .9 --chi-small-max .1')
    with pytest.raises(ValueError, match='component-swap'):
        resolve(ini, tmp_path)


def test_effective_final_cip_snapshot_and_legacy_override(tmp_path):
    from compression.config import file_hash
    ini = configuration(tmp_path)
    args = (tmp_path/'args_cip_list.txt').read_text()
    path = tmp_path/'args_cip_list.txt'
    path.write_text(args+'\n'+args+' --transverse-prior taper-down\n')
    ini.write_text(ini.read_text().replace('assume-precessing = True',
        'assume-precessing = True\ninternal-force-iterations = 1'))
    with pytest.raises(ValueError, match='Ambiguous legacy'):
        resolve(ini, tmp_path)
    snapshot = tmp_path/'auto_fit.final_cip_args.json'
    snapshot.write_text(json.dumps(dict(arguments=args, cip_list_sha256=file_hash(path))))
    assert resolve(ini, tmp_path)[1]['spin_prior'] == 'volumetric'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='snapshot differs'):
        resolve(ini, tmp_path)
    final = tmp_path/'confirmed-final.txt'
    final.write_text(args)
    ini.write_text(ini.read_text()+'final_cip_args = confirmed-final.txt\n')
    with pytest.raises(ValueError, match='Remove legacy'):
        resolve(ini, tmp_path)
    snapshot.unlink()
    contract = resolve(ini, tmp_path)[1]
    assert contract['spin_prior'] == 'volumetric'
    assert contract['input_hashes']['confirmed-final.txt'] == file_hash(final)


def test_legacy_multistage_requires_terminal_execution_record(tmp_path):
    ini = configuration(tmp_path)
    path = tmp_path/'args_cip_list.txt'
    actual = path.read_text()
    path.write_text(actual+'\n'+actual+' --transverse-prior taper-down')
    with pytest.raises(ValueError, match='Ambiguous legacy'):
        resolve(ini, tmp_path)
    (tmp_path/'CIP_prior.sub').write_text('arguments = "'+actual+' --integrate-prior"\n')
    assert resolve(ini, tmp_path)[1]['spin_prior'] == 'volumetric'


def test_truncated_pipeline_records_actual_stage_before_prior_check(tmp_path):
    import configparser
    from test_rf_transverse_pipeline_build import _build, _opt_in_ini
    ini = _opt_in_ini(tmp_path)
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(ini)
    cfg.set('rift-pseudo-pipe', 'internal-force-iterations', '1')
    cfg.set('rift-pseudo-pipe', 'internal-propose-converge-last-stage', 'False')
    cfg.add_section('rift-auto-fit')
    cfg.set('rift-auto-fit', 'enabled', 'true')
    with ini.open('w') as stream:
        cfg.write(stream)
    result, rundir = _build(tmp_path, ini)
    # This early RF stage lacks the final uniform-magnitude prior correction.
    # Refuse its unsupported prior instead of substituting the unexecuted stage.
    assert result.returncode != 0
    selected = json.loads((rundir/'auto_fit.final_cip_args.json').read_text())
    assert '--prior-in-integrand-correction' not in selected['arguments']
    assert 'uniform_over_rbar_singular' in (rundir/'args_cip_list.txt').read_text().splitlines()[-1]
    assert 'Cylindrical spin chart requires' in result.stdout
