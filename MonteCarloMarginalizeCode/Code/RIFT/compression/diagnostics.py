"""Posterior draws and one-dimensional comparisons, separate from fit building."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import jensenshannon
from .cli import read_posterior, write_json
from .config import file_hash
from .model import CompressedFit, normalized, split_groups


def standard_parameters(points, cosmology):
    """Native detector Mc/eta/spins/dL to common RIFT comparison coordinates."""
    x = np.asarray(points, float)
    mc, eta, distance = x[:, 0], x[:, 1], x[:, -1]
    total = mc/eta**.6
    delta = np.sqrt(np.maximum(0., 1-4*eta))
    m1, m2 = total*(1+delta)/2, total*(1-delta)/2
    s1z, s2z = (x[:, 4], x[:, 7]) if x.shape[1] == 9 else (x[:, 2], x[:, 3])
    z = cosmology.redshift(distance)
    fields = dict(mc=mc, eta=eta, m1=m1, m2=m2, mtot=total, q=m2/m1,
                  xi=(m1*s1z+m2*s2z)/total, s1z=s1z, s2z=s2z, dist=distance,
                  mc_source=mc/(1+z), m1_source=m1/(1+z), m2_source=m2/(1+z), redshift=z)
    if x.shape[1] == 9:
        p1, p2 = np.hypot(x[:, 2], x[:, 3]), np.hypot(x[:, 5], x[:, 6])
        q = m2/m1
        phi1, phi2 = np.arctan2(x[:, 3], x[:, 2]), np.arctan2(x[:, 6], x[:, 5])
        phi12 = np.mod(phi2-phi1, 2*np.pi)
        phi12[(p1 == 0) | (p2 == 0) | (phi12 >= 2*np.pi)] = 0.
        a1, a2 = np.hypot(p1, s1z), np.hypot(p2, s2z)
        c1 = np.divide(s1z, a1, out=np.full_like(a1, np.nan), where=a1 > 0)
        c2 = np.divide(s2z, a2, out=np.full_like(a2, np.nan), where=a2 > 0)
        fields.update(s1x=x[:, 2], s1y=x[:, 3], s2x=x[:, 5], s2y=x[:, 6],
                      chi1_perp=p1, chi2_perp=p2, a1=a1, a2=a2, cos_theta1=c1, cos_theta2=c2,
                      theta1=np.arccos(np.clip(c1, -1, 1)), theta2=np.arccos(np.clip(c2, -1, 1)),
                      chi_p=np.maximum(p1, q*(4*q+3)/(4+3*q)*p2), phi1=phi1, phi2=phi2,
                      phi12=phi12)
    return fields


def sample(model_path, output, n=20000, seed=1771):
    """Draw L_fit times its recorded native prior by exact mixture sampling."""
    model = CompressedFit.load(model_path)
    points = model.sample_native(n, seed)
    fields = standard_parameters(points, model.cosmology)
    # Include LI names used by existing RIFT plotting/format-conversion tools.
    aliases = dict(s1x='a1x', s1y='a1y', s1z='a1z', s2x='a2x', s2y='a2y', s2z='a2z',
                   xi='chi_eff', mtot='mtotal', dist='distance')
    fields.update({alias: fields[name] for name, alias in aliases.items() if name in fields})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = output.with_suffix(output.suffix+'.json')
    if output.exists() or metadata.exists():
        raise FileExistsError('Preserve existing posterior draws: '+str(output))
    with output.open('x') as stream:
        np.savetxt(stream, np.column_stack(list(fields.values())), header=' '.join(fields), comments='')
    write_json(metadata, dict(schema='rift-compressed-posterior-draws-v1', model_sha256=file_hash(model_path),
        samples_sha256=file_hash(output), rows=n, seed=seed, native_prior=model.contract,
        interpretation='Independent draws from supported L_fit times the recorded native retained prior; equal weights'))
    return output


def histogram_js(a, wa, b, wb, edges):
    """Squared base-2 Jensen-Shannon distance: divergence in bits, in [0,1]."""
    pa = np.histogram(a, edges, weights=normalized(wa))[0]
    pb = np.histogram(b, edges, weights=normalized(wb))[0]
    if not np.isclose(pa.sum(), 1.) or not np.isclose(pb.sum(), 1.):
        raise ValueError('Diagnostic histogram lost probability outside its edges')
    return float(jensenshannon(pa, pb, base=2)**2)


def compare(fit_directory, target, samples, output, bins=64):
    """Compare independent fit draws to original, held-out extrinsic groups."""
    if not isinstance(bins, int) or not 8 <= bins <= 512:
        raise ValueError('diagnostic_bins must be in [8,512]')
    root = Path(fit_directory)
    output = Path(output)
    if output.suffix != ".json":
        raise ValueError("Comparison output must end in .json; companion table uses .tsv")
    if output.exists() or output.with_suffix('.tsv').exists():
        raise FileExistsError('Preserve existing diagnostic results: '+str(output))
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete' or target not in manifest['settings']['targets']:
        raise ValueError('Diagnostics require a completed fit and a recorded target')
    model_path = root/target/'model.json'
    if file_hash(model_path) != manifest["model_sha256"][target]:
        raise ValueError("Model changed since fitting")
    model = CompressedFit.load(model_path)
    if model.contract != manifest['contract'] or model.artifact['validation'] != manifest['reports'][target]:
        raise ValueError('Model differs from fit manifest')
    posterior = Path(manifest['posterior'])
    if file_hash(posterior) != manifest['posterior_sha256']:
        raise ValueError('Original posterior changed since fitting')
    metadata = json.loads(Path(str(samples)+'.json').read_text())
    if metadata['model_sha256'] != file_hash(model_path) or metadata['samples_sha256'] != file_hash(samples):
        raise ValueError('Posterior draws do not match model/sample hashes')
    aligned, full, weights, groups, _ = read_posterior(posterior,
        manifest['settings']['weight_column'], manifest['settings']['group_column'])
    points = full if model.dimension == 9 else aligned
    _, _, _, held = split_groups(groups, model.artifact['validation']['seed'])
    reference = standard_parameters(points[held], model.cosmology)
    weights = normalized(weights[held])
    draws = np.genfromtxt(samples, names=True, ndmin=1)
    if len(draws) != metadata['rows'] or len(draws) < 100:
        raise ValueError('At least 100 posterior draws matching metadata are required')
    # A same-fit null with matched row weights estimates histogram sampling noise.
    # It does not model correlations between original rows in one intrinsic group.
    null = standard_parameters(model.sample_native(len(weights), metadata['seed']+1), model.cosmology)
    metrics, unavailable = {}, {}
    for name, values in reference.items():
        other = np.asarray(draws[name], float)
        if name in ('theta1', 'theta2', 'cos_theta1', 'cos_theta2') and not np.isfinite(values).all():
            unavailable[name] = 'Tilt undefined for zero-spin reference rows; no substituted angle'
            continue
        if not np.isfinite(other).all():
            raise ValueError('Nonfinite diagnostic samples: '+name)
        pooled = np.r_[values, other, null[name]]
        if name == 'phi12':
            edges = np.linspace(0., 2*np.pi, bins+1)
        elif name in ('phi1', 'phi2'):
            edges = np.linspace(-np.pi, np.pi, bins+1)
        else:
            edges = np.histogram_bin_edges(pooled, bins=bins)
        metrics[name] = dict(js_bits=histogram_js(values, weights, other, np.ones(len(other)), edges),
            same_fit_null_js_bits=histogram_js(null[name], weights, other, np.ones(len(other)), edges),
            bin_edges=edges.tolist())
    group_weights = np.array([weights[groups[held] == g].sum() for g in np.unique(groups[held])])
    report = dict(schema='rift-compressed-diagnostics-v1', target=target, parameters=metrics, unavailable_parameters=unavailable,
        reference='Held-out original extrinsic groups, excluded from training and tuning',
        reference_rows=len(weights), reference_group_ess=float(1/np.sum(group_weights**2)),
        posterior_rows=len(draws), bins=bins, maximum_js_bits=max(v['js_bits'] for v in metrics.values()),
        estimator='Common-edge weighted histograms; squared base-2 JS, in bits; not the KDE estimator',
        scope='Report only: finite-sample 1D native-prior recovery; no joint/tail/population certification or automatic acceptance threshold',
        null_scope='Single same-fit null realization with matching row counts/weights; ignores within-group dependence',
        model_sha256=file_hash(model_path), reference_sha256=file_hash(posterior), samples_sha256=file_hash(samples))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    with output.with_suffix('.tsv').open('x') as stream:
        stream.write('parameter\tjs_bits\tsame_fit_null_js_bits\n')
        for name, value in metrics.items():
            stream.write(f"{name}\t{value['js_bits']:.8g}\t{value['same_fit_null_js_bits']:.8g}\n")
    return report


def sample_main():
    parser = argparse.ArgumentParser(description='Draw a native-prior posterior from a saved compressed fit')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=1771)
    args = parser.parse_args()
    sample(args.model, args.output, args.n_samples, args.seed)


def compare_main():
    parser = argparse.ArgumentParser(description='Compare fit posterior draws with held-out original extrinsic samples')
    parser.add_argument('--fit-directory', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bins', type=int, default=64)
    args = parser.parse_args()
    report = compare(args.fit_directory, args.target, args.samples, args.output, args.bins)
    print(json.dumps(dict(target=args.target, maximum_js_bits=report['maximum_js_bits'])))
