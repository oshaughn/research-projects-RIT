#!/usr/bin/env python
"""Discrete NR-catalog posterior: plan from cleaned ILE spokes, combine conditionals.

Planning/combining ASCII needs only NumPy; XML/catalog checks need RIFT/lalsuite.
Installed instructions: sys.prefix/share/RIFT/nr-best-effort/nr_best_effort.rst.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def probabilities(log_weights):
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("invalid log weights")
    peak = np.max(values)
    if not np.isfinite(peak):
        raise ValueError("all components have zero posterior probability")
    weights = np.exp(values - peak)
    return weights / weights.sum()


def logsumexp(values):
    peak = np.max(values)
    return float(peak + np.log(np.exp(np.asarray(values) - peak).sum()))


def coordinates(intrinsic):
    values = [intrinsic['m2'] / intrinsic['m1']]
    values += [intrinsic[k] for k in ('a1x', 'a1y', 'a1z', 'a2x', 'a2y', 'a2z')]
    values += [intrinsic[k] for k in ('eccentricity', 'meanPerAno') if k in intrinsic]
    return np.array(values)


def signature(intrinsic, digits):
    return json.dumps([round(float(v), digits) for v in coordinates(intrinsic)], separators=(',', ':'))


def make_plan(data, mass_range, power=0., digits=4, eccentric=False, mean_per_ano=False,
              simulation_priors=None, allow_partial_mass_range=False, coalesce_repeats=False):
    """Trapezoid quadrature of L_marg(M) p(M) on each physical spoke.

    Repeated intrinsic points are rejected: combine repeated Monte Carlo estimates
    with util_CleanILE.py first, rather than counting them as extra prior volume.
    """
    if mean_per_ano and not eccentric:
        raise ValueError("meanPerAno requires eccentricity")
    lo, hi = mass_range
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi and np.isfinite(power)):
        raise ValueError("mass range must be finite, positive, and increasing; prior power finite")
    extra = int(eccentric) + int(mean_per_ano)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if not data.size or data.shape[1] != 13 + extra or not np.all(np.isfinite(data)):
        raise ValueError("expected finite cleaned ILE rows: index,masses,6 spins,[ecc,meanPerAno],lnL,sigma/L,ntot,neff")
    if np.any(data[:, 1:3] <= 0) or np.any(data[:, 10 + extra] < 0):
        raise ValueError("nonpositive masses or negative relative likelihood error")
    fields = ['m1', 'm2', 'a1x', 'a1y', 'a1z', 'a2x', 'a2y', 'a2z']
    fields += ['eccentricity'] if eccentric else []
    fields += ['meanPerAno'] if mean_per_ano else []
    records = []
    for row in data:
        intrinsic = dict(zip(fields, map(float, row[1:9 + extra])))
        records.append((coordinates(intrinsic), intrinsic, row))
    # Decimal bins split a single spoke when rounded component masses put q
    # on opposite sides of a bin boundary. Cluster by maximum coordinate span,
    # without transitive chaining. Sorting makes the result input-order invariant.
    records.sort(key=lambda item: tuple(item[0]))
    minima = np.empty((len(records), len(records[0][0])))
    maxima = np.empty_like(minima)
    group_keys = []
    groups = {}
    tolerance = 10. ** (-digits)
    for coord, intrinsic, row in records:
        n = len(group_keys)
        # Records (and new group minima) are monotone in q. Exclude groups
        # already outside the q tolerance before comparing the other labels;
        # large completed catalogs otherwise pay O(rows * all_simulations).
        start = int(np.searchsorted(minima[:n, 0], coord[0] - tolerance*(1+1e-12)))
        matches = np.flatnonzero(np.all(
            np.maximum(maxima[start:n], coord) - np.minimum(minima[start:n], coord) <= tolerance, axis=1)) + start
        if len(matches) > 1:
            raise ValueError("ambiguous near-coincident spokes; increase --digits or select simulations explicitly")
        if len(matches):
            index = int(matches[0])
            key = group_keys[index]
            minima[index] = np.minimum(minima[index], coord)
            maxima[index] = np.maximum(maxima[index], coord)
        else:
            key = signature(intrinsic, digits)
            if key in groups:
                raise ValueError("spoke signature collision; increase --digits")
            group_keys.append(key)
            minima[n] = maxima[n] = coord
            groups[key] = []
        groups[key].append((float(row[1] + row[2]), intrinsic, row))
    priors = {key: 1. for key in groups} if simulation_priors is None else simulation_priors
    if set(priors) != set(groups) or any(not np.isfinite(v) or v <= 0 for v in priors.values()):
        raise ValueError("simulation priors must give a finite positive weight for every spoke, with no extra keys")
    prior_total = sum(priors.values())
    if not np.isfinite(prior_total):
        raise ValueError("simulation prior total overflows")
    # Normalize the same power-law mass prior on the SAME interval for all spokes.
    # Integrate in log-M to avoid cancellation near power=-1.
    exponent = power + 1.
    log_ratio = np.log(hi / lo)
    if abs(exponent) < 1e-12:
        log_norm = np.log(log_ratio)
    else:
        z = exponent * log_ratio
        log_abs_expm1 = z + np.log1p(-np.exp(-z)) if z > 50 else np.log(abs(np.expm1(z)))
        log_norm = exponent * np.log(lo) + log_abs_expm1 - np.log(abs(exponent))
    components = []
    summaries = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda item: item[0])
        input_count = len(rows)
        if coalesce_repeats:
            merged = []
            buckets = {}
            for item in rows:
                # Use the same quantizer as duplicate detection below. Python's
                # round and numpy.round differ at binary half-way values.
                buckets.setdefault(float(np.round(item[0], digits)), []).append(item)
            for bucket in buckets.values():
                if len(bucket) == 1:
                    merged.append(bucket[0])
                    continue
                likelihoods = np.array([item[2][9 + extra] for item in bucket])
                errors = np.array([item[2][10 + extra] for item in bucket])
                if np.any(errors <= 0):
                    raise ValueError("duplicate estimates need positive relative MC uncertainties")
                log_sigma = np.log(errors) + likelihoods
                log_precision_total = logsumexp(-2 * log_sigma)
                log_weights = -2 * log_sigma - log_precision_total
                mean = logsumexp(likelihoods + log_weights)
                row = bucket[0][2].copy()
                row[9 + extra] = mean
                row[10 + extra] = np.exp(-.5 * log_precision_total - mean)
                row[11 + extra] = sum(item[2][11 + extra] for item in bucket)
                row[12 + extra] = -1
                merged.append((bucket[0][0], bucket[0][1], row))
            rows = merged
        combined_repeats = input_count - len(rows)
        masses_all = np.array([item[0] for item in rows])
        if np.any(np.diff(np.round(masses_all, digits)) <= 0):
            raise ValueError("duplicate masses on spoke {}; clean repeated estimates first, or use --coalesce-repeats for same-waveform replicas".format(key))
        rows = [item for item in rows if lo-1e-8 <= item[0] <= hi+1e-8]
        masses = np.clip(np.array([item[0] for item in rows]), lo, hi)
        complete = len(rows) >= 2 and np.isclose(masses[0], lo, rtol=0, atol=1e-8) and np.isclose(masses[-1], hi, rtol=0, atol=1e-8)
        if len(rows) < 2 or (not complete and not allow_partial_mass_range):
            raise ValueError("spoke {} needs grid nodes at both mass-prior boundaries {}; no extrapolation".format(key, mass_range))
        widths = np.zeros(len(rows))
        widths[:-1] += np.diff(masses) / 2
        widths[1:] += np.diff(masses) / 2
        mass_logs = np.log(widths) + power * np.log(masses) - log_norm
        likelihoods = np.array([item[2][9 + extra] for item in rows])
        log_z = logsumexp(mass_logs + likelihoods)
        sim_log_prior = float(np.log(priors[key]) - np.log(prior_total))
        summaries.append(dict(simulation=key, log_evidence=log_z, log_prior=sim_log_prior,
                              integrated_mass_range=[float(masses[0]), float(masses[-1])],
                              complete_mass_range=bool(complete),
                              input_rows=input_count, combined_repeat_rows=combined_repeats,
                              max_relative_error=float(max(item[2][10 + extra] for item in rows))))
        for i, (mass, intrinsic, row) in enumerate(rows):
            components.append(dict(id=len(components), simulation=key, intrinsic=intrinsic,
                                   mass=mass, log_marginal_likelihood=float(likelihoods[i]),
                                   log_weight=float(sim_log_prior + mass_logs[i] + likelihoods[i]),
                                   relative_error=float(row[10 + extra]), samples=[], status_files=[]))
    sim_probs = probabilities([s['log_evidence'] + s['log_prior'] for s in summaries])
    for summary, prob in zip(summaries, sim_probs):
        summary['posterior_probability'] = float(prob)
    return dict(schema_version=1, sample_kind='fairdraw', components=components,
                simulations=summaries, mass_range=list(mass_range), mass_prior_power=power,
                digits=digits, eccentric=eccentric, mean_per_ano=mean_per_ano,
                allow_partial_mass_range=allow_partial_mass_range,
                coalesce_repeats=coalesce_repeats,
                interpretation='discrete physical-spoke prior; detector-frame mass quadrature')


def export_xml(plan, draws, template_path, output_dir, posterior_only=False, settings_only=False):
    # Lazy imports keep likelihood-table planning and posterior combination kernel-free.
    import lal
    from RIFT import lalsimutils
    templates = lalsimutils.xml_to_ChooseWaveformParams_array(str(template_path),
                                                           **({'minrow': 0, 'maxrow': 1} if settings_only else {}))
    if not templates:
        raise ValueError('template XML is empty')
    required = {plan['components'][int(i)]['simulation'] for i in draws} if posterior_only else None
    representatives = {}
    for component in plan['components']:
        if required is not None and component['simulation'] not in required:
            continue
        representatives.setdefault(component['simulation'], coordinates(component['intrinsic']))
    keys = sorted(representatives, key=lambda k: representatives[k][0])
    coords = np.array([representatives[k] for k in keys])
    lookup = {}
    for point in ([] if settings_only else templates):
        intrinsic = dict(m1=point.m1 / lal.MSUN_SI, m2=point.m2 / lal.MSUN_SI,
                         **{out: getattr(point, src) for out, src in
                            zip(('a1x','a1y','a1z','a2x','a2y','a2z'),
                                ('s1x','s1y','s1z','s2x','s2y','s2z'))})
        for name, enabled in [('eccentricity', plan['eccentric']), ('meanPerAno', plan['mean_per_ano'])]:
            if enabled:
                intrinsic[name] = getattr(point, name)
        coord = coordinates(intrinsic)
        tolerance = 10.**(-plan['digits'])
        start = int(np.searchsorted(coords[:,0],coord[0]-tolerance*(1+1e-12)))
        stop = int(np.searchsorted(coords[:,0],coord[0]+tolerance*(1+1e-12),side='right'))
        matches = np.flatnonzero(np.all(np.abs(coords[start:stop] - coord) <= tolerance, axis=1)) + start
        if len(matches) > 1:
            raise ValueError("template XML ambiguously matches multiple spokes")
        if len(matches):
            lookup.setdefault(keys[int(matches[0])], point)
            if len(lookup)==len(keys):
                break

    def copy_component(component):
        if not settings_only and component['simulation'] not in lookup:
            raise ValueError("template XML lacks spoke " + component['simulation'])
        template = templates[0] if settings_only else lookup[component['simulation']]
        point = template.manual_copy()
        point.tref = template.tref
        point.m1 = component['intrinsic']['m1'] * lal.MSUN_SI
        point.m2 = component['intrinsic']['m2'] * lal.MSUN_SI
        for out, src in zip(('a1x','a1y','a1z','a2x','a2y','a2z'),
                            ('s1x','s1y','s1z','s2x','s2y','s2z')):
            setattr(point, src, component['intrinsic'][out])
        for name in ('eccentricity', 'meanPerAno'):
            if name in component['intrinsic']:
                setattr(point, name, component['intrinsic'][name])
        return point

    # One event per mass cell: row index is manifest component id.
    posterior = [copy_component(plan['components'][int(i)]) for i in draws]
    if not posterior_only:
        grid = [copy_component(c) for c in plan['components']]
        lalsimutils.ChooseWaveformParams_array_to_xml(grid, str(output_dir / 'quadrature-grid'))
    lalsimutils.ChooseWaveformParams_array_to_xml(posterior, str(output_dir / 'intrinsic-posterior'))


def check_catalog(plan, group, nrwf=None):
    """Match native intrinsic labels via the maintained metadata-only public API.

    Never invert q without also exchanging spins, and never run NRSimulationLookup:
    historical implementations fold q>1 independently of the spin labels. Report
    all identities rather than silently choosing a resolution/extraction.
    """
    if nrwf is None:
        from RIFT.physics._nrwf_loader import get_nrwf
        nrwf, available = get_nrwf()
        if not available:
            raise ValueError('NR backend unavailable; configure NR_BASE and the public NRWaveformCatalogManager3 module')
    if group not in nrwf.internal_ParametersAvailable:
        raise ValueError('NR group is absent from this backend: ' + group)
    representatives = {}
    for component in plan['components']:
        representatives.setdefault(component['simulation'], component['intrinsic'])
    keys = list(representatives)
    targets = np.array([coordinates(representatives[k]) for k in keys])
    matches = {k: [] for k in keys}
    for param in nrwf.internal_ParametersAvailable[group]:
        point = nrwf.WaveformModeCatalog(group, param, metadata_only=True).P
        intrinsic = dict(m1=float(point.m1), m2=float(point.m2),
            **{out: float(getattr(point, src)) for out, src in zip(
                ('a1x','a1y','a1z','a2x','a2y','a2z'), ('s1x','s1y','s1z','s2x','s2y','s2z'))})
        for name in ('eccentricity', 'meanPerAno'):
            if name in representatives[keys[0]]:
                value = getattr(point, name, None)
                if value is None:
                    raise ValueError('catalog metadata lacks native ' + name)
                intrinsic[name] = float(value)
        coord = coordinates(intrinsic)
        found = np.flatnonzero(np.all(np.abs(targets-coord) <= 10.**(-plan['digits']), axis=1))
        for index in found:
            matches[keys[int(index)]].append(dict(group=group, parameter=str(param)))
    absent = [k for k in keys if not matches[k]]
    if absent:
        raise ValueError('{} spokes do not match native catalog metadata (first: {}). Check archive version, component ordering, and remapped eccentricities.'.format(len(absent), absent[0]))
    module_path = getattr(nrwf, '__file__', None)
    result = dict(group=group, backend=getattr(nrwf, '__name__', type(nrwf).__name__),
                  module_path=module_path, nr_base=os.environ.get('NR_BASE'),
                  waveform_io_validated=False, matches=matches,
                  ambiguous_spokes=[k for k in keys if len(matches[k]) != 1])
    if module_path and Path(module_path).is_file():
        result['module_sha256'] = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    if hasattr(nrwf, 'legacy_module'):
        legacy = nrwf.legacy_module()
        legacy_path = getattr(legacy, '__file__', None)
        if legacy_path and Path(legacy_path).is_file():
            result['wrapped_module_path'] = legacy_path
            result['wrapped_module_sha256'] = hashlib.sha256(Path(legacy_path).read_bytes()).hexdigest()
    return result


def combine(plan, manifest_dir, count, seed):
    if plan.get('schema_version') != 1 or plan.get('sample_kind') != 'fairdraw':
        raise ValueError("manifest must declare schema_version=1 and sample_kind=fairdraw")
    components = plan['components']
    if count <= 0:
        raise ValueError("output sample count must be positive")
    ids = [c['id'] for c in components]
    if any(not isinstance(i, int) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("component ids must be unique integers")
    probs = probabilities([c['log_weight'] for c in components])
    rng = np.random.default_rng(seed)
    # Require ALL positive-weight components, even if a random draw happens to
    # omit one. Missing followups must never silently renormalize the posterior.
    pools = {}
    diagnostics = {}
    names = None
    seen = set()
    for i, (component, prob) in enumerate(zip(components, probs)):
        if not isinstance(component.get('samples'), list):
            raise ValueError("component samples must be a list of filenames")
        if not component.get('samples'):
            if np.isfinite(component['log_weight']):
                raise ValueError("missing fairdraw samples for component {}".format(component['id']))
            continue
        arrays = []
        files = []
        statuses = []
        if not isinstance(component.get('status_files', []), list):
            raise ValueError('component status_files must be a list of filenames')
        for filename in component.get('status_files', []):
            path = (manifest_dir / filename).resolve()
            status = json.loads(path.read_text())
            if status.get('collapsed') is not False:
                raise ValueError('conditional samples are not a valid posterior: '+str(path))
            statuses.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for filename in component['samples']:
            path = (manifest_dir / filename).resolve()
            if path in seen:
                raise ValueError("sample file used more than once: " + str(path))
            seen.add(path)
            files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            data = np.atleast_1d(np.genfromtxt(path, names=True, dtype=float))
            if not len(data) or data.dtype.names is None:
                raise ValueError("empty or invalid posterior table: " + str(path))
            if names is None:
                names = data.dtype.names
            if data.dtype.names != names:
                raise ValueError("posterior schemas differ: " + str(path))
            if not all(np.all(np.isfinite(data[n])) for n in names):
                raise ValueError("nonfinite posterior samples: " + str(path))
            # A planned fixed-mass cell must not accidentally consume a different
            # intrinsic point or a whole mass-marginalized posterior.
            for name, value in component.get('intrinsic', {}).items():
                if name not in names:
                    raise ValueError("posterior table lacks intrinsic column " + name)
                if not np.allclose(data[name], value, rtol=1e-6, atol=1e-7):
                    raise ValueError("intrinsic mismatch in {}: {}".format(path, name))
            arrays.append(data)
        pools[i] = np.concatenate(arrays)
        diagnostics[str(component['id'])] = dict(available=len(pools[i]), drawn=0,
            unique_drawn=0, posterior_probability=float(prob), files=files, status_files=statuses)
    choices = rng.choice(len(components), size=count, p=probs)
    result = np.empty(count, dtype=pools[int(choices[0])].dtype)
    for i in np.unique(choices):
        where = np.flatnonzero(choices == i)
        pool = pools[int(i)]
        indices = rng.integers(len(pool), size=len(where))
        result[where] = pool[indices]
        diagnostics[str(components[i]['id'])].update(drawn=len(where), unique_drawn=len(np.unique(indices)))
    return result, choices, diagnostics


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('plan', help='bootstrap a discrete catalog posterior from cleaned all.net')
    build.add_argument('--composite', type=Path, required=True)
    build.add_argument('--mass-range', nargs=2, type=float, required=True, metavar=('MIN', 'MAX'))
    build.add_argument('--mass-prior-power', type=float, default=0.)
    build.add_argument('--allow-partial-mass-range', action='store_true',
                       help='integrate only observed in-range mass nodes; keep the common prior normalization (truncated evidence)')
    build.add_argument('--coalesce-repeats', action='store_true',
                       help='inverse-variance combine repeated same-waveform MC estimates at a spoke mass (never distinct waveform resolutions)')
    build.add_argument('--simulation-priors', type=Path, help='JSON object: physical spoke signature -> prior weight')
    build.add_argument('--digits', type=positive_int, default=4)
    build.add_argument('--eccentric', action='store_true')
    build.add_argument('--mean-per-ano', action='store_true')
    build.add_argument('--template-xml', type=Path, help='also export quadrature and posterior XML using original grid settings')
    build.add_argument('--posterior-xml-only', action='store_true', help='with --template-xml, skip the full quadrature-grid XML (intrinsic followups need only posterior draws)')
    build.add_argument('--template-settings-only', action='store_true', help='use only the first XML row for shared run settings; overwrite every intrinsic label from the composite instead of matching template spokes')
    build.add_argument('--output-dir', type=Path, required=True)
    build.add_argument('--n-output-samples', type=positive_int, default=3000)
    build.add_argument('--seed', type=int, default=0)
    catalog = sub.add_parser('catalog-check', help='check native spoke labels against NRWaveformCatalogManager3 metadata')
    catalog.add_argument('--manifest', type=Path, required=True)
    catalog.add_argument('--group', required=True)
    catalog.add_argument('--output', type=Path, required=True)
    merge = sub.add_parser('combine', help='combine saved conditional fair posterior draws')
    merge.add_argument('--manifest', type=Path, required=True)
    merge.add_argument('--output', type=Path, required=True)
    merge.add_argument('--n-output-samples', type=positive_int, default=3000)
    merge.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == 'plan':
            priors = json.loads(args.simulation_priors.read_text()) if args.simulation_priors else None
            plan = make_plan(np.loadtxt(args.composite), args.mass_range, args.mass_prior_power,
                             args.digits, args.eccentric, args.mean_per_ano, priors,
                             args.allow_partial_mass_range, args.coalesce_repeats)
            plan['source'] = dict(path=str(args.composite.resolve()),
                                  sha256=hashlib.sha256(args.composite.read_bytes()).hexdigest())
            plan['seed'] = args.seed
            draws = np.random.default_rng(args.seed).choice(len(plan['components']), args.n_output_samples,
                        p=probabilities([c['log_weight'] for c in plan['components']]))
            out = args.output_dir
            out.mkdir(parents=True, exist_ok=False)
            if args.template_xml:
                export_xml(plan, draws, args.template_xml, out, args.posterior_xml_only, args.template_settings_only)
                plan['posterior_xml_only'] = args.posterior_xml_only
                plan['template_settings_only'] = args.template_settings_only
                plan['template_xml'] = dict(path=str(args.template_xml.resolve()),
                    sha256=hashlib.sha256(args.template_xml.read_bytes()).hexdigest())
            fields = list(plan['components'][0]['intrinsic'])
            np.savetxt(out / 'intrinsic_posterior_samples.dat',
                       [[plan['components'][int(i)]['intrinsic'][k] for k in fields] for i in draws],
                       header=' '.join(fields))
            np.savetxt(out / 'intrinsic_component_ids.dat', draws, fmt='%d', header='component_id')
            (out / 'manifest.json').write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
            print("Wrote {} mass cells on {} spokes to {}".format(len(plan['components']), len(plan['simulations']), out))
            if any(not s['complete_mass_range'] for s in plan['simulations']):
                print('WARNING: partial mass coverage; evidences omit unobserved mass intervals. Inspect manifest.json.')
        elif args.command == 'catalog-check':
            report = check_catalog(json.loads(args.manifest.read_text()), args.group)
            report['manifest_sha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            with args.output.open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write('\n')
            print('Matched {} spokes; {} have multiple catalog identities. Metadata check only.'.format(
                  len(report['matches']), len(report['ambiguous_spokes'])))
        else:
            for path in (args.output, Path(str(args.output)+'.component_ids'), Path(str(args.output)+'.provenance.json')):
                if path.exists():
                    raise ValueError('refusing to overwrite ' + str(path))
            plan = json.loads(args.manifest.read_text())
            result, choices, diagnostics = combine(plan, args.manifest.resolve().parent,
                                                   args.n_output_samples, args.seed)
            np.savetxt(args.output, np.column_stack([result[n] for n in result.dtype.names]),
                       header=' '.join(result.dtype.names))
            np.savetxt(str(args.output) + '.component_ids',
                       [plan['components'][int(i)]['id'] for i in choices], fmt='%d', header='component_id')
            Path(str(args.output) + '.provenance.json').write_text(json.dumps(dict(
                manifest=str(args.manifest.resolve()),
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                seed=args.seed, n_output_samples=args.n_output_samples,
                sampling='with replacement; output count is not effective sample count',
                components=diagnostics), indent=2) + '\n')
            print("Wrote {} posterior samples to {}".format(len(result), args.output))
    except (ValueError, KeyError, OSError, ImportError, NotImplementedError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
