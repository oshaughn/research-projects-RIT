"""Read bootstrap posteriors without losing eccentric waveform coordinates."""
import numpy as np

# Do not map true/relative anomaly to mean anomaly: those are different coordinates.
ALIASES = {
    'mass_1': 'm1', 'mass_2': 'm2', 'chirp_mass': 'mc',
    'luminosity_distance': 'distance', 'geocent_time': 'time', 'phase': 'phiorb',
    'spin_1x': 'a1x', 'spin_1y': 'a1y', 'spin_1z': 'a1z',
    'spin_2x': 'a2x', 'spin_2y': 'a2y', 'spin_2z': 'a2z',
    'a_1': 'a1', 'a_2': 'a2', 'chi_1': 'a1z', 'chi_2': 'a2z', 'tilt_1': 'tilt1', 'tilt_2': 'tilt2',
    'phi_12': 'phi12', 'iota': 'incl', 'lambda_1': 'lambda1', 'lambda_2': 'lambda2',
    'log_likelihood': 'lnL', 'reference_frequency': 'f_ref',
    'mean_anomaly': 'meanPerAno', 'mean_per_ano': 'meanPerAno',
}


def normalize_samples(samples):
    """Map Bilby names to converter names; refuse conflicting aliases."""
    names = samples.dtype.names if hasattr(samples, 'dtype') else samples.keys()
    columns = {}
    for name in names:
        values = np.atleast_1d(np.asarray(samples[name]))
        if values.dtype.kind not in 'biuf' or values.ndim != 1:
            continue
        if 'snr' in name or 'recalib' in name:
            continue
        target = ALIASES.get(name, name)
        if target in columns and not np.array_equal(columns[target], values, equal_nan=True):
            raise ValueError('Conflicting posterior columns for {}'.format(target))
        columns[target] = values
    if not {'m1', 'm2'}.issubset(columns):
        raise ValueError('Bootstrap posterior requires detector-frame mass_1/mass_2 (or m1/m2)')
    n = len(columns['m1'])
    if n == 0 or any(len(value) != n for value in columns.values()):
        raise ValueError('Bootstrap posterior columns must be nonempty and equally sized')
    for name in ('m1', 'm2', 'eccentricity', 'meanPerAno'):
        if name in columns and not np.all(np.isfinite(columns[name])):
            raise ValueError('Non-finite bootstrap {}'.format(name))
    if np.any(columns['m1'] <= 0) or np.any(columns['m2'] <= 0):
        raise ValueError('Bootstrap masses must be positive')
    if 'eccentricity' in columns:
        if np.any((columns['eccentricity'] < 0) | (columns['eccentricity'] >= 1)):
            raise ValueError('Bootstrap eccentricity must satisfy 0 <= e < 1')
    result = np.empty(n, dtype=[(name, float) for name in columns])
    for name, values in columns.items():
        result[name] = values
    return result


def posterior_label(filename, requested=None):
    """Return a PESummary label, or None for a raw Bilby JSON/HDF5 result."""
    if requested:
        return requested
    if str(filename).endswith(('.json', '.json.gz')):
        return None
    import h5py
    with h5py.File(filename, 'r') as handle:
        labels = [key for key, group in handle.items()
                  if key not in ('version', 'history', 'metadata')
                  and isinstance(group, h5py.Group)
                  and ('posterior_samples' in group or 'posterior' in group)]
        if 'posterior' in handle and not labels:
            return None
    if len(labels) != 1:
        raise ValueError('{} has {} analysis labels ({}); set dataset to choose one'.format(
            filename, len(labels), labels))
    return labels[0]


def dump_samples(filename, label, output):
    """Write normalized ASCII from PESummary or a raw Bilby result.

    Bilby is optional for PESummary input; its own result reader handles raw
    JSON (including gzip) and HDF5 serialization. Pickle input is unsupported.
    """
    import h5py
    filename = str(filename)
    if filename.endswith(('.pickle', '.pkl')):
        raise ValueError('Use Bilby JSON/HDF5 or a PESummary HDF5 metafile')
    raw = filename.endswith(('.json', '.json.gz'))
    if not raw:
        with h5py.File(filename, 'r') as handle:
            raw = 'posterior' in handle and not (
                label is not None and label in handle
                and isinstance(handle[label], h5py.Group)
                and ('posterior_samples' in handle[label] or 'posterior' in handle[label]))
            if not raw:
                if label is None:
                    label = posterior_label(filename)
                group = handle[label]
                table = group['posterior_samples'] if 'posterior_samples' in group else group['posterior']
                samples = normalize_samples({name: table[name][()] for name in (
                    table.dtype.names if isinstance(table, h5py.Dataset) else table.keys())})
    if raw:
        try:
            from bilby.core.result import read_in_result
        except ImportError as exc:
            raise ImportError('Raw Bilby bootstrap requires bilby; alternatively provide a PESummary metafile') from exc
        result = read_in_result(filename=filename)
        columns = {name: np.asarray(result.posterior[name]) for name in result.posterior.columns}
        reference = result.meta_data.get('likelihood', {}).get('waveform_arguments', {}).get('reference_frequency')
        if reference is not None and not {'reference_frequency', 'f_ref'}.intersection(columns):
            columns['f_ref'] = np.full(len(result.posterior), float(reference))
        samples = normalize_samples(columns)
    np.savetxt(output, samples, header=' '.join(samples.dtype.names))
    return samples
