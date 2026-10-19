"""Select the early random-transverse puff without changing subdag arguments."""
import shlex


def discovery_arguments(arguments):
    """Replace the mass/spin covariance chart, retaining native cuts and scales.

    The original argument string remains the subdag configuration. The early
    random redraw deliberately has no legacy force-away exclusion: distance in
    mass/aligned coordinates would discard useful new transverse orientations.
    Refuse nuisance-coordinate puffing instead of silently dropping it.
    """
    tokens = shlex.split(arguments)
    output = []
    mass_spin = {'mc', 'mtot', 'm1', 'm2', 'eta', 'delta_mc', 'q',
                 'chieff_aligned', 's1z', 's2z', 's1z_bar', 's2z_bar',
                 'phi1', 'phi2', 'chi1_perp_u', 'chi2_perp_u'}
    allowed = {'--puff-factor', '--fmin', '--fref', '--enforce-duration-bound',
               '--downselect-parameter', '--downselect-parameter-range',
               '--mc-range', '--eta-range', '--mtot-range', '--random-seed'}
    i = 0
    while i < len(tokens):
        name, equals, value = tokens[i].partition('=')
        if name == '--fail-if-empty':
            output.append(name)
            i += 1
            continue
        if name not in allowed | {'--parameter', '--reflect-parameter', '--force-away'}:
            raise ValueError('early mu/random-transverse puff cannot inherit ' + name)
        if not equals:
            i += 1
            if i == len(tokens):
                raise ValueError('missing value for ' + name)
            value = tokens[i]
        if name in ('--parameter', '--reflect-parameter'):
            if value not in mass_spin:
                raise ValueError('early mu puff cannot replace nuisance coordinate ' + value)
        elif name != '--force-away':
            output.extend((name, value))
        i += 1
    return shlex.join(['--puff-coordinate-system', 'mu-random-transverse'] + output)


def stage_arguments(cip_lines, arguments):
    """Pair puff settings to CIP batches, changing only the first finite batch."""
    prefixes = [line.split()[0] for line in cip_lines]
    if len(prefixes) < 2 or prefixes[0].startswith('Z'):
        raise ValueError('mu discovery needs an initial finite CIP batch followed by a refinement batch')
    if int(prefixes[0].lstrip('G')) < 1:
        raise ValueError('the first mu discovery batch must have positive length')
    first = discovery_arguments(arguments)
    return [prefix + ' ' + (first if index == 0 else arguments)
            for index, prefix in enumerate(prefixes)]
