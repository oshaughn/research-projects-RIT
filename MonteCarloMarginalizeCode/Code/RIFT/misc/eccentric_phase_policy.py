"""Deterministic build-time LVK anchor; no posterior or waveform evaluation."""
import hashlib
import json
import math
from pathlib import Path
import shlex

VERSION = 'lvk-mass-v1'
MTSUN = 4.9254909476412675e-6
ANCHOR_X = 0.05
LVK_BAND = (20., 40.)
RECEIPT = 'eccentric_phase_build.json'


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('{} must be finite and positive'.format(name))
    return value


def last_option(arguments, name, default=None):
    tokens = shlex.split(arguments)
    values = []
    for i, token in enumerate(tokens):
        if token == name:
            values.append(tokens[i+1])
        elif token.startswith(name+'='):
            values.append(token.split('=', 1)[1])
    return values[-1] if values else default


def native_epoch(arguments, approximant, explicit=None):
    start = positive(last_option(arguments, '--fmin-template'), 'template start')
    reference = positive(last_option(arguments, '--reference-freq', start), 'waveform reference')
    if explicit is not None:
        return positive(explicit, 'coordinate reference'), 'explicit', start, reference
    if start == reference:
        return start, 'equal-template-start-and-reference', start, reference
    if approximant == 'SEOBNRv5EHM':
        return start, 'SEOBNRv5EHM-start-angle-convention', start, reference
    raise ValueError('Ambiguous eccentricity/anomaly epoch: set --eccentric-phase-reference-frequency explicitly')


def network_floor(arguments):
    tokens = shlex.split(arguments)
    flows = {}
    for i, token in enumerate(tokens):
        value = tokens[i+1] if token == '--fmin-ifo' else token.split('=',1)[1] if token.startswith('--fmin-ifo=') else None
        if value is not None:
            detector, frequency = value.split('=',1)
            flows[detector] = positive(frequency, 'detector low frequency')
    return min(flows.values()) if flows else LVK_BAND[0]


def lvk_interval(detector_floor, maximum_frequency):
    lower = max(LVK_BAND[0], positive(detector_floor, 'detector floor'))
    upper = max(lower, LVK_BAND[1])
    if maximum_frequency is not None:
        upper = min(upper, positive(maximum_frequency, 'maximum frequency'))
    if upper < lower:
        raise ValueError('No LVK anchor interval below the analysis maximum frequency')
    return lower, upper


def mass_anchor(total_mass, native, detector_floor=20., maximum_frequency=None):
    total_mass = positive(total_mass, 'detector-frame total mass')
    native = positive(native, 'native epoch')
    lower, upper = lvk_interval(detector_floor, maximum_frequency)
    raw = ANCHOR_X**1.5 / (math.pi * MTSUN * total_mass)
    clipped = min(upper, max(lower, raw))
    # If the weak-field landmark lies below the observed band, moving it up
    # to the detector floor would extrapolate the PN shear into later dynamics.
    # Retain the native circular chart instead; never transport backwards.
    target = native if raw < lower else max(native, clipped)
    return dict(policy_version=VERSION, detector_total_mass_msun=total_mass,
                anchor_x=ANCHOR_X, lvk_band_hz=list(LVK_BAND),
                allowed_band_hz=[lower, upper], raw_mass_frequency_hz=raw,
                band_clipped_frequency_hz=clipped, target_frequency_hz=target,
                identity_fallback=(target == native),
                fallback=('mass-landmark-below-band-identity' if raw < lower else
                          'native-epoch-identity' if target == native else
                          'lvk-ceiling-for-low-mass' if raw > upper else 'none'),
                achieved_x=(math.pi*MTSUN*total_mass*target)**(2./3.))


def resolve(options, event, ile_arguments, approximant):
    explicit_ref = getattr(options, 'eccentric_phase_reference_frequency', None)
    explicit_target = getattr(options, 'eccentric_phase_target_frequency', None)
    policy = getattr(options, 'eccentric_phase_frequency_policy', 'lvk-mass')
    if policy not in ('lvk-mass', 'fixed'):
        raise ValueError('Unknown eccentric-phase frequency policy')
    native, epoch_source, start, reference = native_epoch(ile_arguments, approximant, explicit_ref)
    total = getattr(options, 'eccentric_phase_detector_mass', None)
    masses = None
    mass_source = 'explicit-detector-total-mass' if total is not None else None
    if total is None and not event.get('rf_mass_is_placeholder', False):
        if event.get('m1') is not None and event.get('m2') is not None:
            masses = [positive(event['m1'], 'detector mass1'), positive(event['m2'], 'detector mass2')]
            total = sum(masses)
            mass_source = 'build-event-detector-component-masses'
    if explicit_target is not None or policy == 'fixed':
        target = positive(20. if explicit_target is None else explicit_target, 'coordinate target')
        decision = dict(policy_version='explicit-v1' if explicit_target is not None else 'fixed-20-v1',
                        target_frequency_hz=target, identity_fallback=(target == native))
    else:
        if total is None:
            raise ValueError('LVK mass anchor needs measured detector-frame masses; provide --eccentric-phase-detector-mass or an explicit target (placeholder masses are refused)')
        decision = mass_anchor(total, native, network_floor(ile_arguments), last_option(ile_arguments, '--fmax'))
        target = decision['target_frequency_hz']
    if target < native:
        raise ValueError('Effective phase target must not precede the native epoch')
    decision.update(reference_frequency_hz=native, native_epoch_source=epoch_source,
                    waveform_start_hz=start, waveform_reference_hz=reference,
                    requested_policy=policy, explicit_target=explicit_target,
                    detector_total_mass_msun=None if total is None else positive(total,'detector total mass'),
                    policy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    solar_mass_time_s=MTSUN,
                    detector_component_masses_msun=masses, mass_source=mass_source,
                    approximant=approximant, frozen_at='workflow-build',
                    ile_arguments_sha256=hashlib.sha256(ile_arguments.encode()).hexdigest(),
                    waveform_frequency_change=False, physical_prior_change=False)
    options.eccentric_phase_reference_frequency = native
    options.eccentric_phase_target_frequency = target
    return decision


def write(record, path=RECEIPT):
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')


def read_into(options, path=RECEIPT):
    record = json.loads(Path(path).read_text())
    native = positive(record['reference_frequency_hz'], 'saved reference')
    target = positive(record['target_frequency_hz'], 'saved target')
    if target < native or record['frozen_at'] != 'workflow-build':
        raise ValueError('Invalid saved eccentric-phase build decision')
    for name, value in [('eccentric_phase_reference_frequency', native), ('eccentric_phase_target_frequency', target)]:
        requested = getattr(options, name, None)
        if requested is not None and float(requested) != value:
            raise ValueError('Helper frequency decision conflicts with the explicit build request')
        setattr(options, name, value)
    return record


def check_final_ile(record, arguments):
    # A later manual ILE override must not silently change the angle's epoch.
    start = positive(last_option(arguments, '--fmin-template'), 'final template start')
    reference = positive(last_option(arguments, '--reference-freq', start), 'final waveform reference')
    if start != record['waveform_start_hz'] or reference != record['waveform_reference_hz']:
        raise ValueError('ILE frequency override after eccentric-phase resolution; configure the waveform epoch before the helper build')
    if record['policy_version'] == VERSION:
        try:
            final_band = list(lvk_interval(network_floor(arguments),last_option(arguments,'--fmax')))
        except ValueError as error:
            raise ValueError('ILE detector-band override after eccentric-phase resolution') from error
        if final_band != record['allowed_band_hz']:
            raise ValueError('ILE detector-band override after eccentric-phase resolution; configure frequency limits before the helper build')
