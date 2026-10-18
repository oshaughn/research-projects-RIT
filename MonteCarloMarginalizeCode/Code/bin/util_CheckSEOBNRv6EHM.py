#!/usr/bin/env python
"""Optional public-backend smoke check. Unavailable v6 support is skipped.

This checks bounded eccentric aligned-spin generation, not posterior recovery or
unbound initial conditions. It never substitutes another approximant for v6.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback


class WaveformUnavailable(RuntimeError):
    """The installed public waveform stack does not expose v6 yet."""


def select_generator(factory):
    try:
        return factory('SEOBNRv6EHM')
    except ValueError as exc:
        if 'Approximant not implemented in GWSignal' not in str(exc):
            raise
        raise WaveformUnavailable(str(exc)) from exc


def provenance():
    versions = {}
    for name in ('RIFT', 'lalsuite', 'pyseobnr', 'numpy', 'astropy', 'gwpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        try:
            p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        except OSError:
            # Minimal execute environments may intentionally omit Git.
            return None
        return p.stdout.strip() if p.returncode == 0 else None
    return {'python': sys.version, 'executable': sys.executable, 'packages': versions,
            'rift_commit': git('rev-parse', 'HEAD'), 'rift_tree': git('rev-parse', 'HEAD^{tree}'),
            'rift_dirty': git('status', '--porcelain', '--untracked-files=no')}


def check(options, report):
    import numpy as np
    import lal
    import lalsimulation as lalsim
    from RIFT import lalsimutils as lsu
    from RIFT.physics import GWSignal as gws

    if not gws.has_gws:
        raise WaveformUnavailable('GWSignal cannot be imported in this environment')
    wrapper = Path(gws.__file__).resolve()
    report['rift_wrapper'] = str(wrapper)
    report['rift_wrapper_sha256'] = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    factory = gws.gws.models.gwsignal_get_waveform_generator
    generator = select_generator(factory)
    metadata = generator.metadata
    metadata = metadata() if callable(metadata) else metadata
    report['generator_metadata'] = metadata
    if metadata.get('approximant') != 'SEOBNRv6EHM' or not metadata.get('modes'):
        raise RuntimeError('Generator must identify SEOBNRv6EHM and provide native modes')
    report['checks'] = {}
    metrics = report['checks']
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 60*lal.MSUN_SI, 30*lal.MSUN_SI
    P.s1x = P.s1y = P.s2x = P.s2y = 0.
    P.s1z, P.s2z = .2, -.1
    P.lambda1 = P.lambda2 = 0.
    P.dist, P.incl, P.phiref, P.psi = 500e6*lal.PC_SI, .9, .37, 0.
    P.fmin = P.fref = 20.
    P.deltaT, P.deltaF = 1./4096, None
    P.eccentricity, P.meanPerAno = .2, .7
    P.taper = lalsim.SIM_INSPIRAL_TAPER_NONE
    report['fixture'] = dict(mass1=60., mass2=30., spin1z=.2, spin2z=-.1,
                             distance_mpc=500., inclination=.9, phi_ref=.37,
                             eccentricity=.2, meanPerAno=.7, fmin=20., fref=20.,
                             sample_rate=4096., Lmax=4)
    expected = {(l, sign*m) for l, m in [(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)]
                for sign in [-1,1]}
    def modes(p):
        result = gws.hlmoft(p, Lmax=4, approx_string='SEOBNRv6EHM',
                           gwsignal_extra_kwargs=deepcopy(options))
        if not expected.issubset(result):
            raise RuntimeError('Missing v6 modes: '+str(sorted(expected-set(result))))
        lengths = {h.data.length for h in result.values()}
        epochs = {float(h.epoch) for h in result.values()}
        if len(lengths) != 1 or len(epochs) != 1 or min(lengths) < 2:
            raise RuntimeError('Modes have inconsistent lengths or epochs')
        for lm, h in result.items():
            if h.deltaT != p.deltaT or not np.isfinite(h.data.data).all():
                raise RuntimeError('Invalid sampling or nonfinite mode '+str(lm))
        if np.linalg.norm(result[(2,2)].data.data) == 0:
            raise RuntimeError('Dominant mode is zero')
        return result
    base = modes(P)
    metrics['mode_count'] = len(base)
    metrics['native_samples'] = base[(2,2)].data.length
    def residual(a, b):
        if len(a) != len(b):
            raise RuntimeError('Comparison requires identical time grids')
        return float(np.linalg.norm(a-b)/np.linalg.norm(b))
    def require(value, bound, label):
        metrics[label] = value
        if not np.isfinite(value) or value > bound:
            raise RuntimeError(label+' exceeds '+str(bound)+': '+str(value))
    # Do not fit away phase, polarization, amplitude or time offsets.
    hp = gws.hoft(deepcopy(P), Fp=1., Fc=0., approx_string='SEOBNRv6EHM',
                  gwsignal_extra_kwargs=deepcopy(options))
    hc = gws.hoft(deepcopy(P), Fp=0., Fc=1., approx_string='SEOBNRv6EHM',
                  gwsignal_extra_kwargs=deepcopy(options))
    reconstructed = sum(lal.SpinWeightedSphericalHarmonic(P.incl, -P.phiref, -2, *lm)
                        * h.data.data for lm, h in base.items())
    for h in (hp, hc):
        if abs(float(h.epoch)-float(base[(2,2)].epoch)) > P.deltaT/10:
            raise RuntimeError('Polarization/mode epochs disagree')
    require(residual(reconstructed, hp.data.data-1j*hc.data.data), 1e-6,
            'polarization_relative_residual')
    far = deepcopy(P)
    far.dist *= 2
    doubled = modes(far)
    for lm in expected:
        if float(doubled[lm].epoch) != float(base[lm].epoch):
            raise RuntimeError('Distance changed mode epochs')
        require(residual(2*doubled[lm].data.data, base[lm].data.data), 1e-8,
                'distance_residual_'+str(lm))
    # Tail comparison detects ignored intrinsic parameters without requiring
    # equal durations; length/epoch changes are recorded separately.
    for field, value in [('eccentricity', .1), ('meanPerAno', 1.4)]:
        variant = deepcopy(P)
        setattr(variant, field, value)
        changed = modes(variant)[(2,2)]
        a, b = base[(2,2)].data.data, changed.data.data
        n = min(len(a),len(b))
        difference = float(np.linalg.norm(a[-n:]/np.linalg.norm(a[-n:])
                                         - b[-n:]/np.linalg.norm(b[-n:])))
        metrics[field+'_response'] = difference
        if not np.isfinite(difference) or difference < 1e-6:
            raise RuntimeError(field+' has no measurable waveform response')
    circular = deepcopy(P)
    circular.eccentricity = 0.
    modes(circular)
    metrics['circular_generation'] = True
    # Exercise the actual likelihood-facing FFT/conjugate mode path.
    fft = deepcopy(P)
    fft.taper = lalsim.SIM_INSPIRAL_TAPER_START
    duration = 2**int(np.ceil(np.log2(base[(2,2)].data.length*P.deltaT+1)))
    fft.deltaF = 1./duration
    a, b = gws.std_and_conj_hlmoff(fft, Lmax=4, approx_string='SEOBNRv6EHM',
                                 gwsignal_extra_kwargs=deepcopy(options))
    if set(a) != set(b) or not expected.issubset(a):
        raise RuntimeError('FFT/conjugate modes missing')
    if any(not np.isfinite(h.data.data).all() for group in (a,b) for h in group.values()):
        raise RuntimeError('Nonfinite FFT modes')
    metrics['fft_duration'] = duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='JSON evidence file (also written on failure)')
    parser.add_argument('--backend-options', default='{}', help='JSON GWSignal options; no source overrides')
    parser.add_argument('--require-available', action='store_true',
                        help='Fail instead of skip when v6 is unavailable (production acceptance)')
    args = parser.parse_args()
    report = {'status': 'failed', 'approximant': 'SEOBNRv6EHM', 'provenance': provenance()}
    try:
        options = json.loads(args.backend_options)
        if not isinstance(options, dict):
            raise ValueError('--backend-options must be a JSON object')
        if options.get('condition', 0) != 0:
            raise ValueError('The convention check requires unconditioned polarizations (condition=0)')
        options['condition'] = 0
        report['backend_options'] = options
        check(options, report)
        report['status'] = 'passed'
    except WaveformUnavailable as exc:
        report['status'] = 'failed' if args.require_available else 'skipped'
        report['reason'] = str(exc)
        if args.require_available:
            report['error'] = str(exc)
    except Exception as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
        report['traceback'] = traceback.format_exc()
    Path(args.output).write_text(json.dumps(report, indent=2, default=str)+'\n')
    print('SEOBNRv6EHM gate: '+report['status']+'; '+str(Path(args.output).resolve()))
    if report['status'] == 'failed':
        print(report['error'], file=sys.stderr)
    elif report['status'] == 'skipped':
        print(report['reason'])
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
