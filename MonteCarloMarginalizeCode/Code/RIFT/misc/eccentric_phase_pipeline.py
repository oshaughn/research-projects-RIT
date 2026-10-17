"""Opt-in argument propagation; never change sampling parameters or priors."""
import math
import shlex

FLAG = '--eccentric-phase-coordinates'
REFERENCE = '--eccentric-phase-reference-frequency'
TARGET = '--eccentric-phase-target-frequency'
ECCENTRICITIES = {'eccentricity','eccentricity_ln','eccentricity_squared'}


def add_arguments(parser):
    parser.add_argument(FLAG,action='store_true',help='Use an effective eccentric-phase coordinate for eligible fits and puffing, retaining native priors and waveform inputs')
    parser.add_argument(REFERENCE,type=float,default=None,help='Native backend eccentricity/anomaly epoch; inferred at build time when unambiguous')
    parser.add_argument(TARGET,type=float,default=None,help='Explicit fixed target override; otherwise resolve the build-time frequency policy')
    parser.add_argument('--eccentric-phase-frequency-policy',choices=['lvk-mass','fixed'],default='lvk-mass',help='Build-time anchor policy; lvk-mass uses detector total mass and an LVK early-band interval')
    parser.add_argument('--eccentric-phase-detector-mass',type=float,default=None,help='Explicit detector-frame total mass in solar masses for manual builds without an event mass estimate')


def option_tokens(line):
    """Recognize both argparse spellings without rewriting the saved command."""
    tokens = []
    for token in shlex.split(line):
        tokens.extend(token.split('=', 1) if token.startswith('--') and '=' in token else [token])
    return tokens


def values(tokens, flag):
    return [tokens[i+1] for i,t in enumerate(tokens[:-1]) if t==flag]


def frequency_arguments(reference, target):
    if not all(math.isfinite(x) for x in [reference,target]) or reference<=0 or target<reference:
        raise ValueError('Effective eccentric phase requires finite 0 < reference <= target frequencies')
    return '{} {} {} {}'.format(REFERENCE,float(reference),TARGET,float(target))


def _append(line, reference, target):
    # Pipeline stages arrive from readlines(); append options before the newline.
    line = line.rstrip('\r\n')
    tokens=option_tokens(line)
    for flag,value in [(REFERENCE,reference),(TARGET,target)]:
        previous=values(tokens,flag)
        if previous and any(float(v)!=float(value) for v in previous):
            raise ValueError('Conflicting effective eccentric-phase frequency settings')
    if FLAG not in tokens:
        line += ' '+FLAG
    if REFERENCE not in tokens:
        line += ' {} {}'.format(REFERENCE,float(reference))
    if TARGET not in tokens:
        line += ' {} {}'.format(TARGET,float(target))
    return line


def stage_arguments(line, reference=10., target=20.):
    """Leave no-anomaly-fit stages intact; transform eligible fit stages only."""
    frequency_arguments(reference,target)
    tokens=option_tokens(line)
    fitting=values(tokens,'--parameter')+values(tokens,'--parameter-implied')
    eligible=('--use-eccentricity' in tokens and '--use-meanPerAno' in tokens
              and 'meanPerAno' in fitting and bool(ECCENTRICITIES.intersection(fitting)))
    if not eligible:
        if FLAG in tokens:
            raise ValueError('Effective eccentric-phase flag on a stage that does not fit eccentricity/anomaly')
        return line
    if 'quadratic' in values(tokens, '--fit-method'):
        raise ValueError('Effective phase quadratic fitting is unsupported on the constrained circle')
    return _append(line,reference,target)


def puff_arguments(line, reference=10., target=20.):
    frequency_arguments(reference,target)
    names=values(option_tokens(line),'--parameter')
    if 'meanPerAno' not in names or not ECCENTRICITIES.intersection(names):
        raise ValueError('Effective eccentric-phase puff requires native eccentricity and meanPerAno parameters')
    return _append(line,reference,target)


def validate_request(options):
    if getattr(options,'eccentric_phase_frequency_policy','lvk-mass') not in ('lvk-mass','fixed'):
        raise ValueError('Unknown eccentric-phase frequency policy')
    for name in ['eccentric_phase_reference_frequency','eccentric_phase_target_frequency','eccentric_phase_detector_mass']:
        value=getattr(options,name,None)
        if value is not None and (not math.isfinite(value) or value<=0):
            raise ValueError(name+' must be finite and positive')
    reference=getattr(options,'eccentric_phase_reference_frequency',None)
    target=getattr(options,'eccentric_phase_target_frequency',None)
    if reference is not None and target is not None:
        frequency_arguments(reference,target)


def build_arguments(options):
    validate_request(options)
    line=' --eccentric-phase-coordinates --eccentric-phase-frequency-policy '+options.eccentric_phase_frequency_policy
    for flag,name in [(REFERENCE,'eccentric_phase_reference_frequency'),(TARGET,'eccentric_phase_target_frequency'),('--eccentric-phase-detector-mass','eccentric_phase_detector_mass')]:
        value=getattr(options,name,None)
        if value is not None:line+=' {} {}'.format(flag,float(value))
    return line
