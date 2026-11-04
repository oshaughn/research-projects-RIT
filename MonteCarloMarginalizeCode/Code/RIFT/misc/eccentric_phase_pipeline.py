"""Opt-in argument propagation; preserve physical priors and external parameter conventions."""
import math
import shlex

FLAG = '--eccentric-phase-coordinates'
SAMPLING = '--eccentric-phase-sampling'
REFERENCE = '--eccentric-phase-reference-frequency'
TARGET = '--eccentric-phase-target-frequency'
ECCENTRICITIES = {'eccentricity','eccentricity_ln','eccentricity_squared'}


def add_arguments(parser):
    from RIFT.misc.eccentric_phase_models import add_arguments as model_arguments
    model_arguments(parser,build=True)
    parser.add_argument(FLAG,action='store_true',help='Use an effective eccentric-phase coordinate for eligible fits and puffing, retaining native priors and waveform inputs')
    parser.add_argument(SAMPLING,action='store_true',help='Also sample the periodic sheared angle internally; implies --eccentric-phase-coordinates')
    parser.add_argument(REFERENCE,type=float,default=None,help='Native eccentricity/anomaly epoch; defaults to the waveform reference frequency fref')
    parser.add_argument(TARGET,type=float,default=None,help='Explicit fixed target override; otherwise resolve the build-time frequency policy')
    parser.add_argument('--eccentric-phase-frequency-policy',choices=['lvk-mass','fixed'],default='lvk-mass',help='Build-time anchor policy; lvk-mass uses detector total mass and an LVK early-band interval')
    parser.add_argument('--eccentric-phase-detector-mass',type=float,default=None,help='Explicit detector-frame total mass in solar masses for manual builds without an event mass estimate')


def add_helper_arguments(parser):
    """Existing workflow eccentricity choices, accepted by the helper itself."""
    parser.add_argument('--use-eccentricity-squared',action='store_true')
    parser.add_argument('--eccentricity-prior',default='uniform',choices=['uniform','log_uniform'])
    for name in ('ecc-min','ecc-max','meanPerAno-min','meanPerAno-max'):
        parser.add_argument('--force-'+name,type=float,default=None)


def helper_cip_stage(line, options, first=False):
    """Build native eccentric declarations before selecting the worker chart."""
    eccentricity='eccentricity_squared' if first or options.use_eccentricity_squared else 'eccentricity'
    line=line.rstrip('\r\n')+' --use-eccentricity --parameter '+eccentricity
    if options.use_meanPerAno:
        line+=' --use-meanPerAno '+('--parameter-nofit' if first else '--parameter')+' meanPerAno'
    line+=' --eccentricity-prior '+options.eccentricity_prior
    for name in ('ecc-min','ecc-max','meanPerAno-min','meanPerAno-max'):
        value=getattr(options,'force_'+name.replace('-','_'),None)
        if value is not None:line+=' --{} {}'.format(name,value)
    from .eccentric_phase_models import worker_options
    return stage_arguments(line,options.eccentric_phase_reference_frequency,
        options.eccentric_phase_target_frequency,sampling=options.eccentric_phase_sampling,
        **worker_options(options))


def requested(options):
    """Recognize the user's fitting or sampling request without rewriting it."""
    return bool(getattr(options,'eccentric_phase_coordinates',False) or
                getattr(options,'eccentric_phase_sampling',False))


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


def _append(line, reference, target, procedure='peters-full-e', map_data=None):
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
    for flag,value in [('--eccentric-phase-procedure',procedure),('--eccentric-phase-map-data',map_data)]:
        previous=values(tokens,flag)
        if previous and any(v!=value for v in previous):raise ValueError('Conflicting frozen eccentric-phase procedure/map')
        if value is not None and not previous:line+=' {} {}'.format(flag,value)
    return line


def eligible(line):
    tokens=option_tokens(line)
    fitting=values(tokens,'--parameter')+values(tokens,'--parameter-implied')
    return ('--use-eccentricity' in tokens and '--use-meanPerAno' in tokens
              and 'meanPerAno' in fitting and bool(ECCENTRICITIES.intersection(fitting)))


def stage_arguments(line, reference=10., target=20., procedure='peters-full-e', map_data=None, sampling=False):
    """Leave no-anomaly-fit stages intact; transform eligible fit stages only."""
    frequency_arguments(reference,target)
    tokens=option_tokens(line)
    is_eligible=eligible(line)
    if not is_eligible:
        if FLAG in tokens or SAMPLING in tokens:
            raise ValueError('Effective eccentric-phase flag on a stage that does not fit eccentricity/anomaly')
        return line
    if 'quadratic' in values(tokens, '--fit-method') and '--integrate-prior' not in tokens:
        raise ValueError('Effective phase quadratic fitting is unsupported on the constrained circle')
    line = _append(line,reference,target,procedure,map_data)
    if sampling and SAMPLING not in option_tokens(line):
        line += ' '+SAMPLING
    return line


def puff_arguments(line, reference=10., target=20., procedure='peters-full-e', map_data=None):
    frequency_arguments(reference,target)
    names=values(option_tokens(line),'--parameter')
    if 'meanPerAno' not in names or not ECCENTRICITIES.intersection(names):
        raise ValueError('Effective eccentric-phase puff requires native eccentricity and meanPerAno parameters')
    return _append(line,reference,target,procedure,map_data)


def validate_request(options):
    from RIFT.misc.eccentric_phase_models import map_path
    path=getattr(options,"eccentric_phase_map",None)
    if path is not None:options.eccentric_phase_map=str(map_path(path))
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
    if getattr(options,'eccentric_phase_sampling',False):line+=' '+SAMPLING
    if getattr(options,'use_eccentricity_squared',False):line+=' --use-eccentricity-squared'
    line+=' --eccentricity-prior '+getattr(options,'eccentricity_prior','uniform')
    for name in ('ecc-min','ecc-max','meanPerAno-min','meanPerAno-max'):
        value=getattr(options,'force_'+name.replace('-','_'),None)
        if value is not None:line+=' --force-{} {}'.format(name,value)
    for flag,name in [(REFERENCE,'eccentric_phase_reference_frequency'),(TARGET,'eccentric_phase_target_frequency'),('--eccentric-phase-detector-mass','eccentric_phase_detector_mass')]:
        value=getattr(options,name,None)
        if value is not None:line+=' {} {}'.format(flag,float(value))
    for flag,name in [('--eccentric-phase-procedure','eccentric_phase_procedure'),('--eccentric-phase-map','eccentric_phase_map'),('--eccentric-phase-map-data','eccentric_phase_map_data')]:
        value=getattr(options,name,None)
        if value is not None:line+=' {} {}'.format(flag,shlex.quote(value))
    return line
