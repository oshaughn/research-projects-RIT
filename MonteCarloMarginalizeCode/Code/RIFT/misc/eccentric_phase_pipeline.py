"""Opt-in argument propagation; never change sampling parameters or priors."""
import math
import shlex

FLAG = '--eccentric-phase-coordinates'
REFERENCE = '--eccentric-phase-reference-frequency'
TARGET = '--eccentric-phase-target-frequency'
ECCENTRICITIES = {'eccentricity','eccentricity_ln','eccentricity_squared'}


def add_arguments(parser):
    parser.add_argument(FLAG,action='store_true',help='Use an effective eccentric-phase coordinate for eligible fits and puffing, retaining native priors and waveform inputs')
    parser.add_argument(REFERENCE,type=float,default=10.,help='Reference frequency label for the effective coordinate only; normally the eccentricity reference frequency')
    parser.add_argument(TARGET,type=float,default=20.,help='Target frequency label for the effective coordinate only')


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
    if 'quadratic' in values(tokens, '--fit-method') and '--integrate-prior' not in tokens:
        raise ValueError('Effective phase quadratic fitting is unsupported on the constrained circle')
    return _append(line,reference,target)


def puff_arguments(line, reference=10., target=20.):
    frequency_arguments(reference,target)
    names=values(option_tokens(line),'--parameter')
    if 'meanPerAno' not in names or not ECCENTRICITIES.intersection(names):
        raise ValueError('Effective eccentric-phase puff requires native eccentricity and meanPerAno parameters')
    return _append(line,reference,target)
