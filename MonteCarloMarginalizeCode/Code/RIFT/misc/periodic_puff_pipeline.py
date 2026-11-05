"""Resolve native periodic puff coordinates after all pipeline basis rewrites."""
import shlex


def arguments(text, mode='off'):
    if mode not in ('off', 'gap', 'mean'):
        raise ValueError("puff-periodic-mode must be off, gap, or mean")
    if mode == 'off':
        return text
    from RIFT.lalsimutils import periodic_params
    tokens = shlex.split(text.replace('--parameter=', '--parameter '))
    names = [b for a, b in zip(tokens, tokens[1:]) if a == '--parameter']
    effective = '--eccentric-phase-coordinates' in tokens
    existing = [b for a, b in zip(tokens, tokens[1:]) if a == '--unwrap-parameter']
    for name in dict.fromkeys(names):
        if name in periodic_params and name not in existing and not (effective and name == 'meanPerAno'):
            text += ' --unwrap-parameter ' + name
    if '--unwrap-parameter' in text:
        text += ' --unwrap-center ' + mode
    return text
