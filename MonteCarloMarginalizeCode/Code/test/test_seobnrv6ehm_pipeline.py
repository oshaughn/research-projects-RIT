"""Build-only v6 routing checks. No waveform backend is mocked or executed.

Numerical v6 acceptance is separately enforced by util_CheckSEOBNRv6EHM.py.
"""
import ast
import configparser
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
lal = pytest.importorskip('lal')
from RIFT import lalsimutils as lsu

CODE = Path(__file__).resolve().parents[1]


def condor_argv(text):
    """Decode HTCondor new-syntax arguments independently of RIFT's encoder."""
    assert text.startswith('"') and text.endswith('"')
    text = text[1:-1].replace('""', '"')
    argv, token, quoted, active = [], '', False, False
    i = 0
    while i < len(text):
        char = text[i]
        if char == "'":
            if quoted and i + 1 < len(text) and text[i + 1] == "'":
                token += "'"
                i += 2
                continue
            quoted = not quoted
            active = True
        elif char.isspace() and not quoted:
            if active:
                argv.append(token)
                token, active = '', False
        else:
            token += char
            active = True
        i += 1
    assert not quoted
    if active:
        argv.append(token)
    return argv


def test_eccentric_pipeline_preserves_model_and_intrinsics(tmp_path):
    fixtures = CODE.parents[1] / '.travis/ref_ini'
    config = configparser.RawConfigParser()
    config.optionxform = str
    assert config.read(fixtures / 'GW150914.ini')
    config.set('engine', 'approx', 'SEOBNRv6EHM')
    config.set('engine', 'aligned-spin', '')
    overrides = {
        'approx': repr('SEOBNRv6EHM'), 'use-gwsignal': 'True',
        'assume-nonprecessing': 'True', 'assume-eccentric': 'True',
        'use-meanPerAno': 'True', 'force-initial-grid-size': '1',
        'cip-explode-jobs': '1', 'cip-explode-jobs-last': '1',
        'internal-marginalize-distance': 'False', 'use_osg': 'False',
        'use_osg_file_transfer': 'False', 'use_osg_cip': 'False',
        'internal-truncate-files-for-osg-file-transfer': 'False',
        'ile-force-gpu': 'False', 'ile-no-gpu': 'True',
        'add-extrinsic': 'True', 'batch-extrinsic': 'False',
        'add-extrinsic-time-resampling': 'False', 'calibration-reweighting': 'False',
        'internal-use-aligned-phase-coordinates': 'False',
        'internal-correlate-default': 'False',
        'manual-extra-ile-args': ' --internal-waveform-extra-kwargs '+shlex.quote("{'lmax_nyquist': 4}"),
    }
    for key, value in overrides.items():
        config.set('rift-pseudo-pipe', key, value)
    ini = tmp_path / 'v6.ini'
    with ini.open('w') as stream:
        config.write(stream)
    p = lsu.ChooseWaveformParams()
    p.m1, p.m2 = 60*lal.MSUN_SI, 30*lal.MSUN_SI
    p.tref = lal.LIGOTimeGPS(1126259462.391)
    p.fmin = p.fref = 20.
    p.eccentricity, p.meanPerAno = .2, .7
    grid = tmp_path / 'manual-grid'
    lsu.ChooseWaveformParams_array_to_xml([p], fname=str(grid))
    # Serialization checks real XML fields, not a Python object clone.
    restored = lsu.xml_to_ChooseWaveformParams_array(str(grid)+'.xml.gz')[0]
    assert restored.eccentricity == pytest.approx(.2)
    assert restored.meanPerAno == pytest.approx(.7)
    cache = tmp_path / 'fake.cache'
    cache.write_text('')
    env = dict(os.environ, GW_SURROGATE='', PYTHONPATH=str(CODE),
               PATH=os.pathsep.join([str(CODE/'bin'), str(Path(sys.executable).parent), os.environ['PATH']]))
    command = [sys.executable, str(CODE/'bin/util_RIFT_pseudo_pipe.py'),
               '--use-ini', str(ini), '--use-coinc', str(fixtures/'coinc.xml'),
               '--use-rundir', str(tmp_path/'run'), '--manual-initial-grid', str(grid)+'.xml.gz',
               '--fake-data-cache', str(cache), '--extra-args-helper', ' --assume-fiducial-psd-files ',
               '--skip-reproducibility']
    (tmp_path/'command.txt').write_text(shlex.join(command)+'\n')
    built = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    log = built.stdout+'\n'+built.stderr
    (tmp_path/'build.log').write_text(log)
    assert built.returncode == 0 and 'Traceback' not in log, log
    run = tmp_path/'run'
    assert (run/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').is_file()
    for filename in ('ILE.sub', 'ILE_extr.sub'):
        text = (run/filename).read_text()
        arguments = next(line.partition('=')[2].strip() for line in text.splitlines()
                         if line.startswith('arguments ='))
        argv = condor_argv(arguments)
        for flag in ('--use-gwsignal', '--save-eccentricity', '--save-meanPerAno',
                     '--time-marginalization', '--vectorized', '--gpu', '--force-xpy'):
            assert flag in argv, (filename, flag, argv)
        assert argv[argv.index('--approx')+1] == 'SEOBNRv6EHM'
        assert ast.literal_eval(argv[argv.index('--internal-waveform-extra-kwargs')+1]) == {'lmax_nyquist': 4}
    cip = (run/'args_cip_list.txt').read_text()
    assert '--use-eccentricity' in cip and '--use-meanPerAno' in cip
    assert '--parameter meanPerAno' in cip and '--parameter eccentricity' in cip
    assert 'chi1_perp' not in cip and 'chi2_perp' not in cip
    initial = lsu.xml_to_ChooseWaveformParams_array(str(run/'overlap-grid-0.xml.gz'))[0]
    assert initial.eccentricity == pytest.approx(.2)
    assert initial.meanPerAno == pytest.approx(.7)
