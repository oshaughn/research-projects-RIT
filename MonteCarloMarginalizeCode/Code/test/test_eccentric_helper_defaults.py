"""Boundary retention and dimension scaling, without waveform dependencies."""
import ast
import importlib.util
from pathlib import Path
import unittest
import numpy as np
CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("defaults", CODE/"RIFT/misc/eccentric_defaults.py")
policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)

class Defaults(unittest.TestCase):
    def test_counts(self):
        for d, expected in [(2,1000),(4,750),(8,625)]:
            self.assertEqual(policy.iteration_points(500,d),expected)
        self.assertEqual(policy.iteration_points(500,2,1),750)
        self.assertEqual(policy.iteration_points(501,4),752)
    def test_reflect_multiple_crossings(self):
        x=np.array([-1.01,-.01,0,.1,.25,.26,1.01])
        y=policy.reflect_into(x,0,.25)
        np.testing.assert_allclose(y,[.01,.01,0,.1,.25,.24,.01])
        self.assertEqual(len(x),len(y));self.assertTrue(np.all((y>=0)&(y<=.25)))
    def test_real_eccentric_puff_loss(self):
        rng=np.random.RandomState(20261017)
        e=.03+rng.normal(0,.06,2000)
        old=(e>=0)&(e<=.25)
        self.assertLess(old.sum(),1800)
        reflected=policy.reflect_into(e,0,.25)
        self.assertEqual(np.sum((reflected>=0)&(reflected<=.25)),2000)
    def test_invalid_bounds_and_dimensions(self):
        for bounds in [(0,0),(.25,0),(0,float('inf'))]:
            with self.assertRaises(ValueError):policy.reflect_into(np.array([.1]),*bounds)
        with self.assertRaises(ValueError):policy.iteration_points(500,0)
    def test_emitted_helper_contract(self):
        tree=ast.parse((CODE/"bin/helper_LDG_Events.py").read_text())
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and n.test.attr=="assume_eccentric" and any(isinstance(c,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="eccentric_puff_range" for t in c.targets) for c in n.body)]
        self.assertEqual(len(nodes),1)
        from types import SimpleNamespace
        for use_angle in [False,True]:
            env={"opts":SimpleNamespace(use_meanPerAno=use_angle),"helper_puff_args":"X ","ecc_range_str":"[0,0.25]"}
            exec(compile(ast.Module(body=nodes[0].body,type_ignores=[]),"helper","exec"),env)
            self.assertIn("--reflect-parameter eccentricity",env["helper_puff_args"])
            self.assertIn("--downselect-parameter-range [0,0.25]",env["helper_puff_args"])
    def test_explicit_count_override_still_last(self):
        text=(CODE/"bin/util_RIFT_pseudo_pipe.py").read_text()
        self.assertGreater(text.index("npts_it = opts.internal_n_evaluations_per_iteration"),text.index("npts_it = iteration_points"))


class BranchIntegration(unittest.TestCase):
    def test_executed_pipeline_counts_and_overrides(self):
        from types import SimpleNamespace
        import sys
        from unittest.mock import patch
        source = (CODE/'bin/util_RIFT_pseudo_pipe.py').read_text()
        start = source.index('npts_it = 500\n')
        end = source.index('if opts.internal_ile_use_lnL:', start)
        block = compile(source[start:end], 'pipeline-counts', 'exec')
        override = next(n for n in ast.walk(ast.parse(source))
                        if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
                        and n.test.attr == 'internal_n_evaluations_per_iteration')
        class Options(SimpleNamespace):
            def __getattr__(self, name):
                return None
        with patch.dict(sys.modules, {'RIFT.misc.eccentric_defaults': policy}):
            for nospin, precessing, matter, bh, highq, angle, eccentric, expected in [
                (True,False,False,False,False,True,True,1000),
                (True,False,False,False,False,False,True,750),
                (False,False,False,False,False,True,True,750),
                (False,True,False,False,False,True,True,1875),
                (True,False,True,True,False,True,True,1667),
                (True,False,True,False,False,True,True,1500),
                (True,False,False,False,True,True,True,2000),
                (True,False,False,False,False,False,False,500),
            ]:
                with self.subTest(expected=expected, matter=matter, bh=bh, highq=highq):
                    opts = Options(assume_nospin=nospin, assume_matter=matter,
                                   assume_matter_but_primary_bh=bh, assume_highq=highq,
                                   use_meanPerAno=angle)
                    env = dict(opts=opts, is_analysis_precessing=precessing,
                               is_analysis_eccentric=eccentric, fmin=20, fmin_template=20,
                               base_dir='/tmp', dirname_run='run', helper_psd_args='',
                               eccentric_phase_pipeline=SimpleNamespace(requested=lambda _:False))
                    exec(block, env)
                    self.assertEqual(env['npts_it'], expected)
                    opts.internal_n_evaluations_per_iteration = 123
                    exec(compile(ast.Module(body=[override], type_ignores=[]), 'override', 'exec'), env)
                    self.assertEqual(env['npts_it'], 123)

    def test_branch_reflection_retains_eccentricity_without_touching_mass(self):
        source = (CODE/'bin/util_ParameterPuffball.py').read_text()
        start = source.index("\nreflect_ranges = {")
        end = source.index('# Sanity check parameters', start)
        rng = np.random.RandomState(20261017)
        samples = np.column_stack([rng.uniform(20,30,2000), .03+rng.normal(0,.06,2000)])
        env = dict(np=np, coord_names=['mc','eccentricity'], X_out=samples.copy(),
                   explicit_reflect_ranges={'eccentricity':(0.,.25)})
        exec(compile(source[start:end], 'branch-reflection', 'exec'), env)
        np.testing.assert_array_equal(env['X_out'][:,0], samples[:,0])
        self.assertTrue(np.all((env['X_out'][:,1]>=0)&(env['X_out'][:,1]<=.25)))
        np.testing.assert_allclose(env['X_out'][:,1], policy.reflect_into(samples[:,1],0,.25))


# XML-level coverage is optional in minimal environments; CI with LAL runs it.
def test_eccentric_reflection_xml_retains_rows_and_mass(tmp_path):
    import os
    import subprocess
    import sys
    import pytest
    lal = pytest.importorskip('lal')
    from RIFT import lalsimutils
    rng = np.random.default_rng(240)
    points = []
    for _ in range(400):
        p = lalsimutils.ChooseWaveformParams()
        p.m1, p.m2 = rng.uniform(60,61)*lal.MSUN_SI, rng.uniform(20,21)*lal.MSUN_SI
        p.eccentricity = rng.uniform(.001,.04)
        p.meanPerAno = rng.uniform(0,2*np.pi)
        p.fmin = p.fref = 20.
        points.append(p)
    lalsimutils.ChooseWaveformParams_array_to_xml(points, fname=str(tmp_path/'in'))
    command = [sys.executable, str(CODE/'bin/util_ParameterPuffball.py'),
               '--inj-file', str(tmp_path/'in.xml.gz'), '--inj-file-out', str(tmp_path/'out'),
               '--puff-factor','3','--random-seed','240',
               '--parameter','meanPerAno','--parameter','eccentricity',
               '--parameter','delta_mc','--parameter','mc',
               '--reflect-parameter','eccentricity','--downselect-parameter','eccentricity',
               '--downselect-parameter-range','[0,0.25]']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60,
                            env=dict(os.environ, PYTHONPATH=str(CODE), GW_SURROGATE='', OMP_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    out = lalsimutils.xml_to_ChooseWaveformParams_array(str(tmp_path/'out.xml.gz'))
    assert len(out) == 400
    assert all(0 <= p.eccentricity <= .25 for p in out)
    assert all(0 <= p.meanPerAno < 2*np.pi for p in out)
    assert np.median([p.extract_param('mc')/lal.MSUN_SI for p in out]) > 20

if __name__ == "__main__": unittest.main()
