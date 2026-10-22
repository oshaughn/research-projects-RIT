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

if __name__=="__main__": unittest.main()
