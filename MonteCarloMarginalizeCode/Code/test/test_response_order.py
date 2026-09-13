import unittest
from unittest import mock
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np

_MODULE = (Path(__file__).resolve().parents[1] / "RIFT" / "likelihood"
           / "response_order.py")
_rift = types.ModuleType("RIFT")
_likelihood = types.ModuleType("RIFT.likelihood")
_likelihood.__path__ = []
_fl = types.ModuleType("RIFT.likelihood.factored_likelihood")
_fl.ComputeYlmsArrayVector = lambda modes, incl, phase: np.ones(
    (len(modes), len(incl)), dtype=complex)
sys.modules.setdefault("RIFT", _rift)
sys.modules.setdefault("RIFT.likelihood", _likelihood)
sys.modules.setdefault("RIFT.likelihood.factored_likelihood", _fl)
_SPEC = importlib.util.spec_from_file_location(
    "RIFT.likelihood.response_order", _MODULE)
response_order = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = response_order
_SPEC.loader.exec_module(response_order)


class ResponseOrderTest(unittest.TestCase):
    def products(self):
        meta = dict(Qmax=2, p_list=[0, 1, 2, 3], modes=[(2, 2)],
                    event_time_geo=1000000000.0, L_arm=None)
        diagonal = [1.0, 1.0e-2, 1.0e-4, 1.0e-8]
        U = {'H1': {}}
        V = {'H1': {}}
        for p in meta['p_list']:
            for pp in meta['p_list']:
                U['H1'][(p, pp)] = np.array(
                    [[diagonal[p] if p == pp else 0.0]], dtype=complex)
                V['H1'][(p, pp)] = np.zeros((1, 1), dtype=complex)
        primary = {'H1': {p: 'p%d' % p for p in meta['p_list']}}
        return primary, U, V, primary, meta

    def test_snr_tightens_selected_order(self):
        products = self.products()
        coeff = lambda meta, det, ra, dec, psi: (
            np.ones((len(ra), 4), dtype=complex),
            np.ones((len(ra), 4), dtype=complex))
        with mock.patch.object(response_order, '_coefficients', coeff):
            low = response_order.estimate_response_orders(
                products[4], products[1], products[2], 10.0,
                lnL_tolerance=0.1, n_samples=8, selected_q=0,
                vary_p=False, vary_q=True)
            high = response_order.estimate_response_orders(
                products[4], products[1], products[2], 100.0,
                lnL_tolerance=0.1, n_samples=8, selected_q=0,
                vary_p=False, vary_q=True)
        self.assertEqual(low['chosen']['Qmax'], 0)
        self.assertEqual(high['chosen']['Qmax'], 1)
        self.assertTrue(low['selected_passes'])
        self.assertFalse(high['selected_passes'])

    def test_truncate_products(self):
        out = response_order.truncate_precompute_products(
            self.products(), p_max=0, q_max=0)
        self.assertEqual(out[4]['p_list'], [0, 1])
        self.assertEqual(out[4]['Qmax'], 0)
        self.assertEqual(set(out[0]['H1']), {0, 1})
        self.assertEqual(set(out[1]['H1']), {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_rotation_truncation_removes_reference_only_harmonics(self):
        indices = [(p, n) for p in (0, 1) for n in range(-3, 4)]
        meta = dict(p_max=1, harmonics=tuple(range(-3, 4)), a_list=indices,
                    modes=[(2, 2)], event_time_geo=1000000000.0,
                    f_sidereal=1.0, post_phase_required=True)
        primary = {'H1': {a: a for a in indices}}
        cross = {'H1': {(a, ap): {(2, 2): 0j} for a in indices for ap in indices}}
        out = response_order.truncate_precompute_products(
            (primary, cross, cross, primary, meta), p_max=0, q_max=0)
        self.assertEqual(out[4]['a_list'], [(0, n) for n in range(-2, 3)])
        self.assertEqual(out[4]['harmonics'], tuple(range(-2, 3)))

    def test_pack_uv_from_raw_does_not_touch_data_bank(self):
        products = self.products()
        raw_u = {'H1': {(p, pp): {((2, 2), (2, 2)): values[0, 0]}
                        for (p, pp), values in products[1]['H1'].items()}}
        raw_v = {'H1': {(p, pp): {((2, 2), (2, 2)): values[0, 0]}
                        for (p, pp), values in products[2]['H1'].items()}}
        U, V = response_order.pack_uv_from_raw(products[4], raw_u, raw_v)
        self.assertEqual(U['H1'][(1, 1)].shape, (1, 1))
        self.assertEqual(U['H1'][(1, 1)][0, 0], 1.0e-2)
        self.assertEqual(V['H1'][(2, 2)][0, 0], 0.0)

    def test_angular_design_spans_independent_coordinates(self):
        design = np.asarray(response_order._angular_design(256))
        unit = np.vstack((design[0] / (2 * np.pi),
                          (np.sin(design[1]) + 1) / 2,
                          (np.cos(design[2]) + 1) / 2,
                          design[3] / np.pi,
                          design[4] / (2 * np.pi)))
        corr = np.corrcoef(unit)
        self.assertLess(np.max(np.abs(corr - np.eye(5))), 0.15)

    def test_reference_bank_guard_refuses_large_compound_bank(self):
        size = response_order.reference_bank_size(
            'combined', p_max=4, q_max=10, lmax=4, n_detectors=5)
        self.assertEqual(size['basis'], 1090)
        self.assertGreater(size['uv_gib'], 10.0)
        with self.assertRaises(ValueError):
            response_order.guard_reference_bank(
                'combined', 4, 10, 4, 5, max_bank_gib=4.0)

    def test_one_shell_is_not_a_resolved_reference(self):
        products = response_order.truncate_precompute_products(
            self.products(), p_max=0, q_max=1)
        coeff = lambda meta, det, ra, dec, psi: (
            np.ones((len(ra), len(meta['p_list'])), dtype=complex),
            np.ones((len(ra), len(meta['p_list'])), dtype=complex))
        with mock.patch.object(response_order, '_coefficients', coeff):
            report = response_order.estimate_response_orders(
                products[4], products[1], products[2], 10.0,
                lnL_tolerance=0.1, n_samples=8, selected_q=0,
                vary_p=False, vary_q=True)
        self.assertFalse(report['reference_resolved'])

    def test_combined_one_axis_resolution_ignores_unvaried_axis(self):
        indices = [(b, 0, 0) for b in range(4)]
        meta = dict(feature='rotation_freqresponse', p_max=0, Qmax=2,
                    a_list=indices, modes=[(2, 2)],
                    event_time_geo=1000000000.0, f_sidereal=1.0)
        diagonal = [1.0, 1.0e-2, 1.0e-4, 1.0e-8]
        U = {'H1': {}}
        V = {'H1': {}}
        for i, a in enumerate(indices):
            for j, ap in enumerate(indices):
                U['H1'][(a, ap)] = np.array(
                    [[diagonal[i] if i == j else 0.0]], dtype=complex)
                V['H1'][(a, ap)] = np.zeros((1, 1), dtype=complex)
        coeff = lambda meta, det, ra, dec, psi: (
            np.ones((len(ra), len(indices)), dtype=complex),
            np.ones((len(ra), len(indices)), dtype=complex))
        with mock.patch.object(response_order, '_coefficients', coeff):
            report = response_order.estimate_response_orders(
                meta, U, V, 10.0, n_samples=8, selected_p=0,
                selected_q=0, vary_p=False, vary_q=True)
        self.assertTrue(report['reference_resolved'])
        self.assertEqual([row['axis'] for row in report['reference_shells']], ['q'])
        for row in report['rows']:
            expected = (np.sqrt(row['finite_reference_mu'])
                        + np.sqrt(report['reference_tail_mu'])) ** 2
            self.assertAlmostEqual(row['max_mu'], expected)
        self.assertGreater(report['rows'][0]['max_mu'],
                           report['rows'][0]['finite_reference_mu'])


if __name__ == '__main__':
    unittest.main()
