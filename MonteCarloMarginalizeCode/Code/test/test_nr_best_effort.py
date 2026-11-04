"""Analytic mixture checks; no waveform library, frames, GPU, or cluster needed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'bin' / 'util_NRBestEffortPosterior.py'
if not SCRIPT.exists():  # also allow the development staging directory
    SCRIPT = Path(__file__).with_name('util_NRBestEffortPosterior.py')
spec = importlib.util.spec_from_file_location('nr_best_effort', SCRIPT)
nr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nr)


def rows(masses=(10., 11., 14.), ratio=1., log_l=1000.):
    return np.array([[-1, m/(1+ratio), m*ratio/(1+ratio), 0,0,0,0,0,0,
                      log_l, .1, 10000, 100] for m in masses])


class NRBestEffortTests(unittest.TestCase):
    def test_demo_resolves_inputs_for_isolated_working_directory(self):
        demo=SCRIPT.parents[1]/'demo/nr_best_effort/bootstrap_existing_run.py'
        if not demo.exists(): demo=SCRIPT.with_name('bootstrap_existing_run.py')
        demo_spec=importlib.util.spec_from_file_location('bootstrap_demo',demo)
        module=importlib.util.module_from_spec(demo_spec);demo_spec.loader.exec_module(module)
        args=['--cache','local.cache','--psd-file','H1=H1.xml.gz','--event','2']
        result=module.absolute_inputs(args,Path('/source/run'))
        self.assertEqual(result,['--cache','/source/run/local.cache','--psd-file','H1=/source/run/H1.xml.gz','--event','2'])
        self.assertEqual(args[1],'local.cache')

    def test_demo_run_never_uses_source_as_working_directory(self):
        demo=SCRIPT.parents[1]/'demo/nr_best_effort/bootstrap_existing_run.py'
        if not demo.exists(): demo=SCRIPT.with_name('bootstrap_existing_run.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original';source.mkdir();out=root/'followup';out.mkdir()
            ile=root/'fake_ile';converter=root/'fake_converter'
            ile.write_text('#!'+sys.executable+'\nimport sys\nfrom pathlib import Path\n'
                'Path("cwd-marker").write_text("here")\n'
                'p=sys.argv[sys.argv.index("--output-file")+1]\nPath(p+"_0_.xml.gz").write_text("fixture")\n'
                'Path(p+"_0_integrator_status.json").write_text(\'{"collapsed":false}\')\n')
            converter.write_text('#!'+sys.executable+'\nprint("# m1 m2 distance")\nprint("5 5 10")\n')
            ile.chmod(0o755);converter.chmod(0o755)
            (out/'config.json').write_text(json.dumps(dict(source_hashes={},samples=1,seed=17,
                source_run=str(source),ile=str(ile),converter=str(converter),args=[],columns=13)))
            subprocess.run([sys.executable,str(demo),'run','--out',str(out),'--index','0'],
                           check=True,capture_output=True)
            self.assertTrue((out/'events/0/cwd-marker').is_file())
            self.assertFalse((source/'cwd-marker').exists())

    def test_demo_rejects_collapsed_or_missing_status(self):
        demo=SCRIPT.parents[1]/'demo/nr_best_effort/bootstrap_existing_run.py'
        if not demo.exists(): demo=SCRIPT.with_name('bootstrap_existing_run.py')
        demo_spec=importlib.util.spec_from_file_location('bootstrap_status_demo',demo)
        module=importlib.util.module_from_spec(demo_spec);demo_spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp)
            with self.assertRaisesRegex(ValueError,'integrator status'): module.posterior_status(job)
            p=job/'ile_0_integrator_status.json'
            p.write_text(json.dumps({'collapsed':True,'collapse_reason':'ESS=1'}))
            with self.assertRaisesRegex(ValueError,'not a valid posterior'): module.posterior_status(job)
            p.write_text(json.dumps({'collapsed':False}))
            self.assertEqual(module.posterior_status(job),p)
    def test_demo_finish_preserves_equal_per_intrinsic_draw_and_rejects_mismatch(self):
        demo=SCRIPT.parents[1]/'demo/nr_best_effort/bootstrap_existing_run.py'
        if not demo.exists(): demo=SCRIPT.with_name('bootstrap_existing_run.py')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'plan').mkdir()
            (out/'config.json').write_text(json.dumps(dict(source_hashes={},samples=2,seed=17)))
            np.savetxt(out/'plan/intrinsic_posterior_samples.dat',[[5,5],[6,6]],header='m1 m2')
            for index,(mass,count,distance) in enumerate([(5,1,10),(6,100,20)]):
                job=out/'events'/str(index);job.mkdir(parents=True)
                np.savetxt(job/'posterior.dat',np.tile([mass,mass,distance],(count,1)),header='m1 m2 distance')
                (job/'ile_0_integrator_status.json').write_text(json.dumps({'collapsed':False}))
            command=[sys.executable,str(demo),'finish','--out',str(out)]
            subprocess.run(command,check=True,capture_output=True)
            result=np.genfromtxt(out/'extrinsic_posterior_samples.dat',names=True)
            np.testing.assert_equal(result['distance'],[10,20])
            (out/'extrinsic_posterior_samples.dat').unlink()
            np.savetxt(out/'events/1/posterior.dat',[[5,5,20]],header='m1 m2 distance')
            result=subprocess.run(command,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'intrinsic mismatch',result.stderr)

    def test_catalog_matches_native_labels_without_reciprocal_q_or_waveform_io(self):
        data=rows(ratio=2.)
        data[:,3]=.3
        data=np.insert(data,9,.2,axis=1)
        plan=nr.make_plan(data,(10,14),eccentric=True)
        def constructor(group,param,metadata_only=False):
            self.assertTrue(metadata_only)
            point=types.SimpleNamespace(m1=1.,m2=2. if param!='reciprocal' else .5,
                s1x=.3,s1y=0.,s1z=0.,s2x=0.,s2y=0.,s2z=0.,eccentricity=.2)
            return types.SimpleNamespace(P=point)
        backend=types.SimpleNamespace(internal_ParametersAvailable={'public':['native','reciprocal']},
            WaveformModeCatalog=constructor)
        report=nr.check_catalog(plan,'public',backend)
        matched=list(report['matches'].values())[0]
        self.assertEqual([m['parameter'] for m in matched],['native'])
        self.assertFalse(report['waveform_io_validated'])
        data[:,9]=.4
        with self.assertRaisesRegex(ValueError,'native catalog metadata'):
            nr.check_catalog(nr.make_plan(data,(10,14),eccentric=True),'public',backend)

    def test_empty_table_is_a_clean_error(self):
        with self.assertRaises(ValueError): nr.make_plan(np.empty((0,13)),(10,14))

    def test_nonuniform_grid_and_constant_evidence(self):
        plan = nr.make_plan(np.vstack([rows(), rows(ratio=.5, log_l=1000+np.log(3))]), (10,14))
        np.testing.assert_allclose([s['posterior_probability'] for s in plan['simulations']], [.75,.25])
        for s in plan['simulations']:
            components = [c for c in plan['components'] if c['simulation'] == s['simulation']]
            np.testing.assert_allclose(nr.probabilities([c['log_weight'] for c in components]), [.125,.5,.375])
        self.assertAlmostEqual(plan['simulations'][1]['log_evidence'], 1000)

    def test_linear_likelihood_integral(self):
        data = rows(log_l=0)
        data[:,9] = np.log([1,2,5])  # L(M)=M-9; integral / range = 3
        plan = nr.make_plan(data, (10,14))
        self.assertAlmostEqual(plan['simulations'][0]['log_evidence'], np.log(3))

    def test_linear_mass_prior(self):
        plan = nr.make_plan(rows(), (10,14), power=1)
        self.assertAlmostEqual(plan['simulations'][0]['log_evidence'],1000)
        np.testing.assert_allclose(nr.probabilities([c['log_weight'] for c in plan['components']]),
                                   np.array([.5,2,1.5])*[10,11,14]/48)

    def test_log_mass_prior(self):
        plan = nr.make_plan(rows(), (10,14), power=-1)
        self.assertTrue(np.isfinite(plan['simulations'][0]['log_evidence']))

    def test_simulation_prior(self):
        data = np.vstack([rows(),rows(ratio=.5)])
        plan = nr.make_plan(data,(10,14))
        priors = {s['simulation']: float(i+1) for i,s in enumerate(plan['simulations'])}
        plan = nr.make_plan(data,(10,14), simulation_priors=priors)
        np.testing.assert_allclose([s['posterior_probability'] for s in plan['simulations']], [1/3,2/3])

    def test_reject_duplicate_and_missing_mass_support(self):
        for data in [np.vstack([rows(), rows()[:1]]), rows((11,12,14)), rows((10,11,13))]:
            with self.assertRaises(ValueError):
                nr.make_plan(data,(10,14))

    def test_reject_bad_input(self):
        for data in [rows()[:,:12], rows()*np.nan, rows()[:1]]:
            with self.assertRaises(ValueError):
                nr.make_plan(data,(10,14))
        for bounds in [(0,14),(14,10),(10,float('inf'))]:
            with self.assertRaises(ValueError):
                nr.make_plan(rows(),bounds)

    def test_partial_mass_support_keeps_common_prior_normalization(self):
        data=np.vstack([rows(),rows((11,12,14),ratio=.5)])
        plan=nr.make_plan(data,(10,14),allow_partial_mass_range=True)
        np.testing.assert_allclose([s['posterior_probability'] for s in plan['simulations']],[3/7,4/7])
        self.assertFalse(plan['simulations'][0]['complete_mass_range'])

    def test_eccentric_and_anomaly_are_distinct_spokes(self):
        data = np.insert(rows(), 9, .2, axis=1)
        data = np.insert(data, 10, .1, axis=1)
        other = data.copy()
        other[:,10] = .7
        plan = nr.make_plan(np.vstack([data,other]),(10,14), eccentric=True, mean_per_ano=True)
        self.assertEqual(len(plan['simulations']),2)
        self.assertEqual(plan['components'][0]['intrinsic']['eccentricity'], .2)

    def test_sigma_does_not_change_catalog_prior(self):
        data = rows()
        plan = nr.make_plan(data,(10,14))
        data[:,10] = [.01,.3,.8]
        other = nr.make_plan(data,(10,14))
        self.assertEqual([c['log_weight'] for c in plan['components']],
                         [c['log_weight'] for c in other['components']])

    def test_ratio_rounding_boundary_does_not_split_spoke(self):
        data=rows(ratio=.50004998)
        data[1]=rows(ratio=.50005002)[1]
        plan=nr.make_plan(data,(10,14))
        self.assertEqual(len(plan['simulations']),1)
        other=nr.make_plan(data[::-1],(10,14))
        self.assertEqual(plan,other)

    def test_repeat_coalescence_matches_known_variance_weighted_mean(self):
        data=rows()
        repeat=data[:1].copy()
        repeat[:,9]+=np.log(2)
        repeat[:,10]=.05  # same absolute sigma as original -> equal precision
        plan=nr.make_plan(np.vstack([data,repeat]),(10,14),coalesce_repeats=True)
        first=plan['components'][0]
        self.assertAlmostEqual(first['log_marginal_likelihood'],1000+np.log(1.5))
        self.assertAlmostEqual(first['relative_error'],.1/np.sqrt(2)/1.5)
        self.assertEqual(plan['simulations'][0]['combined_repeat_rows'],1)

    def test_mass_halfway_rounding_uses_one_quantizer(self):
        plan=nr.make_plan(rows((100.,100.00005,101.,104.)),(100,104),coalesce_repeats=True)
        self.assertEqual(plan['simulations'][0]['combined_repeat_rows'],1)

    def test_pruned_grouping_matches_full_group_scan(self):
        pieces=[]
        for ratio in [.2,.33335,.66695,1.,1.3,2.]:
            for spin in [0.,.2,.4]:
                data=rows(ratio=ratio)
                data[:,3]=spin
                pieces.append(data)
        data=np.vstack(pieces)
        plan=nr.make_plan(data,(10,14))
        with patch.object(nr.np,'searchsorted',return_value=0):
            reference=nr.make_plan(data[::-1],(10,14))
        self.assertEqual(plan,reference)

    def test_log_weight_validation(self):
        for logs in [[],[np.nan],[np.inf],[-np.inf]]:
            with self.assertRaises(ValueError):
                nr.probabilities(logs)
        np.testing.assert_allclose(nr.probabilities([1000,1000-np.log(3)]),[.75,.25])

    def setup_pools(self, directory):
        # Unequal conditional sample counts must NOT bias the mixture.
        np.savetxt(directory/'a.dat', np.ones((10,1)),header='distance')
        np.savetxt(directory/'b.dat', np.zeros((1000,1)),header='distance')
        return dict(schema_version=1,sample_kind='fairdraw',components=[
            dict(id=0,log_weight=1000+np.log(3),samples=['a.dat']),
            dict(id=1,log_weight=1000,samples=['b.dat'])])

    def test_combination_ignores_pool_size_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plan = self.setup_pools(directory)
            result,ids,stats = nr.combine(plan,directory,30000,71)
            self.assertAlmostEqual(result['distance'].mean(),.75,delta=.01)
            other,other_ids,_ = nr.combine(plan,directory,30000,71)
            np.testing.assert_equal(result,other)
            np.testing.assert_equal(ids,other_ids)
            self.assertEqual(stats['0']['available'],10)

    def test_combine_checks_attached_integrator_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);plan=self.setup_pools(directory)
            path=directory/'status.json';path.write_text(json.dumps({'collapsed':True}))
            plan['components'][0]['status_files']=['status.json']
            with self.assertRaisesRegex(ValueError,'not a valid posterior'):
                nr.combine(plan,directory,10,0)
            path.write_text(json.dumps({'collapsed':False}))
            _,_,stats=nr.combine(plan,directory,10,0)
            self.assertEqual(stats['0']['status_files'][0]['path'],str(path.resolve()))

    def test_missing_pool_is_never_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            plan=self.setup_pools(directory)
            plan['components'][1]['log_weight']=-100
            plan['components'][1]['samples']=[]
            with self.assertRaisesRegex(ValueError,'missing fairdraw'):
                nr.combine(plan,directory,1,0)

    def test_reject_reused_file_raw_samples_schema_and_intrinsic_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            plan=self.setup_pools(directory)
            plan['sample_kind']='importance'
            with self.assertRaises(ValueError): nr.combine(plan,directory,10,0)
            plan['sample_kind']='fairdraw'
            plan['components'][1]['samples']=['a.dat']
            with self.assertRaisesRegex(ValueError,'more than once'): nr.combine(plan,directory,10,0)
            plan['components'][1]['samples']=['b.dat']
            np.savetxt(directory/'b.dat', [[0]],header='ra')
            with self.assertRaisesRegex(ValueError,'schemas differ'): nr.combine(plan,directory,10,0)
            np.savetxt(directory/'b.dat', [[0]],header='distance')
            plan['components'][1]['intrinsic']={'m1':10}
            with self.assertRaisesRegex(ValueError,'lacks intrinsic'): nr.combine(plan,directory,10,0)
            np.savetxt(directory/'a.dat', [[10,10]],header='distance m1')
            np.savetxt(directory/'b.dat', [[0,12]],header='distance m1')
            with self.assertRaisesRegex(ValueError,'intrinsic mismatch'): nr.combine(plan,directory,10,0)

    def test_cli_bootstrap_and_combine(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            np.savetxt(directory/'all.net',np.vstack([rows(),rows(ratio=.5,log_l=1000+np.log(3))]))
            command=[sys.executable,str(SCRIPT)]
            subprocess.run(command+['plan','--composite',str(directory/'all.net'),
                '--mass-range','10','14','--output-dir',str(directory/'planned')],check=True,capture_output=True)
            manifest=directory/'planned'/'manifest.json'
            plan=json.loads(manifest.read_text())
            for c in plan['components']:
                path=manifest.parent/('cell-{}.dat'.format(c['id']))
                intrinsic=c['intrinsic']
                np.savetxt(path, [list(intrinsic.values())+[c['mass']]],header=' '.join(list(intrinsic)+['distance']))
                c['samples']=[path.name]
            manifest.write_text(json.dumps(plan))
            output=directory/'posterior.dat'
            subprocess.run(command+['combine','--manifest',str(manifest),'--output',str(output)],check=True,capture_output=True)
            samples=np.genfromtxt(output,names=True)
            self.assertEqual(len(samples),3000)
            self.assertTrue(Path(str(output)+'.provenance.json').exists())
            self.assertEqual(len(np.loadtxt(str(output)+'.component_ids')),3000)
            rerun=subprocess.run(command+['combine','--manifest',str(manifest),'--output',str(output)],capture_output=True)
            self.assertNotEqual(rerun.returncode,0)
            self.assertIn(b'refusing to overwrite',rerun.stderr)

    def test_xml_roundtrip_when_lalsuite_available(self):
        try:
            import lal
            from RIFT import lalsimutils
        except ImportError:
            self.skipTest('XML integration requires installed RIFT and lalsuite')
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            templates=[]
            for ratio in [1.,.5]:
                point=lalsimutils.ChooseWaveformParams()
                point.m1=10/(1+ratio)*lal.MSUN_SI
                point.m2=10*ratio/(1+ratio)*lal.MSUN_SI
                point.fref=20.
                point.fmin=15.
                point.tref=lal.LIGOTimeGPS(1242442967.4)
                templates.append(point)
            lalsimutils.ChooseWaveformParams_array_to_xml(templates,str(directory/'template'))
            plan=nr.make_plan(np.vstack([rows(),rows(ratio=.5)]),(10,14))
            nr.export_xml(plan,[0,3,2],directory/'template.xml.gz',directory)
            grid=lalsimutils.xml_to_ChooseWaveformParams_array(str(directory/'quadrature-grid.xml.gz'))
            posterior=lalsimutils.xml_to_ChooseWaveformParams_array(str(directory/'intrinsic-posterior.xml.gz'))
            self.assertEqual(len(grid),6)
            self.assertEqual(len(posterior),3)
            for point,component in zip(grid,plan['components']):
                self.assertAlmostEqual(point.m1/lal.MSUN_SI,component['intrinsic']['m1'],places=5)
                self.assertAlmostEqual(point.m2/lal.MSUN_SI,component['intrinsic']['m2'],places=5)
                self.assertAlmostEqual(float(point.tref),1242442967.4,places=5)
            only=directory/'only'
            only.mkdir()
            nr.export_xml(plan,[0,3,2],directory/'template.xml.gz',only,posterior_only=True)
            self.assertTrue((only/'intrinsic-posterior.xml.gz').exists())
            self.assertFalse((only/'quadrature-grid.xml.gz').exists())
            settings=directory/'settings'
            settings.mkdir()
            eccentric_data=np.insert(rows(ratio=.25),9,.3,axis=1)
            eccentric_data[:,3]=.2
            other_plan=nr.make_plan(eccentric_data,(10,14),eccentric=True)
            nr.export_xml(other_plan,[0,2],directory/'template.xml.gz',settings,
                          posterior_only=True,settings_only=True)
            exported=lalsimutils.xml_to_ChooseWaveformParams_array(str(settings/'intrinsic-posterior.xml.gz'))
            self.assertAlmostEqual(exported[0].eccentricity,.3)
            self.assertAlmostEqual(exported[0].s1x,.2)
            self.assertAlmostEqual(exported[0].m2/exported[0].m1,.25)


if __name__ == '__main__':
    unittest.main()
