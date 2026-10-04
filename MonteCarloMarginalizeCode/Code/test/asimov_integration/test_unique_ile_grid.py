"""A unique likelihood grid must preserve the separate posterior weights."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
SPEC=importlib.util.spec_from_file_location('intrinsic_grid',Path(__file__).resolve().parents[2]/'RIFT/misc/intrinsic_grid.py')
GRID=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(GRID)

def point(**changes):
    fields={name:0. for name in GRID.INTRINSIC_FIELDS};fields.update(m1=30.,m2=20.,fref=20.);fields.update(changes)
    return SimpleNamespace(**fields)

def test_grid_only_exact_identity_preserves_posterior_and_distinct_spin():
    posterior=[point(),point(),point(s1x=.3),point(s1x=.3+1e-15),point()]
    keys_before=[vars(p).copy() for p in posterior]
    indices=GRID.unique_intrinsic_indices(posterior)
    assert indices==[0,2,3]
    assert len(posterior)==5
    assert [vars(p) for p in posterior]==keys_before
    assert posterior[0] is not posterior[1]

def test_extrinsic_variation_does_not_launch_duplicate_intrinsic():
    a,b=point(),point();a.incl=0;b.incl=1
    assert GRID.unique_intrinsic_indices([a,b])==[0]

@pytest.mark.parametrize('field',['m1','s2z','lambda2','eccentricity','meanPerAno','fref'])
def test_all_intrinsic_differences_retained(field):
    assert GRID.unique_intrinsic_indices([point(),point(**{field:1.})])==[0,1]

def test_bad_grid_rejected():
    with pytest.raises(ValueError):GRID.unique_intrinsic_indices([point(s1x=float('nan'))])


def test_final_extrinsic_uses_weighted_posterior_grid():
    import ast
    source = (Path(__file__).resolve().parents[2] / "bin/create_event_parameter_pipeline_BasicIteration").read_text()
    tree = ast.parse(source)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    extr = next(n for n in assigns if any(isinstance(t, ast.Name) and t.id == "ile_args_extr" for t in n.targets))
    assert ast.unparse(extr.value).startswith("ile_args_forposterior +")
    posterior = next(n for n in assigns if any(isinstance(t, ast.Name) and t.id == "ile_args_forposterior" for t in n.targets))
    assert "overlap-grid-$(macroiteration)" in ast.unparse(posterior.value)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and any(k.arg == "tag" and isinstance(k.value, ast.Constant) and k.value.value == "ILE_extr" for k in n.keywords))
    assert ast.unparse(next(k.value for k in call.keywords if k.arg == "transfer_files")) == "transfer_file_names_extr"
    assert "name.replace('../'+ordinary_grid_base" in source
    assert "'../overlap-grid-$(macroiteration)" in source


def test_short_unique_grid_pads_with_duplicates_instead_of_failing():
    posterior=[point(),point(),point(s1x=.3),point(),point(s1x=.3)]
    indices=GRID.unique_intrinsic_indices(posterior)
    assert GRID.pad_with_duplicates(indices,5,2)==[0,2]
    assert GRID.pad_with_duplicates(indices,5,4)==[0,2,1,3]
    assert GRID.pad_with_duplicates(indices,5,7)==[0,2,1,3,4,1,3]
    assert GRID.pad_with_duplicates([0,1],2,5)==[0,1,0,1,0]


def test_dedup_cli_pads_short_grid(tmp_path):
    import json, subprocess, sys
    lalsimutils = pytest.importorskip('RIFT.lalsimutils')
    rows = []
    for s1z in (0., 0., 0.1):
        P = lalsimutils.ChooseWaveformParams(); P.m1, P.m2, P.s1z = 30*lalsimutils.lsu_MSUN, 20*lalsimutils.lsu_MSUN, s1z
        rows.append(P)
    lalsimutils.ChooseWaveformParams_array_to_xml(rows, str(tmp_path/'post'))
    exe = Path(__file__).resolve().parents[2]/'bin/util_DeduplicateIntrinsicGrid.py'
    out = subprocess.run([sys.executable, str(exe), '--input-xml', str(tmp_path/'post.xml.gz'), '--output-file', str(tmp_path/'grid'),
                          '--min-points', '4', '--receipt', str(tmp_path/'r.json')], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert 'WARNING' in out.stdout
    record = json.loads((tmp_path/'r.json').read_text())
    assert record['status'] == 'padded_with_duplicates' and record['unique_rows'] == 2
    assert len(lalsimutils.xml_to_ChooseWaveformParams_array(str(tmp_path/'grid.xml.gz'))) == 4
