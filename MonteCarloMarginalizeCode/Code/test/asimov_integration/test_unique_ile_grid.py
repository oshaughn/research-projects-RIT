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
