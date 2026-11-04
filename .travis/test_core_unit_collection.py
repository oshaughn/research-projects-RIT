"""Bounded accounting tests; these are separate from the 105-file core census."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

import core_unit_collection as collection


@pytest.fixture
def manifest(tmp_path):
    files=['test_a.py','sub/test_b.py']
    (tmp_path/'sub').mkdir()
    for name in files:(tmp_path/name).write_text('')
    return files


def test_exact_files_classes_parameters_and_shortened_rootpath(tmp_path,manifest):
    nodeids=['test_a.py::TestClass::test_method[param::value]',
             r'test_a.py::test_regex[\n]', 'sub/test_b.py::test_plain']
    assert collection.count_nodeids(manifest,nodeids,tmp_path,tmp_path)=={
        'test_a.py':2,'sub/test_b.py':1}
    assert collection.count_nodeids([manifest[1]],['test_b.py::test_plain'],tmp_path/'sub',tmp_path)=={manifest[1]:1}


def test_absent_file_is_counted_zero_not_omitted(tmp_path,manifest):
    assert collection.count_nodeids(manifest,['test_a.py::test_one'],tmp_path,tmp_path)=={
        'test_a.py':1,'sub/test_b.py':0}


@pytest.mark.parametrize('nodeids',[
    [],['test_a.py'],['test_a.py::'],['::test'],['../test_a.py::test'],
    ['/test_a.py::test'],['test_a.py::test\nspoof'],['test_a.py::test','test_a.py::test'],
    ['sub\\test_b.py::test'],[None],['unlisted.py::test'],
])
def test_malformed_outside_empty_and_duplicate_nodeids_fail_closed(tmp_path,manifest,nodeids):
    (tmp_path/'unlisted.py').write_text('')
    with pytest.raises((ValueError,OSError)):
        collection.count_nodeids(manifest,nodeids,tmp_path,tmp_path)


def test_duplicate_manifest_alias_empty_or_unsafe_manifest_refuses(tmp_path,manifest):
    for files in [[],[manifest[0],str(tmp_path/manifest[0])],[manifest[0]+'\t']]:
        with pytest.raises(ValueError):
            collection.count_nodeids(files,['test_a.py::test'],tmp_path,tmp_path)


def invoke(tmp_path,files):
    output=tmp_path/'counts.tsv'
    env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,str(Path(collection.__file__).resolve()),str(output),*files],
                          cwd=tmp_path,env=env,text=True,capture_output=True,timeout=20)
    return result,output


def test_one_joint_real_collection_counts_items_not_warning_stdout(tmp_path):
    (tmp_path/'test_a.py').write_text('''import pytest, warnings
warnings.warn("test_b.py::warning_is_not_a_test")
@pytest.mark.parametrize("x",[1,2])
def test_parameter(x): pass
class TestClass:
 def test_method(self): pass
''')
    (tmp_path/'project_helper.py').write_text('VALUE=42\n')
    (tmp_path/'test_b.py').write_text('import project_helper\ndef test_one(): assert project_helper.VALUE==42\n')
    result,output=invoke(tmp_path,['test_a.py','test_b.py'])
    assert result.returncode==0,result.stdout+result.stderr
    assert 'warning_is_not_a_test' in result.stdout
    assert output.read_text()=='test_a.py\t3\ntest_b.py\t1\n'


@pytest.mark.parametrize('content',[
    'def broken(\n', 'import no_such_core_collection_dependency_123\n',
    'raise RuntimeError("collection failed")\n', '',
])
def test_actual_collection_errors_or_no_tests_never_publish_counts(tmp_path,content):
    (tmp_path/'test_bad.py').write_text(content)
    result,output=invoke(tmp_path,['test_bad.py'])
    assert result.returncode!=0 and not output.exists()


def test_import_skipped_file_remains_zero_for_shell_floor(tmp_path):
    (tmp_path/'test_ok.py').write_text('def test_one(): pass\n')
    (tmp_path/'test_skip.py').write_text('import pytest\npytest.skip("missing dependency",allow_module_level=True)\n')
    result,output=invoke(tmp_path,['test_ok.py','test_skip.py'])
    assert result.returncode==0,result.stdout+result.stderr
    assert output.read_text()=='test_ok.py\t1\ntest_skip.py\t0\n'
