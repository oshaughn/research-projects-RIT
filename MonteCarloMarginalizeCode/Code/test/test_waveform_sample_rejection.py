"""Execute the real ILE per-event boundary around a controlled waveform seam."""
import ast
from pathlib import Path
import types
import numpy as np
import pytest

DRIVER = Path(__file__).resolve().parents[1]/"bin/integrate_likelihood_extrinsic_batchmode"


@pytest.mark.parametrize("failure", ["raise", "nan", "inf", "none"])
def test_waveform_failure_skips_point_and_keeps_neighbors(failure):
    tree = ast.parse(DRIVER.read_text())
    analyze = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'analyze_event')
    guards = [node for node in ast.walk(analyze) if isinstance(node, ast.If) and
              any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and
                  call.func.attr in ('isnan', 'isfinite') and
                  any(isinstance(arg, ast.Name) and arg.id == 'guess_snr' for arg in call.args)
                  for call in ast.walk(node.test))]
    assert len(guards) == 1
    loops = [node for node in tree.body if isinstance(node, ast.For) and
             any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and
                 call.func.id == 'analyze_event' for call in ast.walk(node))]
    assert len(loops) == 1
    loop = loops[0]
    guard = compile(ast.Module(body=guards, type_ignores=[]), str(DRIVER), 'exec')
    calls, published = [], []
    def waveform_seam(points, index, data, psd, fmax, opts):
        calls.append(index)
        if index == 1 and failure == 'raise':
            raise RuntimeError('SimInspiralChooseFDWaveform domain error')
        snr = dict(nan=np.nan, inf=np.inf, none=None).get(failure) if index == 1 else 10.+index
        exec(guard, dict(np=np, guess_snr=snr, indx_event=index))
        published.append((index, snr))
        return snr
    option_names = {node.attr for node in ast.walk(loop) if isinstance(node, ast.Attribute) and
                    isinstance(node.value, ast.Name) and node.value.id == 'opts'}
    opts = types.SimpleNamespace(**{name:False for name in option_names})
    points = [types.SimpleNamespace(print_params=lambda:None) for _ in range(3)]
    namespace = dict(np=np, numpy=np, P_list=points, opts=opts, analyze_event=waveform_seam,
        data_dict={}, psd_dict={}, fmax=1024., lnL_sofar=-np.inf,
        QTimePregridError=type('QTimePregridError', (Exception,), {}), mcsampler_AV_ok=False,
        sampler=types.SimpleNamespace(params=[]), _proposal_field=None,
        _clear_warm_state=lambda sampler:None, no_adapt_sky=False)
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(DRIVER), 'exec'), namespace)
    assert calls == [0, 1, 2]
    assert published == [(0, 10.), (2, 12.)]
    assert namespace['lnL_sofar'] == 12.


@pytest.mark.parametrize("bad", [None, np.nan, np.inf])
def test_nonfinite_analyze_event_return_does_not_poison_next_point(bad):
    tree = ast.parse(DRIVER.read_text())
    loop = next(node for node in tree.body if isinstance(node, ast.For) and
                any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and
                    call.func.id == 'analyze_event' for call in ast.walk(node)))
    calls = []
    def backend(points, index, *args):
        calls.append(index)
        return bad if index == 1 else 2.+index
    names = {node.attr for node in ast.walk(loop) if isinstance(node, ast.Attribute) and
             isinstance(node.value, ast.Name) and node.value.id == 'opts'}
    namespace = dict(np=np, numpy=np,
        P_list=[types.SimpleNamespace(print_params=lambda:None) for _ in range(3)],
        opts=types.SimpleNamespace(**{name:False for name in names}), analyze_event=backend,
        data_dict={}, psd_dict={}, fmax=1024., lnL_sofar=-np.inf,
        QTimePregridError=type('QTimePregridError', (Exception,), {}), mcsampler_AV_ok=False,
        sampler=types.SimpleNamespace(params=[]), _proposal_field=None,
        _clear_warm_state=lambda sampler:None, no_adapt_sky=False)
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(DRIVER), 'exec'), namespace)
    assert calls == [0, 1, 2]
    assert namespace['lnL_sofar'] == 4.


@pytest.mark.parametrize("bad", [None, np.nan, np.inf])
@pytest.mark.parametrize("logarithmic", [False, True])
@pytest.mark.parametrize("failure_kind", ["integral", "variance"])
def test_invalid_native_integral_is_rejected_before_result_file(tmp_path, bad, logarithmic, failure_kind):
    tree = ast.parse(DRIVER.read_text())
    analyze = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'analyze_event')
    def uses_res_finite(node):
        return isinstance(node, ast.If) and any(isinstance(call, ast.Call) and
            isinstance(call.func, ast.Attribute) and call.func.attr == 'isfinite' and
            any(isinstance(arg, ast.Name) and arg.id == 'res' for arg in call.args)
            for call in ast.walk(node.test))
    start = next(i for i,node in enumerate(analyze.body) if uses_res_finite(node))
    guards = analyze.body[start:start+4]
    export_guard = next(node for node in analyze.body if isinstance(node, ast.If) and
        any(isinstance(msg, ast.Constant) and isinstance(msg.value, str) and
            msg.value.startswith("Nonfinite exported likelihood") for msg in ast.walk(node)))
    guards = guards + [export_guard]
    writers = [node for node in ast.walk(analyze) if isinstance(node, ast.Call) and
        isinstance(node.func, ast.Attribute) and node.func.attr == 'savetxt' and
        any(isinstance(name, ast.Name) and name.id == 'log_res' for name in ast.walk(node))]
    if not writers:
        # The newer driver assembles the likelihood row immediately before writing.
        row = next(node for node in ast.walk(analyze) if isinstance(node, ast.AugAssign) and
            isinstance(node.target, ast.Name) and node.target.id == '_row' and
            any(isinstance(name, ast.Name) and name.id == 'log_res' for name in ast.walk(node)))
        writer = next(node for node in ast.walk(analyze) if isinstance(node, ast.Call) and
            isinstance(node.func, ast.Attribute) and node.func.attr == 'savetxt' and
            node.lineno == row.end_lineno + 1)
        publication = [row, ast.Expr(value=writer)]
    else:
        writer = min(writers, key=lambda node:node.lineno)
        publication = [ast.Expr(value=writer)]
    assert guards[-1].end_lineno < writer.lineno
    code = compile(ast.fix_missing_locations(ast.Module(body=guards+publication, type_ignores=[])), str(DRIVER), 'exec')
    paths = []
    for index, result in enumerate((0. if logarithmic else 1., bad, 2.)):
        path = tmp_path/(str(index)+'.dat')
        names = {node.id for node in ast.walk(writer) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        namespace = {name:0. for name in names}
        fields = {node.attr:0. for node in ast.walk(writer) if isinstance(node, ast.Attribute) and
                  isinstance(node.value, ast.Name) and node.value.id == 'P'}
        namespace.update(_row=[0.], manual_avoid_overflow_logarithm=0., neff=1., np=np, numpy=np, res=(result if failure_kind == "integral" or index != 1 else 1.),
            var=(result if failure_kind == "variance" and index == 1 else 0.), return_lnL=logarithmic,
            opts=types.SimpleNamespace(internal_use_lnL=logarithmic),
            P=types.SimpleNamespace(**fields), sampler=types.SimpleNamespace(ntotal=5),
            fname_output_txt=str(path))
        try:
            exec(code, namespace)
        except (ValueError, TypeError):
            assert index == 1
        else:
            paths.append(path)
    assert [path.name for path in paths] == ['0.dat', '2.dat']
    assert not (tmp_path/'1.dat').exists()
    assert all(np.isfinite(np.loadtxt(path)).all() for path in paths)
