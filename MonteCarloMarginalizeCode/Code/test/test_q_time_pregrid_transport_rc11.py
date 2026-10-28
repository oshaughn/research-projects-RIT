"""Exercise rc11's actual parser/emission AST without executing data preparation."""
import argparse
import ast
import importlib.util
from pathlib import Path
import pytest

CODE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("q_pregrid_transport",CODE/"RIFT/likelihood/q_time_pregrid.py")
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
FLAG="--internal-ile-q-time-pregrid-factor"

def tree(name):
    return ast.parse((CODE/"bin"/name).read_text())

def parser_for(name):
    parser=argparse.ArgumentParser()
    calls=[n for n in ast.walk(tree(name)) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr=="add_argument"
           and any(isinstance(a,ast.Constant) and a.value==FLAG for a in n.args)]
    assert len(calls)==1
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=calls[0])],type_ignores=[])),
                 name,"exec"),dict(parser=parser,Q_TIME_PREGRID_CHOICES=q.Q_TIME_PREGRID_CHOICES))
    return parser

def emit(name,target,ns):
    nodes=[]
    for node in tree(name).body:
        if isinstance(node,ast.If) and any(
            isinstance(n,(ast.Assign,ast.AugAssign)) and
            any(isinstance(t,ast.Name) and t.id==target for t in
                (n.targets if isinstance(n,ast.Assign) else [n.target])) and
            "q-time-pregrid-factor" in ast.unparse(n)
            for n in node.body):
            nodes.append(node)
    assert len(nodes)==1
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),name,"exec"),ns)
    return ns[target]

@pytest.mark.parametrize("factor",q.Q_TIME_PREGRID_CHOICES)
def test_pipeline_helper_emission_roundtrip(factor):
    pipe=parser_for("util_RIFT_pseudo_pipe.py").parse_args([FLAG,str(factor)])
    cmd=emit("util_RIFT_pseudo_pipe.py","cmd",dict(opts=pipe,cmd="helper_LDG_Events.py "))
    helper=parser_for("helper_LDG_Events.py").parse_args(cmd.split()[1:])
    args=emit("helper_LDG_Events.py","helper_ile_args",
              dict(opts=helper,helper_ile_args="--time-marginalization --vectorized --gpu --srate 4096 "))
    assert args.count("--q-time-pregrid-factor")==1
    assert "--q-time-pregrid-factor "+str(factor) in args
    q.refuse_unless_q_time_pregrid_emitted(factor,args,"args_ile.txt")
    assert "--srate 4096" in args and "--srate-internal" not in args

def test_unset_preserves_argument_bytes():
    for name,target,value in [("util_RIFT_pseudo_pipe.py","cmd","helper_LDG_Events.py "),
                              ("helper_LDG_Events.py","helper_ile_args","--time-marginalization --vectorized --gpu ")]:
        opts=parser_for(name).parse_args([])
        assert emit(name,target,dict(opts=opts,**{target:value}))==value

@pytest.mark.parametrize("name",["util_RIFT_pseudo_pipe.py","helper_LDG_Events.py"])
def test_unsupported_factor_rejected_by_actual_parser(name):
    with pytest.raises(SystemExit):
        parser_for(name).parse_args([FLAG,"3"])

def test_stale_helper_lost_request_is_refused():
    with pytest.raises(ValueError,match="contains no"):
        q.refuse_unless_q_time_pregrid_emitted(2,"--time-marginalization --vectorized --gpu","args_ile.txt")

@pytest.mark.parametrize("extra",["--zero-likelihood","--rotation-slow","--freqresponse",
    "--calibration-envelope-directory draws","--interpolate-time nearest"])
def test_incompatible_final_arguments_are_refused(extra):
    with pytest.raises(ValueError):
        q.refuse_unless_q_time_pregrid_emitted(
            2,"--time-marginalization --vectorized --gpu --q-time-pregrid-factor 2 "+extra,"args_ile.txt")

def test_both_actual_scripts_guard_before_writing():
    for name,filename in [("util_RIFT_pseudo_pipe.py","args_ile.txt"),
                          ("helper_LDG_Events.py","helper_ile_args.txt")]:
        nodes=tree(name).body
        writes=[i for i,n in enumerate(nodes) if isinstance(n,ast.With)
                and filename in ast.unparse(n.items)]
        assert len(writes)==1
        guard=nodes[writes[0]-1]
        assert isinstance(guard,ast.Expr)
        assert isinstance(guard.value,ast.Call)
        assert guard.value.func.id=="refuse_unless_q_time_pregrid_emitted"


@pytest.mark.parametrize("factor",[None,2,4,8])
def test_asimov_template_renders_q_only_key(factor):
    liquid=pytest.importorskip("liquid")
    template=(CODE/"RIFT/asimov/rift.ini").read_text()
    start=template.index("# Q-only pregrid:")
    stop=template.index("{% if sampler contains 'ile' %} {% if sampler['ile'] contains 'rotate phase' %}",start)
    source=template[start:stop]
    sampler={"ile":{}}
    if factor is not None:
        sampler["ile"]["q time pregrid factor"]=factor
    if hasattr(liquid,"Liquid"):
        rendered=liquid.Liquid(source,from_file=False).render(sampler=sampler)
    else:
        rendered=liquid.Environment().from_string(source).render(sampler=sampler)
    if factor is None:
        assert "internal-ile-q-time-pregrid-factor=" not in rendered
    else:
        assert "internal-ile-q-time-pregrid-factor="+str(factor) in rendered
    assert "srate-internal" not in rendered


@pytest.mark.parametrize("value", [True, False, 1.5, 2.0, 2.9, 8.1, float("inf"), float("nan")])
def test_fractional_or_boolean_refinement_requests_are_rejected(value):
    from RIFT.likelihood.q_time_pregrid import validate_q_time_pregrid_factor
    with pytest.raises(ValueError, match="must be an integer"):
        validate_q_time_pregrid_factor(value)


@pytest.mark.parametrize("factor", [2, 4, 8])
@pytest.mark.parametrize("missing", ["--time-marginalization", "--vectorized", "--gpu"])
def test_refined_emission_requires_actual_noloop_dispatch(factor, missing):
    args = ["--time-marginalization", "--vectorized", "--gpu"]
    args.remove(missing)
    with pytest.raises(ValueError, match=missing):
        q.refuse_unless_q_time_pregrid_emitted(factor,
            "X " + " ".join(args) + " --q-time-pregrid-factor " + str(factor), "args_ile.txt")
