"""Build realistic-looking pipeline outputs under each backend.

This script uses :mod:`RIFT.misc.dag_utils_generic` directly to construct
workflows analogous to what the production pipeline drivers (
``create_event_parameter_pipeline_BasicIteration``,
``create_eos_posterior_pipeline``, ``util_RIFT_pseudo_pipe.py``) emit, and
writes the resulting submit / sbatch / DAG artefacts under
``test/backends/outputs/<backend>/<pipeline>/``.

Unlike the shell demos in this directory, this script does NOT require the
science stack (``lalsuite``, ``scipy``, ...) -- it exercises the DAG-emission
layer in isolation.  It is therefore useful for:

  * smoke-testing that all three backends accept the same inputs;
  * showing exactly what each backend produces for a given workflow;
  * sanity-checking new backends written against the registry API.

Run::

    cd MonteCarloMarginalizeCode/Code/test/backends
    python3 synthesize_pipeline_outputs.py

Outputs land in ``outputs/<backend>/<pipeline>/``.  Inspect with
``less`` / ``cat`` to compare backends side-by-side.
"""

import os
import shutil
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
DAG_PATH = os.path.normpath(
    os.path.join(HERE, "..", "..", "RIFT", "misc", "dag_utils_generic.py")
)
OUTPUTS = os.path.join(HERE, "outputs")


def _load_module():
    sys.modules.pop("dag_utils_generic", None)
    spec = importlib.util.spec_from_file_location("dag_utils_generic", DAG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_in_htcondor():
    """Provide a minimal `htcondor` module so the htcondor backend can run.

    Real installs use the actual python bindings; the stub is only here so
    you can run this synthesizer on a laptop without `pip install htcondor`.
    """
    if "htcondor" in sys.modules:
        return
    try:
        import htcondor  # noqa: F401
        return
    except ImportError:
        pass
    import types
    stub = types.ModuleType("htcondor")

    class _Submit(object):
        def __init__(self, d):
            self._d = dict(d)

        def __str__(self):
            return "\n".join("{} = {}".format(k, v) for k, v in self._d.items())

    stub.Submit = _Submit
    sys.modules["htcondor"] = stub


def _stub_in_glue():
    """Provide a minimal `glue.pipeline` module so the glue backend can run."""
    try:
        from glue import pipeline  # noqa: F401
        return
    except ImportError:
        pass
    import types
    src = '''
class CondorDAGJob(object):
    def __init__(self, universe="vanilla", executable=None):
        self.universe = universe; self.executable = executable
        self.opts=[]; self.short_opts=[]; self.file_opts=[]; self.var_opts=[]
        self.args=[]; self.condor_cmds=[]
        self.sub_file=None; self.log_file=None; self.stdout=None; self.stderr=None
        self._CondorJob__queue = 1
    def set_sub_file(self, f): self.sub_file=f
    def get_sub_file(self): return self.sub_file
    def set_log_file(self, f): self.log_file=f
    def set_stdout_file(self, f): self.stdout=f
    def set_stderr_file(self, f): self.stderr=f
    def add_opt(self,n,v=None): self.opts.append((n,v))
    def add_short_opt(self,n,v): self.short_opts.append((n,v))
    def add_var_opt(self,n): self.var_opts.append(n)
    def add_file_opt(self,n,v): self.file_opts.append((n,v))
    def add_arg(self,a): self.args.append(a)
    def add_condor_cmd(self,k,v): self.condor_cmds.append((k,v))
    def write_sub_file(self):
        with open(self.sub_file,"w") as fh:
            fh.write("# (glue.pipeline stub backend output)\\n")
            fh.write("universe = {}\\n".format(self.universe))
            fh.write("executable = {}\\n".format(self.executable))
            parts=["--{}={}".format(k,v) if v is not None else "--{}".format(k) for k,v in self.opts]
            parts+=["-{} {}".format(k,v) for k,v in self.short_opts]
            parts+=["--{}={}".format(k,v) for k,v in self.file_opts]
            parts+=["--{}=$(macro{})".format(n,n) for n in self.var_opts]
            parts+=[str(a) for a in self.args]
            if parts: fh.write("arguments = " + " ".join(parts) + "\\n")
            if self.log_file: fh.write("log = {}\\n".format(self.log_file))
            if self.stdout:   fh.write("output = {}\\n".format(self.stdout))
            if self.stderr:   fh.write("error = {}\\n".format(self.stderr))
            for k,v in self.condor_cmds:
                fh.write("{} = {}\\n".format(k,v))
            fh.write("queue {}\\n".format(self._CondorJob__queue))

class CondorDAG(object):
    def __init__(self, log=None): self.nodes=[]; self.dag_file=None; self.log=log
    def add_node(self,n): self.nodes.append(n)
    def set_dag_file(self,f): self.dag_file=f
    def write_concrete_dag(self):
        path = self.dag_file if self.dag_file.endswith(".dag") else self.dag_file + ".dag"
        with open(path,"w") as fh:
            fh.write("# (glue.pipeline stub DAG output)\\n")
            for n in self.nodes:
                if getattr(n,"job",None) is not None:
                    fh.write("JOB {} {}\\n".format(n.name, n.job.sub_file))
                else:
                    fh.write("SUBDAG EXTERNAL {} {}\\n".format(n.name, getattr(n,"f","<unknown>")))
                for k,v in n.macros.items():
                    fh.write('VARS {} {}="{}"\\n'.format(n.name,k,v))
                if n.retry: fh.write("RETRY {} {}\\n".format(n.name, n.retry))
            for n in self.nodes:
                for p in n.parents:
                    fh.write("PARENT {} CHILD {}\\n".format(p.name, n.name))

class CondorDAGNode(object):
    _ctr=0
    def __init__(self,job):
        CondorDAGNode._ctr += 1
        self.name = "g_{}".format(CondorDAGNode._ctr)
        self.job=job; self.macros={}; self.parents=[]; self.cat=None; self.retry=0
    def add_macro(self,k,v): self.macros[k]=v
    def set_category(self,c): self.cat=c
    def set_retry(self,n): self.retry=n
    def add_parent(self,p): self.parents.append(p)

class CondorDAGManJob(object):
    def __init__(self, f): self.f = f
    def create_node(self):
        n = CondorDAGNode.__new__(CondorDAGNode)
        CondorDAGNode._ctr += 1
        n.name = "g_{}".format(CondorDAGNode._ctr)
        n.job=None; n.macros={}; n.parents=[]; n.cat=None; n.retry=0
        n.f = self.f
        return n
'''
    pkg = types.ModuleType("glue")
    pkg.__path__ = []
    pipeline = types.ModuleType("glue.pipeline")
    exec(src, pipeline.__dict__)
    pkg.pipeline = pipeline
    sys.modules["glue"] = pkg
    sys.modules["glue.pipeline"] = pipeline


def build_basic_iteration_pipeline(m, outdir, n_iterations=2):
    """Mimic create_event_parameter_pipeline_BasicIteration.

    Per iteration we have:  ILE  →  CIP  →  test (convergence)  →  puff
    where ILE is "exploded" (queue N) and parent-child chains glue iterations.
    """
    os.makedirs(outdir, exist_ok=True)
    dag = m.CondorDAG()

    # Init job: writes the seed grid
    init_job = m.CondorDAGJob(universe="vanilla", executable="/usr/bin/cp")
    init_job.set_sub_file(os.path.join(outdir, "init.sub"))
    init_job.set_log_file("logs/init.log")
    init_job.set_stdout_file("logs/init.out")
    init_job.set_stderr_file("logs/init.err")
    init_job.add_arg("overlap-grid.xml.gz")
    init_job.add_arg("overlap-grid-0.xml.gz")
    init_job.add_condor_cmd("request_memory", "256")
    init_job.add_condor_cmd("getenv", "True")
    init_job.write_sub_file()
    init_node = m.CondorDAGNode(init_job)
    dag.add_node(init_node)

    prev_terminal = init_node
    for it in range(n_iterations):
        # ILE job (queue 8 → 8 parallel ILE workers per iteration)
        ile_job = m.CondorDAGJob(
            universe="vanilla",
            executable="integrate_likelihood_extrinsic_batchmode",
        )
        ile_job.set_sub_file(os.path.join(outdir, "ILE-{}.sub".format(it)))
        ile_job.set_log_file("logs/ile-{}.$(cluster).$(process).log".format(it))
        ile_job.set_stdout_file("logs/ile-{}.$(cluster).$(process).out".format(it))
        ile_job.set_stderr_file("logs/ile-{}.$(cluster).$(process).err".format(it))
        ile_job.add_opt("inj-xml", "overlap-grid-{}.xml.gz".format(it))
        ile_job.add_opt("output-file", "ile-{}-$(cluster)-$(process)".format(it))
        ile_job.add_var_opt("event")
        ile_job.add_arg("--time-marginalization")
        ile_job.add_arg("--vectorized")
        ile_job.add_condor_cmd("request_memory", "8192M")
        ile_job.add_condor_cmd("request_disk", "1024M")
        ile_job.add_condor_cmd("request_cpus", "2")
        ile_job.add_condor_cmd("request_gpus", "1")
        ile_job.add_condor_cmd("getenv", "True")
        ile_job.add_condor_cmd("accounting_group", "ligo.dev.o4.cbc.pe.lalinference")
        ile_job.add_condor_cmd("+SlurmPartition", '"gpu"')
        ile_job._CondorJob__queue = 8
        ile_job.write_sub_file()

        ile_node = m.CondorDAGNode(ile_job)
        ile_node.add_macro("event", str(it))
        ile_node.set_retry(2)
        ile_node.add_parent(prev_terminal)
        dag.add_node(ile_node)

        # CIP job: fits the marginal-likelihood evaluations into a posterior
        cip_job = m.CondorDAGJob(
            universe="vanilla",
            executable="util_ConstructIntrinsicPosterior_GenericCoordinates.py",
        )
        cip_job.set_sub_file(os.path.join(outdir, "CIP-{}.sub".format(it)))
        cip_job.set_log_file("logs/cip-{}.log".format(it))
        cip_job.set_stdout_file("logs/cip-{}.out".format(it))
        cip_job.set_stderr_file("logs/cip-{}.err".format(it))
        cip_job.add_opt("input-net", "all-iter-{}.net".format(it))
        cip_job.add_opt("output-file", "overlap-grid-{}.xml.gz".format(it + 1))
        cip_job.add_arg("--n-eff")
        cip_job.add_arg("5000")
        cip_job.add_condor_cmd("request_memory", "16384M")
        cip_job.add_condor_cmd("request_disk", "2048M")
        cip_job.add_condor_cmd("request_cpus", "4")
        cip_job.add_condor_cmd("getenv", "True")
        cip_job.add_condor_cmd("accounting_group", "ligo.dev.o4.cbc.pe.lalinference")
        cip_job.add_condor_cmd("+SlurmPartition", '"compute"')
        cip_job.write_sub_file()

        cip_node = m.CondorDAGNode(cip_job)
        cip_node.add_parent(ile_node)
        cip_node.set_retry(1)
        dag.add_node(cip_node)

        # Convergence test
        test_job = m.CondorDAGJob(universe="vanilla", executable="convergence_test_samples")
        test_job.set_sub_file(os.path.join(outdir, "TEST-{}.sub".format(it)))
        test_job.set_log_file("logs/test-{}.log".format(it))
        test_job.set_stdout_file("logs/test-{}.out".format(it))
        test_job.set_stderr_file("logs/test-{}.err".format(it))
        test_job.add_arg("--threshold 0.02")
        test_job.add_condor_cmd("request_memory", "1024")
        test_job.add_condor_cmd("getenv", "True")
        test_job.write_sub_file()
        test_node = m.CondorDAGNode(test_job)
        test_node.add_parent(cip_node)
        dag.add_node(test_node)

        # Puffball (parameter perturbation for exploration)
        puff_job = m.CondorDAGJob(universe="vanilla", executable="util_ParameterPuffball.py")
        puff_job.set_sub_file(os.path.join(outdir, "PUFF-{}.sub".format(it)))
        puff_job.set_log_file("logs/puff-{}.log".format(it))
        puff_job.set_stdout_file("logs/puff-{}.out".format(it))
        puff_job.set_stderr_file("logs/puff-{}.err".format(it))
        puff_job.add_opt("input", "overlap-grid-{}.xml.gz".format(it + 1))
        puff_job.add_opt("output", "overlap-grid-{}.xml.gz".format(it + 1))
        puff_job.add_condor_cmd("request_memory", "1024M")
        puff_job.add_condor_cmd("getenv", "True")
        puff_job.write_sub_file()
        puff_node = m.CondorDAGNode(puff_job)
        puff_node.add_parent(test_node)
        dag.add_node(puff_node)

        prev_terminal = puff_node

    dag.set_dag_file(os.path.join(outdir, "marginalize_intrinsic_parameters.dag"))
    dag.write_concrete_dag()


def build_eos_posterior_pipeline(m, outdir, n_events=2):
    """Mimic create_eos_posterior_pipeline.

    n_events parallel marg-event jobs fan-out from a single eos-posterior
    job, with a convergence test downstream of the eos-post.
    """
    os.makedirs(outdir, exist_ok=True)
    dag = m.CondorDAG()

    eospost_job = m.CondorDAGJob(universe="vanilla", executable="util_EOSPosterior.py")
    eospost_job.set_sub_file(os.path.join(outdir, "EOSPost.sub"))
    eospost_job.set_log_file("logs/eospost.log")
    eospost_job.set_stdout_file("logs/eospost.out")
    eospost_job.set_stderr_file("logs/eospost.err")
    eospost_job.add_opt("input-net", "joint.net")
    eospost_job.add_opt("output-file", "eos-posterior.dat")
    eospost_job.add_condor_cmd("request_memory", "16384M")
    eospost_job.add_condor_cmd("request_disk", "1024M")
    eospost_job.add_condor_cmd("getenv", "True")
    eospost_job.add_condor_cmd("accounting_group", "ligo.dev.o4.cbc.pe.lalinference")
    eospost_job.add_condor_cmd("+SlurmPartition", '"compute"')
    eospost_job.write_sub_file()
    eospost_node = m.CondorDAGNode(eospost_job)

    # Marg-event jobs (one per event, all converging on eospost)
    marg_event_nodes = []
    for ev in range(1, n_events + 1):
        marg = m.CondorDAGJob(universe="vanilla", executable="util_ConstructIntrinsicPosterior_GenericCoordinates.py")
        marg.set_sub_file(os.path.join(outdir, "MARG-event-{}.sub".format(ev)))
        marg.set_log_file("logs/marg-event-{}.log".format(ev))
        marg.set_stdout_file("logs/marg-event-{}.out".format(ev))
        marg.set_stderr_file("logs/marg-event-{}.err".format(ev))
        marg.add_opt("input-net", "event-{}.net".format(ev))
        marg.add_opt("output-file", "marg-event-{}.dat".format(ev))
        marg.add_var_opt("idx")
        marg.add_condor_cmd("request_memory", "8192M")
        marg.add_condor_cmd("request_disk", "1024M")
        marg.add_condor_cmd("getenv", "True")
        marg.add_condor_cmd("accounting_group", "ligo.dev.o4.cbc.pe.lalinference")
        marg.add_condor_cmd("+SlurmPartition", '"compute"')
        marg._CondorJob__queue = 4
        marg.write_sub_file()
        marg_node = m.CondorDAGNode(marg)
        marg_node.add_macro("idx", str(ev))
        marg_node.set_retry(1)
        marg_event_nodes.append(marg_node)
        dag.add_node(marg_node)

    for n in marg_event_nodes:
        eospost_node.add_parent(n)
    dag.add_node(eospost_node)

    # Convergence test downstream of eos-post
    test_job = m.CondorDAGJob(universe="vanilla", executable="convergence_test_samples")
    test_job.set_sub_file(os.path.join(outdir, "TEST.sub"))
    test_job.set_log_file("logs/test.log")
    test_job.set_stdout_file("logs/test.out")
    test_job.set_stderr_file("logs/test.err")
    test_job.add_arg("--threshold 0.02")
    test_job.add_condor_cmd("request_memory", "1024")
    test_job.add_condor_cmd("getenv", "True")
    test_job.write_sub_file()
    test_node = m.CondorDAGNode(test_job)
    test_node.add_parent(eospost_node)
    dag.add_node(test_node)

    dag.set_dag_file(os.path.join(outdir, "eos_posterior.dag"))
    dag.write_concrete_dag()


def build_pseudo_pipe(m, outdir):
    """Mimic util_RIFT_pseudo_pipe.py.

    pseudo_pipe runs a few setup jobs (PSD, helper) and then submits a
    sub-DAG corresponding to the BasicIteration pipeline.  We model that
    layered structure here.
    """
    os.makedirs(outdir, exist_ok=True)
    dag = m.CondorDAG()

    # Helper job: constructs the initial grid + writes pipeline args
    helper_job = m.CondorDAGJob(universe="local", executable="helper_LDG_Events.py")
    helper_job.set_sub_file(os.path.join(outdir, "helper.sub"))
    helper_job.set_log_file("logs/helper.log")
    helper_job.set_stdout_file("logs/helper.out")
    helper_job.set_stderr_file("logs/helper.err")
    helper_job.add_arg("--event-time 1187008882.4")
    helper_job.add_arg("--use-coinc event-1.net")
    helper_job.add_arg("--manual-ifo-list H1,L1")
    helper_job.add_condor_cmd("request_memory", "2048M")
    helper_job.add_condor_cmd("getenv", "True")
    helper_job.add_condor_cmd("accounting_group", "ligo.dev.o4.cbc.pe.lalinference")
    helper_job.write_sub_file()
    helper_node = m.CondorDAGNode(helper_job)
    dag.add_node(helper_node)

    # PSD job: BayesWave PSD construction
    psd_job = m.CondorDAGJob(universe="vanilla", executable="BayesWave")
    psd_job.set_sub_file(os.path.join(outdir, "PSD.sub"))
    psd_job.set_log_file("logs/psd.log")
    psd_job.set_stdout_file("logs/psd.out")
    psd_job.set_stderr_file("logs/psd.err")
    psd_job.add_arg("--Niter 1000000")
    psd_job.add_arg("--ifo H1")
    psd_job.add_arg("--ifo L1")
    psd_job.add_condor_cmd("request_memory", "8192M")
    psd_job.add_condor_cmd("request_disk", "2048M")
    psd_job.add_condor_cmd("request_cpus", "1")
    psd_job.add_condor_cmd("getenv", "True")
    psd_job.add_condor_cmd("accounting_group", "ligo.dev.o4.cbc.pe.lalinference")
    psd_job.add_condor_cmd("+SlurmPartition", '"compute"')
    psd_job.write_sub_file()
    psd_node = m.CondorDAGNode(psd_job)
    psd_node.add_parent(helper_node)
    psd_node.set_retry(2)
    dag.add_node(psd_node)

    # Inner BasicIteration sub-DAG: emit it as a separate workflow
    inner_dir = os.path.join(outdir, "iteration_subdag")
    build_basic_iteration_pipeline(m, inner_dir, n_iterations=2)

    # Reference the inner DAG via SUBDAG EXTERNAL (HTCondor / glue) or via
    # an inline driver invocation (slurm).  The active backend takes care of
    # producing the right form.
    inner_path = os.path.join(inner_dir, "marginalize_intrinsic_parameters.dag")
    inner_man = m.CondorDAGManJob(inner_path)
    inner_node = inner_man.create_node()
    inner_node.set_retry(1)
    inner_node.add_parent(psd_node)
    dag.add_node(inner_node)

    # Calibration reweighting + posterior plotting (post-pipeline)
    calreweight_job = m.CondorDAGJob(universe="vanilla", executable="util_CalibrationReweight.py")
    calreweight_job.set_sub_file(os.path.join(outdir, "CalReweight.sub"))
    calreweight_job.set_log_file("logs/calreweight.log")
    calreweight_job.set_stdout_file("logs/calreweight.out")
    calreweight_job.set_stderr_file("logs/calreweight.err")
    calreweight_job.add_opt("input-file", "posterior-final.dat")
    calreweight_job.add_opt("output-file", "posterior-final-cal-reweighted.dat")
    calreweight_job.add_condor_cmd("request_memory", "4096M")
    calreweight_job.add_condor_cmd("getenv", "True")
    calreweight_job.add_condor_cmd("+SlurmPartition", '"compute"')
    calreweight_job.write_sub_file()
    calreweight_node = m.CondorDAGNode(calreweight_job)
    calreweight_node.add_parent(inner_node)
    dag.add_node(calreweight_node)

    plot_job = m.CondorDAGJob(universe="vanilla", executable="plot_posterior_corner.py")
    plot_job.set_sub_file(os.path.join(outdir, "PLOT.sub"))
    plot_job.set_log_file("logs/plot.log")
    plot_job.set_stdout_file("logs/plot.out")
    plot_job.set_stderr_file("logs/plot.err")
    plot_job.add_arg("--posterior-file posterior-final-cal-reweighted.dat")
    plot_job.add_condor_cmd("request_memory", "2048M")
    plot_job.add_condor_cmd("getenv", "True")
    plot_job.write_sub_file()
    plot_node = m.CondorDAGNode(plot_job)
    plot_node.add_parent(calreweight_node)
    dag.add_node(plot_node)

    dag.set_dag_file(os.path.join(outdir, "pseudo_pipe.dag"))
    dag.write_concrete_dag()


def main():
    _stub_in_htcondor()
    _stub_in_glue()
    m = _load_module()

    backends = sorted(m._BACKENDS.keys())
    print("Registered backends:", backends)

    pipelines = [
        ("cepp_basic_iteration", lambda outdir: build_basic_iteration_pipeline(m, outdir)),
        ("eos_posterior",        lambda outdir: build_eos_posterior_pipeline(m, outdir)),
        ("pseudo_pipe",          lambda outdir: build_pseudo_pipe(m, outdir)),
    ]

    if os.path.isdir(OUTPUTS):
        # Wipe just our backend directories, leave anything else.  Some
        # filesystems (e.g. mounted FUSE shares) refuse rmtree's recursive
        # unlink; in that case we just overwrite each generated file in
        # place and the existing tree gets superseded.
        for backend in backends:
            d = os.path.join(OUTPUTS, backend)
            if os.path.isdir(d):
                try:
                    shutil.rmtree(d)
                except (OSError, PermissionError) as exc:
                    print("  (could not rmtree {}: {}; will overwrite in place)".format(d, exc))

    for backend in backends:
        m.set_backend(backend)
        for name, build in pipelines:
            outdir = os.path.join(OUTPUTS, backend, name)
            print()
            print("[{}/{}]".format(backend, name))
            try:
                build(outdir)
            except Exception as exc:
                print("  ERROR: {}".format(exc))
                continue
            artefacts = []
            for root, _, files in os.walk(outdir):
                for f in files:
                    if f.endswith((".sub", ".sbatch", ".dag")) or f.endswith("_dag.sh") \
                            or f.endswith(".sh"):
                        artefacts.append(os.path.relpath(os.path.join(root, f), outdir))
            for a in sorted(artefacts):
                print("  {}".format(a))


if __name__ == "__main__":
    main()
