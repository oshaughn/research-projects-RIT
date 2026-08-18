"""Run the ILE with the NoLoop likelihood wrapped, to PROVE which path executes."""
import sys, runpy, atexit
import RIFT.likelihood.factored_likelihood as fl

counts = {}
def wrap(mod, name):
    fn = getattr(mod, name, None)
    if fn is None: return
    counts[name] = 0
    def w(*a, **k):
        counts[name] += 1
        if counts[name] == 1:
            print("NOLOOP-PROBE: first call to %s  time_interp=%r xpy=%s"
                  % (name, k.get('time_interp'), getattr(k.get('xpy'), '__name__', '?')),
                  file=sys.stderr, flush=True)
        return fn(*a, **k)
    setattr(mod, name, w)

wrap(fl, 'DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop')
wrap(fl, 'DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopOrig')
wrap(fl, 'FactoredLogLikelihoodTimeMarginalized')   # the SCALAR loop path

@atexit.register
def report():
    print("NOLOOP-PROBE COUNTS: %r" % (counts,), file=sys.stderr, flush=True)

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
