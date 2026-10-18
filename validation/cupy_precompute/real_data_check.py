"""Compare precompute backends at the real ILE frame/PSD-loading boundary.

Usage: real_data_check.py OUTPUT.json ILE_EXECUTABLE [ordinary ILE arguments].
Stops before sampling. No strain, PSD values or intrinsic parameters are exported.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

import lal
import numpy as np

# Match ILE's optional gwsignal import policy before loading likelihood code.
if "--use-gwsignal" not in sys.argv:
    os.environ["RIFT_NO_GWSIGNAL"] = "True"
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood.precompute_array import FullRatePrecompute


class Checked(BaseException):
    pass


def metadata(series):
    a = np.asarray(series.data.data)
    return dict(length=series.data.length, deltaF=series.deltaF, f0=series.f0,
                epoch=str(series.epoch),
                sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                zero_bins=int(np.count_nonzero(a == 0)))


def compare(ref, out, p, event, lmax):
    errors = {}
    for det in ref[3]:
        d = {}
        for name, index in [("U", 1), ("V", 2), ("Q", 3)]:
            a, b = ref[index][det], out[index][det]
            assert set(a) == set(b)
            if name == "Q":
                scale = max(np.max(np.abs(v.data.data)) for v in a.values())
                err = 0.
                for k in a:
                    assert a[k].epoch == b[k].epoch
                    assert a[k].deltaT == b[k].deltaT
                    assert a[k].data.length == b[k].data.length
                    err = max(err, np.max(np.abs(a[k].data.data-b[k].data.data))/scale)
                    t = float(a[k].epoch) + np.arange(3, a[k].data.length-3, 19)*a[k].deltaT
                    np.testing.assert_allclose(ref[0][det][k](t), out[0][det][k](t),
                                               rtol=1e-10, atol=1e-10*scale)
            else:
                scale = max(abs(v) for v in ref[1][det].values())
                err = max(abs(a[k]-b[k]) for k in a)/scale
            d[name] = float(err)
        assert max(d.values()) < 1e-10, (det, d)
        errors[det] = d
    lerr = []
    for phi in [.2, 1.5, 4.]:
        e = p.manual_copy()
        e.tref, e.dist = event, 1000e6*lal.PC_SI
        e.phi, e.theta, e.psi, e.incl, e.phiref = phi, .3, .4, .6, .7
        a = fl.FactoredLogLikelihood(e, ref[3], ref[0], ref[1], ref[2], lmax)
        b = fl.FactoredLogLikelihood(e, out[3], out[0], out[1], out[2], lmax)
        assert np.isfinite(a) and np.isfinite(b)
        lerr.append(float(abs(a-b)))
    assert max(lerr) < 1e-7, lerr
    return dict(per_ifo_scaled_errors=errors, max_abs_lnL_error=max(lerr),
                snr_abs_error=float(abs(ref[4]-out[4])))


output, executable = sys.argv[1:3]
original = fl.PrecomputeLikelihoodTerms
result = dict(purpose="real frame/PSD precompute parity; no sampling", points=[])


def check(*args, **kwargs):
    event, _, p, data, psds, lmax = args[:6]
    result["inputs"] = {det: dict(data=metadata(data[det]), psd=metadata(psds[det]))
                        for det in data}
    assert len({v["psd"]["sha256"] for v in result["inputs"].values()}) == len(data)
    contexts = {b: FullRatePrecompute(b, 4) for b in ["numpy", "cupy"]}
    import cupy as cp
    result["cupy"] = cp.__version__
    result["gpu"] = str(cp.cuda.runtime.getDeviceProperties(0)["name"])
    for i in range(2):
        a = list(args)
        a[2] = p.manual_copy()
        a[2].m1 *= 1.+i*1e-4
        row = dict(backends={})
        ref = None
        for backend in ["lal", "numpy", "cupy"]:
            kw = dict(kwargs, skip_interpolation=False,
                      precompute_backend=backend, precompute_batch_size=4,
                      precompute_context=contexts.get(backend))
            cp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            out = original(*a, **kw)
            cp.cuda.get_current_stream().synchronize()
            elapsed = time.perf_counter()-start
            if ref is None:
                ref = out
            row["backends"][backend] = dict(seconds=elapsed,
                **compare(ref, out, a[2], event, lmax))
        result["points"].append(row)
    result["passed"] = True
    Path(output).write_text(json.dumps(result, indent=2)+"\n")
    print("REAL_DATA_PRECOMPUTE_PASS", flush=True)
    raise Checked()


fl.PrecomputeLikelihoodTerms = check
sys.argv = [executable] + sys.argv[3:]
try:
    runpy.run_path(executable, run_name="__main__")
except Checked:
    pass
else:
    raise RuntimeError("ILE did not reach the instrumented precompute boundary")
