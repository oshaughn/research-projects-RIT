#!/usr/bin/env python3
"""Run one bounded GPU-only Q2 point using untouched native generated inputs.

Development witness only: no scheduler, package overlay, source wrapper, or
post-generation repair. Apply any branch profile before Asimov configuration.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-directory", required=True, type=Path)
p.add_argument("--grid", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--factor", type=int, choices=(2,4,8), default=2)
a = p.parse_args()
assert not os.environ.get("PYTHONPATH"), "Use installed candidate code without PYTHONPATH"
run = a.run_directory.resolve()
grid = a.grid.resolve()
root = a.output.resolve()
assert run.is_dir() and grid.is_file()
root.mkdir(parents=True, exist_ok=False)
os.environ["CUPY_CACHE_DIR"] = str(root / "cupy-cache")
os.environ["CUDA_CACHE_PATH"] = str(root / "cuda-cache")
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()
def generated():
    paths = set(run.rglob("*.dag")) | set(run.rglob("*.sub")) | {run/"args_ile.txt"}
    return {str(path): sha(path) for path in paths}
def value(tokens, flag):
    return tokens[len(tokens)-1-tokens[::-1].index(flag)+1]
r = dict(passed=False, development_only=True, submitted=False,
         generated_files_patched=False, python=sys.version.split()[0],
         host=socket.getfqdn(), packages={k:md.version(k) for k in ("RIFT","asimov","cupy-cuda12x")},
         source_url=json.loads(md.distribution("RIFT").read_text("direct_url.json")))
def save():
    (root/"receipt.json").write_text(json.dumps(r,indent=2)+"\n")
save()
try:
    import RIFT
    assert Path(RIFT.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    driver = Path(sys.prefix)/"bin/integrate_likelihood_extrinsic_batchmode"
    r["installed_driver_sha256"] = sha(driver)
    tokens = shlex.split((run/"args_ile.txt").read_text())
    assert float(value(tokens,"--srate")) == 4096
    assert int(value(tokens,"--q-time-pregrid-factor")) == a.factor
    assert "--srate-internal" not in tokens
    assert all(flag in tokens for flag in ("--time-marginalization","--vectorized","--gpu"))
    files = {grid,run/"args_ile.txt"}
    cache_flag = "--cache" if "--cache" in tokens else "--cache-file"
    cache = Path(value(tokens, cache_flag))
    if not cache.is_absolute():cache=run/cache
    files.add(cache)
    for line in cache.read_text().splitlines():
        if not line.strip():continue
        url=urlparse(line.split()[-1])
        assert url.scheme == "file", url.scheme
        path=Path(unquote(url.path))
        if not path.is_absolute():path=run/path
        assert path.is_file(), path
        files.add(path)
    for n,token in enumerate(tokens):
        if token == "--psd-file":
            path=Path(tokens[n+1].split("=",1)[1])
            files.add(path if path.is_absolute() else run/path)
    r["scientific_input_sha256"]={str(path.resolve()):sha(path) for path in files}
    before=generated()
    import cupy as cp
    r["gpu"]=cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    r["visible_devices"]=os.environ.get("CUDA_VISIBLE_DEVICES")
    r["cuda_runtime"]=cp.cuda.runtime.runtimeGetVersion()
    r["cuda_driver"]=cp.cuda.runtime.driverGetVersion()
    r["cuda_toolkit"]=os.environ.get("CUDA_PATH")
    r["gpu_checksum"]=float(cp.arange(1000,dtype=cp.float64).sum().get())
    assert r["gpu_checksum"] == 499500
    command=[sys.executable,str(driver),*tokens,"--sim-xml",str(grid),
             "--event","0","--n-events-to-analyze","1","--n-max","2000",
             "--n-eff","1","--n-chunk","1000","--force-gpu-only",
             "--sampler-xpy","cupy","--output-file",str(root/"smoke.xml.gz")]
    r["command"]=command
    save()
    started=time.monotonic()
    with (root/"ile.log").open("w") as log:
        proc=subprocess.run(command,cwd=run,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    r["returncode"]=proc.returncode
    r["seconds"]=time.monotonic()-started
    text=(root/"ile.log").read_text()
    active=("status=active factor="+str(a.factor) in text or
            ("Q_TIME_PREGRID" in text and "'factor': "+str(a.factor) in text))
    qdt=1/(4096*a.factor)
    spacing=("q_deltaT="+format(qdt,".12g") in text or "'q_deltaT': "+str(qdt) in text)
    r["native_refinement_active"]=active
    r["q_deltaT_verified"]=spacing
    import numpy as np
    products=list(root.glob("smoke.xml.gz*.dat"))
    finite=False
    if len(products)==1:
        values=np.atleast_2d(np.loadtxt(products[0]))
        finite=values.shape[0]==1 and values.shape[1]>=13 and bool(np.isfinite(values).all())
        r["numeric_shape"]=list(values.shape)
    r["finite_products"]=finite
    r["generated_inputs_unchanged"]=before==generated()
    r["scientific_inputs_unchanged"]=all(sha(Path(path))==digest for path,digest in r["scientific_input_sha256"].items())
    r["passed"]=bool(proc.returncode==0 and active and spacing and finite and
                     r["generated_inputs_unchanged"] and r["scientific_inputs_unchanged"])
    assert r["passed"],text[-6000:]
except Exception as error:
    r["error"]=repr(error)
    raise
finally:
    save()
