"""Compress a completed RIFT run without new likelihood evaluations."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .config import file_hash, resolve
from .model import fit_posterior


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def read_posterior(path, weight_column=None, group_column=None):
    path = Path(path)
    if path.name.endswith((".xml", ".xml.gz")):
        from igwn_ligolw import utils, lsctables
        document = utils.load_filename(str(path))
        table = lsctables.SimInspiralTable.get_table(document)
        columns = {name: np.array([getattr(row, name) for row in table], float) for name in
            ("mass1", "mass2", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "distance")}
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            columns = {k: data[k].copy() for k in data.files}
    else:
        # RIFT exports both plain and '#'-prefixed column headers.
        with path.open() as stream:
            names = stream.readline().lstrip("# ").split()
        data = np.loadtxt(path, skiprows=1, ndmin=2)
        if data.shape[1] != len(names) or len(set(names)) != len(names):
            raise ValueError("Posterior header/data mismatch")
        columns = dict(zip(names, data.T))
    def column(*names):
        present = [np.asarray(columns[k], float) for k in names if k in columns]
        if not present:
            raise ValueError("Missing posterior column: " + names[0])
        if any(not np.allclose(present[0], p, rtol=1e-8, atol=1e-12) for p in present[1:]):
            raise ValueError("Conflicting posterior aliases: " + str(names))
        return present[0]
    m1, m2 = column("m1", "mass1", "mass_1"), column("m2", "mass2", "mass_2")
    if np.any(m1 < m2) or np.any(m2 <= 0):
        raise ValueError("Expected positive ordered detector masses; spin labels must match")
    total = m1+m2
    eta = m1*m2/total**2
    mc = total*eta**.6
    if "mc" in columns and not np.allclose(columns["mc"], mc, rtol=1e-7):
        raise ValueError("Detector chirp mass conflicts with component masses")
    distance = column("dist", "distance", "luminosity_distance")
    sz = np.column_stack([column("s1z", "a1z", "spin1z", "spin_1z"), column("s2z", "a2z", "spin2z", "spin_2z")])
    full = None
    aliases = (("s1x", "a1x", "spin1x", "spin_1x"), ("s1y", "a1y", "spin1y", "spin_1y"), ("s2x", "a2x", "spin2x", "spin_2x"), ("s2y", "a2y", "spin2y", "spin_2y"))
    if all(any(a in columns for a in names) for names in aliases):
        xy = [column(*names) for names in aliases]
        full = np.column_stack([mc, eta, xy[0], xy[1], sz[:, 0], xy[2], xy[3], sz[:, 1], distance])
    aligned = np.column_stack([mc, eta, sz, distance])
    weights = column(weight_column) if weight_column else np.ones(len(mc))
    # Preserve intrinsic ancestry when export omitted a dedicated group column.
    intrinsic = (full if full is not None else aligned)[:, :-1]
    groups = column(group_column) if group_column else np.unique(intrinsic, axis=0, return_inverse=True)[1]
    for array in (aligned, weights, groups):
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite posterior rows/weights/groups")
    return aligned, full, weights, groups, columns


def run(config, rundir, posterior=None, output=None):
    runpath = Path(rundir).resolve()
    cfg, contract, settings = resolve(config, runpath)
    posterior = Path(posterior or settings["posterior"])
    if not posterior.is_absolute():
        posterior = runpath/posterior
    output = Path(output or settings["output"])
    if not output.is_absolute():
        output = runpath/output
    # Exclusive output directories preserve failed evidence and concurrent work.
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="rift-auto-fit-run-v1", status="running", posterior=str(posterior),
                    posterior_sha256=file_hash(posterior), config_sha256=file_hash(config),
                    producer_hashes={p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
                    contract=contract, settings=settings)
    write_json(output/"manifest.json", manifest)
    try:
        aligned, full, weights, groups, columns = read_posterior(posterior, settings["weight_column"], settings["group_column"])
        # Recompute source masses from detector masses; validate any existing derived columns.
        from .cosmology import CosmologyContext
        ctx = CosmologyContext(contract["cosmology"], contract["bounds"]["distance"][1])
        z = ctx.redshift(aligned[:, -1])
        for name, expected in (("redshift", z), ("mc_source", aligned[:, 0]/(1+z))):
            if name in columns and not np.allclose(columns[name], expected, rtol=2e-4, atol=1e-7):
                raise ValueError("Existing source columns disagree with resolved cosmology: " + name)
        reports, model_hashes = {}, {}
        for target in settings["targets"]:
            if target == "precessing_source_redshift":
                if full is None or not contract["precessing"]:
                    raise ValueError("Full-spin target requires a precessing run with six recorded spin components")
                points = full
            else:
                points = aligned
            model, report = fit_posterior(points, weights, groups, contract, target,
                components=settings["components"], seed=settings["seed"], maxiter=settings["maxiter"], tolerance=settings["tolerance"])
            directory = output/target
            directory.mkdir()
            write_json(directory/"model.json", model.artifact)
            model_hashes[target] = file_hash(directory/"model.json")
            write_json(directory/"validation.json", report)
            reports[target] = report
        manifest.update(status="complete" if all(r["status"] == "pass_native_recovery" for r in reports.values()) else "failed_validation", reports=reports, model_sha256=model_hashes)
        write_json(output/"manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(output/"manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Original high-level production INI")
    parser.add_argument("--rundir", required=True, help="Completed run containing args_ile.txt, args_cip_list.txt and paired posterior")
    parser.add_argument("--posterior", help="Override posterior path; relative to rundir")
    parser.add_argument("--output", help="New output directory; relative to rundir")
    args = parser.parse_args()
    try:
        result = run(args.config, args.rundir, args.posterior, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, str(error)+"\n")
    print(json.dumps(dict(status=result["status"], posterior=result["posterior"])))
    return 0 if result["status"] == "complete" else 1
