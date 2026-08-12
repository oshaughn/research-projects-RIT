#!/usr/bin/env python3
"""Filesystem-level regression tests for the survey_scan tooling."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import collect_results
import emit_condor_jobs
import gpu_inventory
from common import manifest_entries


MANIFEST = """\
schema_version: 1
containers:
  - label: legacy
    image: osdf://example.org/rift-legacy.sif
    cuda_capability_min: 6.0
    cuda_capability_max: 8.9
  - label: modern
    image: /cvmfs/example/rift-modern.sif
    cuda_capability_min: 9.0
    cuda_capability_max: 12.0
"""


class SurveyScanTests(unittest.TestCase):
    def test_manifest_entries_and_inventory_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.yaml"
            manifest.write_text(MANIFEST, encoding="utf-8")

            entries = manifest_entries(manifest)
            self.assertEqual([entry.label for entry in entries], ["legacy", "modern"])
            self.assertEqual(entries[0].cuda_capability_max, 8.9)

            summary = Counter(
                {
                    ("NVIDIA_A100", "8.0", "40960"): 3,
                    ("NVIDIA_H100", "9.0", "81920"): 2,
                    ("unknown", "undefined", "undefined"): 1,
                }
            )
            coverage = gpu_inventory._coverage(summary, manifest)
            by_device = {row["device"]: row for row in coverage}
            self.assertEqual(by_device["NVIDIA_A100"]["manifest_matches"], ["legacy"])
            self.assertEqual(by_device["NVIDIA_H100"]["manifest_matches"], ["modern"])
            self.assertFalse(by_device["unknown"]["covered"])

            bands = gpu_inventory._recommend_bands(summary)
            self.assertEqual([band["label"] for band in bands], ["cc60-90", "cc90-120"])

    def test_emit_jobs_builds_constraints_and_osdf_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.yaml"
            manifest.write_text(MANIFEST, encoding="utf-8")
            survey = root / "survey"
            out = survey / "jobs"

            rc = emit_condor_jobs.main(
                [
                    "--survey",
                    str(survey),
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(out),
                    "--profiles",
                    "cupy",
                ]
            )
            self.assertEqual(rc, 0)

            legacy_submit = (out / "legacy_rift_cupy_common.sub").read_text(encoding="utf-8")
            self.assertIn("(Capability >= 6.0)", legacy_submit)
            self.assertIn("(Capability <= 8.9)", legacy_submit)
            self.assertIn("transfer_output_files = legacy_rift_cupy_common.json", legacy_submit)

            legacy_runner = (out / "run_legacy_rift_cupy_common.sh").read_text(encoding="utf-8")
            self.assertIn("stashcp", legacy_runner)
            self.assertIn("pelican object get", legacy_runner)
            self.assertIn("apptainer exec --nv", legacy_runner)
            self.assertIn("JAX_COMPILATION_CACHE_DIR", legacy_runner)
            self.assertTrue((out / "rift_cupy_common.py").exists())
            self.assertTrue((out / "submit_all.sh").exists())

    def test_collect_results_records_success_and_malformed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            survey = Path(tmp) / "survey"
            jobs = survey / "jobs"
            jobs.mkdir(parents=True)
            (jobs / "good.json").write_text(
                json.dumps(
                    {
                        "profile": "cupy",
                        "ok": True,
                        "elapsed_s": 1.25,
                        "device": {"name": "A100"},
                        "cache": {"bytes": 4096},
                    }
                ),
                encoding="utf-8",
            )
            (jobs / "bad.json").write_text("{not-json", encoding="utf-8")

            out = survey / "summary.json"
            rc = collect_results.main(["--survey", str(survey), "--out", str(out)])
            self.assertEqual(rc, 0)

            summary = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(summary["n_results"], 2)
            self.assertEqual(sum("error" in row for row in summary["results"]), 1)
            markdown = out.with_suffix(".md").read_text(encoding="utf-8")
            self.assertIn("| cupy | PASS | A100 | 1.25 | 4096 |", markdown)
            self.assertIn("| ? | FAIL | ? |", markdown)


if __name__ == "__main__":
    unittest.main()
