#! /usr/bin/env python
#
# OBJECTIVE
#    unit tests for RIFT outcomes, railing tests
# BASED ON
#   https://git.ligo.org/daniel-williams/catalog-tools/-/blob/main/tests/test_railing.py?ref_type=heads
from asimov.testing import AsimovTest
import os, sys,glob
import numpy as np
import ast
import os
import glob
import dill
import h5py as h5
import peconfigurator.auxiliary.run_analyzer as configurator

def open_pesummary_samples(metafile):
    with h5.File(metafile) as metafile_h:
        analyses = list(metafile_h.keys())
        if "history" in analyses:
            analyses.remove("history")
        if "version" in analyses:
            analyses.remove("version")

        key = analyses[0]

        posterior = np.array(metafile_h[key]['posterior_samples'])
        prior = dict(metafile_h[key]['priors']['samples'])
        prior = {name: np.array(data) for name, data in prior.items()}
        #print(prior)
        
    return posterior, prior



class TestRIFTAnalysesForRailing(AsimovTest):
    def test_distance_railing(self):
        """Check for railing in the distance posterior"""

        for event in self.events:
            for production in event.productions:
                if (production.pipeline.name.lower() == "rift") \
                   and (production.review.status not in ["REJECTED", "APPROVED", "DEPRECATED"]):
                    with self.subTest(event=event.name, production=production.name):
                        rundir = production.rundir
                        target_file = rundir + "/extrinsic_posterior_samples.dat"
                        if not(os.path.exists(target_file)):
                            continue
                        dat = np.genfromtxt(target_file, names=True)
                        if len(dat)==0:
                            self.assertFalse(len(dat)==0, msg="** Run failed to generate output ** ")
                        railing_down, railing_up, rec, _ = configurator.railing_check(dat["distance"],
                                                             Nbin=50,
                                                             tolerance=2.0,
                                                             label="luminosity_distance")
                        self.assertFalse(railing_up, msg=f"Railing detected in luminosity distance at the upper bound; recommend this is adjusted to {_[1]}")
                        #self.assertFalse(railing_down)

    def test_chirp_railing(self):
        """Check for railing in the chirp_mass posterior"""

        for event in self.events:
            for production in event.productions:
                if (production.pipeline.name.lower() == "rift") \
                   and (production.review.status not in ["REJECTED", "APPROVED", "DEPRECATED"]):
                    with self.subTest(event=event.name, production=production.name):
                        rundir = production.rundir
                        target_file = rundir + "/extrinsic_posterior_samples.dat"
                        if not(os.path.exists(target_file)):
                            continue
                        dat = np.genfromtxt(target_file, names=True)
                        if len(dat)==0:
                            self.assertFalse(len(dat)==0, msg="** Run failed to generate output ** ")
                        railing_down, railing_up, rec, _ = configurator.railing_check(dat["mc"],
                                                             Nbin=50,
                                                             tolerance=2.0,
                                                             label="chirp_mass")
                        self.assertFalse(railing_up, msg=f"Railing detected in chirp mass at the upper bound; recommend this is adjusted to {_[1]}")
                        self.assertFalse(railing_down, msg=f"Railing detected in chirp mass at the lower bound; recommend this is adjusted to {_[0]}")
