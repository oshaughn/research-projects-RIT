#! /usr/bin/env python
#
# OBJECTIVE
#    unit tests for RIFT outcomes, to make sure run completes successfully (run quality test)

from asimov.testing import AsimovTest
import os, sys,glob
import numpy as np


class TestRIFTConvergence(AsimovTest):
    def test_final_internal_convergence(self):
        """
        the rundir/iter*test/logs/*.out have convergence information : make sure we achieve our target (<0.02)
        """
        print('RIFT check for internal convergence test NOT IMPLEMENTED')
        for event in self.events:
            for production in event.productions:
                if production.pipeline.name.lower() == "rift":
                    with self.subTest("convergence", production=production.name):
                        rundir = production.rundir
#                        print('RIFT convergence test for ', rundir, "NOT IMPLEMENTED")
#                        repo = event.event_object.repository.directory


class TestRIFTCalmargSampleSize(AsimovTest):
    def test_calmarg_sample_size(self):
        """
        this will count the size of the number of distinct samples in the final file, and require it is more than 1000
        """
        for event in self.events:
            for production in event.productions:
                if production.pipeline.name.lower() == "rift":
                    rundir = production.rundir
                    print(rundir)
                    target_file = rundir + "/reweighted_posterior_samples.dat"
                    if os.path.exists(target_file):
                        with self.subTest(event=event, production=production.name):
                            dat = np.loadtxt(target_file,skiprows=1) # skip header row
                            print(event, production.name, len(dat))  # verbose, most common failure mode, can easily eyeball
                            self.assertFalse(  len(dat) <  1000  )



class TestRIFTFinalCIPneff(AsimovTest):
    def test_final_cip_neff(self):
        """
        this will look at the last CIP directory and assess the n_eff reported in each worker file.  None should be small, if the file is nonempty
        """
        for event in self.events:
            for production in event.productions:
                if production.pipeline.name.lower() == "rift":
                    rundir = production.rundir
                    # completion check
                    target_file = rundir + "/extrinsic_posterior_samples.dat"
                    if not(os.path.exists(target_file)):
                        continue

                    dir_cip_last = glob.glob(rundir +  "/iteration_*_cip")  # usually iteration_6_cip
                    dir_cip_last.sort()
                    if dir_cip_last is None or dir_cip_last is []:
                        print("   RIFT CIP n_eff : -should not happen-, skipping cip test ")
                        continue
                    dir_cip_last = dir_cip_last[-2] # second to last
                    with self.subTest(event=event, production=production.name):
                        cip_out_annotate  = glob.glob(dir_cip_last + "/overlap-grid-*_withpriorchange+annotation.dat")
#                        print(dir_cip_last, cip_out_annotate)
                        if len(cip_out_annotate) < 1:
                            print("   RIFT CIP n_eff stage 2 : -should not happen-, skipping cip test ")
                        n_eff_list = []
                        for name in cip_out_annotate:
                            dat  = np.genfromtxt(name, names=True)
                            n_eff_list.append(dat['neff'])
                        print(event, production.name, np.min(n_eff_list), 1000/len(n_eff_list) )
                        self.assertFalse( any(np.array(n_eff_list) < 1000/len(n_eff_list)) )   # attempt to require the sum of n_eff is at least 1000, distributed evenly over all of them

class TestRIFTMarginalizedLikelihoods(AsimovTest):
    def test_lnLmarg_final(self):
        """
        this will look at the 'all.net' file in the final run, and assess if we have too few high-likleihood points, or other pathologies such as large marginal likleihood error
        Note that to read this file you will need to know if it is eccentric, including matter,  or not, DEFAULT FILE READING
        """

        for event in self.events:
            for production in event.productions:
                if production.pipeline.name.lower() == "rift":
                    rundir = production.rundir
                    # completion check
                    target_file = rundir + "/extrinsic_posterior_samples.dat"
                    if not(os.path.exists(target_file)):
                        continue
                    # test on marginalized likelihood
                    all_net_name = production.pipeline.collect_assets()['lnL_marg']
                    # check file has nonzero size
                    if os.stat(all_net_name).st_size ==0:
                        continue
                    dat = np.loadtxt(all_net_name)  # lnL col is -3, always
                    if len(dat) < 1:
                        continue 
                    lnL_col = -4
                    sigma_col = -3
                    # sort by lnL
                    indx_sort = np.argsort( dat[:,lnL_col])
                    dat = dat[indx_sort]
                    # syntactic sugar
                    lnL_vals = dat[:,lnL_col]
                    sigma_vals = dat[:,sigma_col]
                    n_lines = len(dat)
                    with self.subTest(event=event, production=production.name):
                        # Check file size
                        self.assertFalse(len(lnL_vals)< 5000) # too few lines
                    with self.subTest(event=event, production=production.name):
                        # check top 10% of file has reasonable sigma_vals. Use MEAN
                        n_test = int(n_lines*0.1)
                        self.assertFalse( np.mean(sigma_vals[:n_test]) > 0.4)

                    with self.subTest(event=event, production=production.name):
                        # check that we have enough points within 10 of the peak (typical model dimension constraint).  Argue we need at least 1000
                        self.assertFalse( np.sum( np.max(lnL_vals) - lnL_vals < 10 ) < 1000)

                        
if __name__ == '__main__':
    import unittest
    unittest.main()                    
