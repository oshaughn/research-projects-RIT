
## Plotting tools

* ``plot_final_clean.sh``: Useful to generate standard intrinsic/extrinsic plots of the final extrinsic samples, overlaid with injection parameters. 

## PP plot tools

* ``pp_plot_bilbystyle``: Usage ``pp_plot_bilbystyle.py --base-dir DIR``. Default is intrinsic-only (mc,q) plus extrinsic PP plot. Draws from extrinsic_posterior_samples.dat files directly. Renders conventions like bilby, to help with updake of results


**Validating prior**: Starting recently, ``pp_RIFT_with_ini`` will generate plots of the mass distribution in m1,m2 and mc,q to make sure we have not shot ourselves in the foot with the common mistake of mismatches between PE and injection priors (e.g., due to boundary effects).  However, prior draws may sometimes be inadvertently extreme in some way, particularly when one has done a lot of them (eg., very close to a prior boundary).  To better understand the prior draws, we have a few scripts to make a "PP plot" of our injections relative to the prior.
