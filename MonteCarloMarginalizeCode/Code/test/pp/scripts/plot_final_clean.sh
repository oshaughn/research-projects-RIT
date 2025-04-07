#! /bin/bash
#  Sample usage:
#      for i in `seq 0 99`; do ../plot_final_clean.sh $i; done
#  Target situation
#      - in run directory setup for pp_RIFT_with_ini
cd analysis_event_$1/rundir
echo $1

if [ ! -e extrinsic_posterior_samples.dat ]; then exit 0; fi

EXTRA=''



if [ ! -e final_ra_dec_distance.png ]; then 
plot_posterior_corner.py --parameter ra --parameter dec --parameter distance --ci-list [0.9] --quantiles None --posterior-file extrinsic_posterior_samples.dat --truth-file ../../mdc.xml.gz --truth-event $1 ${EXTRA}
mv 	corner_ra_dec_distance.png final_ra_dec_distance.png

plot_posterior_corner.py --parameter time --parameter phiorb --parameter psi --ci-list [0.9] --quantiles None --posterior-file extrinsic_posterior_samples.dat --truth-file ../../mdc.xml.gz --truth-event $1  ${EXTRA}
mv corner_time_phiorb_psi.png final_tref_phiref_psi.png
fi

if [ ! -e final_mc_q_xi_chi1perp.png ]; then
plot_posterior_corner.py --parameter mc --parameter q --parameter xi --parameter chi1_perp --composite-file all.net --lnL-cut 15 --use-all-composite-but-grayscale --ci-list [0.9] --quantiles None --posterior-file extrinsic_posterior_samples.dat  --use-legend --posterior-label 'baseline'   --truth-file ../../mdc.xml.gz --truth-event $1 ${EXTRA}
mv      corner_mc_q_xi_chi1_perp.png final_mc_q_xi_chi1perp.png
fi
if [ ! -e final_ra_dec_distance_time.png ]; then
plot_posterior_corner.py --parameter ra --parameter dec --parameter distance --parameter time --ci-list [0.9] --quantiles None --posterior-file extrinsic_posterior_samples.dat --use-legend --posterior-label 'new'   --truth-file ../../mdc.xml.gz --truth-event $1
mv 	corner_ra_dec_distance_time.png final_ra_dec_distance_time.png
fi
if [ ! -e final_tref_phiref_psi.png ]; then

plot_posterior_corner.py --parameter time --parameter phiorb --parameter psi --ci-list [0.9] --quantiles None --posterior-file extrinsic_posterior_samples.dat --use-legend --posterior-label 'new'   --truth-file ../../mdc.xml.gz --truth-event $1
mv corner_time_phiorb_psi.png final_tref_phiref_psi.png
fi
if [ ! -e final_distance_incl.png ]; then
	plot_posterior_corner.py --parameter distance --parameter incl --ci-list [0.9] --quantiles None --posterior-file extrinsic_posterior_samples.dat --use-legend --posterior-label 'new'   --truth-file ../../mdc.xml.gz --truth-event $1 ${EXTRA}
	mv corner_distance_incl.png final_distance_incl.png
fi
