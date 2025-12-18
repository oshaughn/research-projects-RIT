
cd $1
echo $1
shift
 ~/plot_iterationsAndSubdag_animation_with.sh  --parameter mc --parameter q --parameter a1z --parameter a2z --parameter eccentricity --eccentricity --use-legend --ci-list [0.9] --quantiles None  --lnL-cut 15 --use-all-composite-but-grayscale   --bind-param eccentricity --param-bound [0,0.5] $@
