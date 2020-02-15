my_runs.tgz:
	tar cvzf my_runs.tgz *pp*/net_pp.dat* #*pp*/analysis*/post*.dat *pp*/analysis*/all.net *pp*/mdc.xml.gz


check_zerospin_runs:
	for i in zerospin_pp_*/*clean; do wc -l $$i; cat $$i | awk '{print $$4}' | sort | uniq | wc -l; done
