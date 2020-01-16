#! /bin/bash
#  USAGE
#    report_convergence analysis_event_
#      - output will be iteration-by-iteration output for convergence information ('test')
for i in $1*
do
  echo $i; cat $i/iter*test/logs/*.out
done
