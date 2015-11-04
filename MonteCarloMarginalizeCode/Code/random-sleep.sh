#!/bin/bash
SLEEPTIME=`echo $RANDOM/600 | bc` 
echo " Sleeping for " ${SLEEPTIME}
sleep $SLEEPTIME
