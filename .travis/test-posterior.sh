#!/bin/bash

python .travis/make_fake_composite.py
# Test default sampler
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots
# Test alternative sampler
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots --sampler-method GMM
