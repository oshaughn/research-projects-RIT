#!/bin/bash

python .travis/make_fake_composite.py
# Test default sampler (constant fit)
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots
# Test alternative sampler (constant fit)
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots --sampler-method GMM

# Test standard sampler (GP fit)
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots  --fit-method gp

# Test standard sampler (GP fit)
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z  --use-precessing --no-plots  --fit-method rf
