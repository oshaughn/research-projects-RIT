#!/bin/bash

python .travis/make_fake_composite.py
util_ConstructIntrinsicPosterior_GenericCoordinates.py  --fname fake.composite  --parameter mtot --parameter q --parameter s1z --parameter s2z --sampler-method GMM --use-precessing --no-plots
