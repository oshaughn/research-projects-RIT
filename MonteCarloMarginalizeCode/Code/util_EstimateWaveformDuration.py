#! /usr/bin/env python

import lalsimutils
import ourparams
opts,  rosDebugMessagesDictionary = ourparams.ParseStandardArguments()
lalsimutils.rosDebugMessagesDictionary            = rosDebugMessagesDictionary


Psig = ourparams.PopulatePrototypeSignal(opts)

timeWaveform = lalsimutils.estimateWaveformDuration(Psig)

print(float(timeWaveform))
