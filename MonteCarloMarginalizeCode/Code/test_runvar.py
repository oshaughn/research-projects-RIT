#!/usr/bin/env python
import sys
import numpy
from statutils import cumvar

assert cumvar(numpy.ones(1000))[-1] == 0.0

rvs = numpy.random.normal(0, 1, 1000)
print rvs.std(ddof=1)**2, cumvar(rvs)[-1]
#assert abs(rvs.std()**2 - cumvar(rvs)[-1]) < sys.float_info.epsilon*10

var, mean = None, None
for i in range(10):
    chunk = rvs[i*100:(i+1)*100]
    var = cumvar(chunk, mean, var, i*100)[-1]
    #print mean, numpy.average(rvs[:(i+1)*100])
    mean = numpy.average(rvs[:(i+1)*100])

print rvs.std(ddof=1)**2, var
#assert abs(rvs.std()**2 - var) < sys.float_info.epsilon*10
