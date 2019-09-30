#! /bin/env python
#


import spokes
import numpy as np
from matplotlib import pyplot as plt


#xvals = np.linspace(-1,-0.5,30)
xvals = np.linspace(-1,0.1,30)
fnHere = lambda x: 200*np.cos(x)
yvals =fnHere(xvals)   # Note lnL scale is an option, need to be careful to choose a reasonable default

plt.plot(xvals,yvals,'o')

code, xvals_new = spokes.Refine(xvals,yvals,npts=30)
print(code,len(xvals_new))
if code == 'refined' or code=='extended':
    yvals_new = fnHere(xvals_new)
    plt.plot(xvals_new,yvals_new,'r')

plt.show()
