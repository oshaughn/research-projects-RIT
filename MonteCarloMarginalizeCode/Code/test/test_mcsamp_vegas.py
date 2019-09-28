import RIFT.integrators.mcsampler as mcsampler
sampler=mcsampler.MCSampler()
sampler.add_parameter('x', lambda x: x+1+0.2*x**2, left_limit=0, right_limit=1,prior_pdf=lambda x:1)
sampler.integrate_vegas(lambda x: 1,param_order=['x'])
