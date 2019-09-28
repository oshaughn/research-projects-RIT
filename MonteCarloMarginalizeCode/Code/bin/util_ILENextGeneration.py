#! /usr/bin/env python
#
# GOAL
#   General-purpose iterative refinement tool for ILE
#   Accepts *.composite input format (indx m1 m2 s1x s1y s1z s2x s2y s2z lnL sigmalnL ... )
#   Returns suggestions for where to put the next points to maximize science
#
#   Several methods will be integrated in this one tool
#         - spokes : util_TestSpokesIO.py  (replacement
#         - quadratic fits : calls to BayesianLeastSquares.py
#                       - random resampling
#                       - targeted resampling, based on error (but downweighted by posterior)...start at peak and walk to largest error point
#         - gaussian process fits:
#                        - targeted resampling, based on error (downweighted by posterior)
#                        - random resampling, based on mcsampler posteriors and draws
