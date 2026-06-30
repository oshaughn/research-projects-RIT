"""
Applications that exploit the differentiable jax_gp likelihood export.

These are the use cases that justify the GP over the (faster, non-AD) random
forest: gradient-based sampling and AD population inference. See ../DESIGN.md.
"""
