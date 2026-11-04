#!/usr/bin/env python
"""Waveform-free paired export for a known two-component intrinsic/distance GMM.

Proposal rows are drawn from L, weighted by the volumetric-spin/Euclidean-
distance native prior, then resampled into joint posterior fairdraws. Original
proposal ancestry is retained without using the compression implementation.
"""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=591)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    means = np.array([[30., .5, .12, -.1, .2, -.1, .15, -.15, 1000.],
                      [35., .5, 0., -.1, .2, -.1, .15, -.15, 1160.]])
    widths = np.array([1.5, .025, .04, .04, .05, .04, .04, .05, 80.])
    correlation = np.eye(9)
    correlation[0, -1] = correlation[-1, 0] = .45
    correlation[1, 2] = correlation[2, 1] = -.25
    covariance = correlation*widths[:, None]*widths[None, :]
    labels = rng.choice(2, args.rows, p=[.65, .35])
    x = means[labels]+rng.multivariate_normal(np.zeros(9), covariance, args.rows)
    mc, delta, distance = x[:, 0], x[:, 1], x[:, -1]
    eta = (1-delta**2)/4
    total = mc*eta**(-.6)
    m1, m2 = total*(1+delta)/2, total*(1-delta)/2
    # Uniform Cartesian density inside the spin balls; no retained spin factor.
    weight = mc*eta**(-1.2)/2*distance**2
    # Fairdraw native posterior, retaining original proposal ancestry.
    index = rng.choice(args.rows, args.rows, p=weight/weight.sum())
    data = np.column_stack([m1, m2, mc, eta, x[:, 2:8], distance, np.ones(args.rows), np.arange(args.rows)])[index]
    np.savetxt(args.output, data, header="m1 m2 mc eta s1x s1y s1z s2x s2y s2z dist posterior_weight intrinsic_group", comments="")



if __name__ == "__main__":
    main()
