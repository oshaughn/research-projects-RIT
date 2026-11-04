"""lookup_key for the GW PE synthetic-targeted backend.

Bucket key: a coarse rounding of the most discriminative source params.
Coarser than `same_q`'s tolerance (so true duplicates always share a
bucket), finer than typical population spacing so unrelated events
don't collide.
"""

from __future__ import annotations


def lookup_key(params):
    return (
        round(float(params.get("mc",            0.0)), 3),
        round(float(params.get("eta",           0.0)), 4),
        round(float(params.get("distance",      0.0)), 0),
        round(float(params.get("ra",            0.0)), 3),
        round(float(params.get("dec",           0.0)), 3),
        round(float(params.get("geocent_time",  0.0)), 1),
        params.get("approximant") or "IMRPhenomXPHM",
    )
