# Restricted EMRI response contract

Source record, 2026-10-22. This may be superseded. The tested implementation
and executed numerical comparisons win on disagreement.

`emri_response.equal_arm_aet` accepts explicit SSB `hp,hc` arrays. It applies
the full time-domain link projection and first-generation TDI using
`fastlisaresponse==1.1.17` and `lisaanalysistools==1.2.8` on CPU.
It does not invoke BBH spherical-mode alignment or phase factorization.

The orbit is the synthetic equal-arm HDF5 asset from lisaanalysistools commit
`28c13db056bb0abe757ba592ff5b3cd52dc89bbe`, Git blob
`c1f7f0579cfd6d82d2037fae85ed825110c9855c`. The adapter verifies its bytes
before constructing an orbit. Missing data raise before backend loading.
It never uses the backend's automatic orbit download fallback.

The input epoch is seconds relative to that orbit's zero. Only zero is
supported because the pinned response constructs its time arrays internally.
Both ends need generated waveform padding. The adapter trims an explicit
integer number of samples from each end and advances the returned epoch.
It rejects changes to sample count and coverage outside the orbit.

The backend receives `hp+i*hc`. FEW's default complex output is `hp-i*hc`;
callers using the FEW polarization pair should pass it without another sign flip.
`test/test_lisa_emri_polarization_basis.py` checks this sign independently: FLR link
outputs follow e+ = uu - vv, ex = uv + vu with u x v along propagation, and a FEW orbit
whose angular momentum points at the observer rotates counterclockwise in (u, v).
The sky input uses ecliptic colatitude and longitude. The response converts
colatitude to latitude once.

Output channel order is A,E,T, with fractional-frequency TDI normalization:
`A=(Z-X)/sqrt(2)`, `E=(X-2Y+Z)/sqrt(6)`, `T=(X+Y+Z)/sqrt(3)`.
Noise covariance belongs to the likelihood caller. These channels do not imply
independent noise. A synthetic independent-noise experiment must state that
assumption; physical noise requires the corresponding channel covariance.

The source tests check input guards, dispatch, signs, padding and epochs.
The opt-in CPU test also compares the standard response wrapper and the
XYZ-to-AET transformation. Set `RIFT_EMRI_ORBIT_FILE` to the staged orbit path
to run it. A skip is not numerical response evidence. The campaign record is
`RIFT_roboto_paper/analyses/paper5_emri_few_20261022/RESPONSE_IMPLEMENTATION.md`.
