"""Closed form for argmax_u A(u)^2/(2 B(u)) when A0 == 0 and B1 == 0.

With A(u) = Re(A1 e^{iu}), B(u) = B0 + Re(B2 e^{2iu}), the non-trivial
stationary condition 2 A' B = A B' reduces (z = e^{iu}) to

    z^2 (B0 A1 - conj(A1) B2) = conj(B0 A1 - conj(A1) B2)

i.e. z^2 = conj(w)/w with w = B0*A1 - conj(A1)*B2, so u* = -arg(w) mod pi and

    e^{i u*} = s * conj(w)/|w|,   s = sign(Re(A1 * conj(w)))   [pick A(u*) > 0]

B has only EVEN u-harmonics, so E(u) = E(u+pi): the two roots of z^2 carry the
same E and are the global maxima; the other two stationary points are the
A = 0 minima, dropped when the common factor A was divided out.  Angle-free
(no arg()); reduces to conj(A1)/|A1| when B2 = 0.
"""
import numpy as np
rng = np.random.default_rng(7)
N = 20000
A1 = rng.normal(size=N) + 1j*rng.normal(size=N)
B0 = np.abs(rng.normal(size=N))*3 + 0.05
r = rng.uniform(0, 0.99999, N)
B2 = B0*r*np.exp(1j*rng.uniform(0, 2*np.pi, N))

w = B0*A1 - np.conj(A1)*B2
ph = np.conj(w)/np.maximum(np.abs(w), 1e-300)
s = np.sign(np.real(A1*np.conj(ph)))          # A(u*) = Re(A1 e^{iu*}) > 0
s = np.where(s == 0, 1.0, s)
ph = ph*s
A_st = np.real(A1*ph); B_st = B0 + np.real(B2*ph*ph)
E_st = A_st**2/(2*B_st)

u = np.linspace(0, 2*np.pi, 400001, endpoint=False)
e1 = np.exp(1j*u); e2 = e1*e1
best = np.full(N, -np.inf)
for k in range(0, N, 250):
    sl = slice(k, k+250)
    A = np.real(A1[sl, None]*e1); B = B0[sl, None] + np.real(B2[sl, None]*e2)
    best[sl] = (np.where(A > 0, A*A/(2*np.maximum(B, 1e-300)), 0.0)).max(-1)
rel = (best - E_st)/np.maximum(np.abs(E_st), 1e-300)
print("closed-form vs 400001-point brute force over %d random (A1,B0,B2):" % N)
print("  A(u*) > 0 at %.4f%% of points; B(u*) > 0 at %.4f%%"
      % (100*(A_st > 0).mean(), 100*(B_st > 0).mean()))
print("  relative shortfall (brute - closed)/closed: median %.3e  p99 %.3e  MAX %.3e"
      % tuple(np.percentile(rel, [50, 99, 100])))
print("  worst r = %.6f" % r[np.argmax(rel)])
# and it must reduce to the old rule when B2 == 0
ph0 = np.conj(B0*A1)/np.abs(B0*A1)
print("  B2=0 limit matches conj(A1)/|A1|: max dev %.3e"
      % np.abs(ph0 - np.conj(A1)/np.abs(A1)).max())
