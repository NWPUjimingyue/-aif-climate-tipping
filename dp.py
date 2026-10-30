"""Bayes-optimal (dual-control) policy maximising the probability of staying within the
carbon budget for N rounds.  State: (t, remaining budget on the lattice B0 - alpha*k - ..., ell).
Emissions per round: 1 (refusal) or 1 - alpha (reciprocation)."""
import numpy as np
from core import Params, sig

def solve_dp(P, N, B0, lmax=14.0, nl=561, nx=201, store_policy=False):
    alpha = P.alpha
    # budget lattice in units of q, where 1 = n1*q and 1-alpha = n0*q
    from fractions import Fraction
    fa = Fraction(alpha).limit_denominator(64)
    q = Fraction(1, fa.denominator)
    n_ref = int(1 / q)                 # lattice steps for a refusal
    n_acc = int((1 - fa) / q)          # lattice steps for reciprocation
    nB = int(np.floor(B0 / float(q) + 1e-9)) + 1   # budget index j: B = j*q, j = 0..nB-1
    ell = np.linspace(-lmax, lmax, nl)
    dl = ell[1] - ell[0]
    w = sig(ell)
    x = np.unique(np.concatenate([np.linspace(0, 1, nx), [P.mL, P.mH, 0.5]]))
    pl, ph = P.pL(x), P.pH(x)
    up = np.log(pl) - np.log(ph)
    dn = np.log1p(-pl) - np.log1p(-ph)

    def shift_weights(c):
        pos = (ell + c + lmax) / dl
        pos = np.clip(pos, 0, nl - 1)
        i0 = np.minimum(np.floor(pos).astype(int), nl - 2)
        f = pos - i0
        return i0, f

    Wup = [shift_weights(c) for c in up]
    Wdn = [shift_weights(c) for c in dn]
    V = np.ones((nB, nl))  # terminal: survived (budget index >= 0)
    pol = np.zeros((N, nB, nl), dtype=np.int16) if store_policy else None

    def take_rows(Vn, k):
        # value after spending k lattice steps: row j -> j - k ; negative -> 0 (tipped)
        out = np.zeros_like(Vn)
        if k < nB:
            out[k:] = Vn[:nB - k]
        return out

    for t in range(N - 1, -1, -1):
        Va = take_rows(V, n_acc)
        Vr = take_rows(V, n_ref)
        best = np.full((nB, nl), -1.0)
        arg = np.zeros((nB, nl), dtype=np.int16)
        for k in range(len(x)):
            i0, f = Wup[k]
            va = Va[:, i0] * (1 - f) + Va[:, i0 + 1] * f
            i0, f = Wdn[k]
            vr = Vr[:, i0] * (1 - f) + Vr[:, i0 + 1] * f
            pb = w * pl[k] + (1 - w) * ph[k]
            Q = pb[None, :] * va + (1 - pb)[None, :] * vr
            better = Q > best + 1e-13
            best = np.where(better, Q, best)
            arg = np.where(better, k, arg)
        V = best
        if store_policy:
            pol[t] = arg
    return dict(V0=V, ell=ell, x=x, q=float(q), nB=nB, policy=pol)

def bo_survival(P, N, B0, ell0=0.0, **kw):
    out = solve_dp(P, N, B0, **kw)
    j = out['nB'] - 1
    return float(np.interp(ell0, out['ell'], out['V0'][j]))
