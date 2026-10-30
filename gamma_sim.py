"""Finite policy precision: designs sampled from q(x) ~ exp(-gamma G_w(x)).
The policy posterior is sampled in the Laplace approximation around the global minimizer. Competing branches carry relative weights of order exp(-gamma*dG). For lambda > lambda_c
these are negligible; for lambda = 1.4 the compromise branch lies dG = 0.049 above the type test and carries a
posterior mass of about 0.6% at gamma = 100, which the sampling neglects (SM Sec. S7)."""
import numpy as np
from core import *

def G_vec(P, w, x, lam, epi):
    pl, ph = P.pL(x), P.pH(x)
    pb = w * pl + (1 - w) * ph
    g = -lam * pb
    if epi:
        g = g - (Hb(pb) - w * Hb(pl) - (1 - w) * Hb(ph))
    return g

def run_gamma(P, tab, lam, epi, gamma, theta, ell0, T, nrun, rng, B0=None, ell_esc=None):
    pt = P.pL if theta == 'L' else P.pH
    ell = np.full(nrun, float(ell0)) if np.ndim(ell0) == 0 else np.array(ell0, float)
    budget = np.full(nrun, np.inf if B0 is None else float(B0))
    alive = np.ones(nrun, bool)
    t_esc = np.full(nrun, -1)
    h = 1e-5
    for t in range(T):
        w = sig(ell)
        xg = tab.x_of_ell(ell)
        lo = np.where(xg < P.mL, 0.0, np.where(xg < P.mH, P.mL, P.mH))
        hi = np.where(xg < P.mL, P.mL, np.where(xg < P.mH, P.mH, 1.0))
        xa = np.clip(xg, lo + h, hi - h)
        g0 = G_vec(P, w, xa, lam, epi); gp = G_vec(P, w, xa + h, lam, epi); gm = G_vec(P, w, xa - h, lam, epi)
        d2 = np.maximum((gp - 2 * g0 + gm) / h ** 2, 1e-6)
        x = np.clip(xg + rng.standard_normal(nrun) / np.sqrt(gamma * d2), 0.0, 1.0)
        pl, ph = P.pL(x), P.pH(x)
        a = rng.random(nrun) < pt(x)
        inc = np.where(a, np.log(pl) - np.log(ph), np.log1p(-pl) - np.log1p(-ph))
        ell = np.clip(np.where(alive, ell + inc, ell), -700, 700)
        budget = np.where(alive, budget - (1 - P.alpha * a), budget)
        alive &= ~(budget < -1e-9)
        if ell_esc is not None:
            new = (t_esc < 0) & (np.abs(ell) > ell_esc)
            t_esc[new] = t + 1
            if np.all(t_esc >= 0):
                break
    return dict(ell=ell, alive=alive, t_esc=t_esc)
