"""Noisy saddle-node tipping element driven by cumulative emissions.

dy = (mu - y^2) dt + sqrt(2D) dW,  mu = kappa_B * B (remaining budget).
For B > 0: stable state +sqrt(mu), saddle -sqrt(mu).  Tipping: y < -y_esc.
Each negotiation round lasts one time unit, integrated with n_sub Euler-Maruyama steps.
After the N negotiation rounds, emissions stop (net zero) and the element is followed for
N_post further time units at frozen mu.
"""
import numpy as np
from core import sig

def run_fold(P, pol, theta, ell0, N, nrun, rng, B0, kappa_B, D, N_post=None, n_sub=20, y_esc=3.0):
    N_post = N if N_post is None else N_post
    pt = P.pL if theta == 'L' else P.pH
    ell = np.array(ell0, float) if np.ndim(ell0) else np.full(nrun, float(ell0))
    B = np.full(nrun, float(B0))
    y = np.sqrt(kappa_B * B0) * np.ones(nrun)
    alive = np.ones(nrun, bool)
    dt = 1.0 / n_sub
    amp = np.sqrt(2 * D * dt)
    for t in range(N + N_post):
        if t < N:
            x = pol.x_of_ell(ell)
            pl, ph = P.pL(x), P.pH(x)
            a = rng.random(nrun) < pt(x)
            inc = np.where(a, np.log(pl) - np.log(ph), np.log1p(-pl) - np.log1p(-ph))
            ell = np.clip(ell + inc, -700, 700)
            d = 1 - P.alpha * a
        else:
            d = 0.0
        # emissions spread uniformly over the round
        for k in range(n_sub):
            B = B - d * dt
            mu = kappa_B * B
            y = y + (mu - y * y) * dt + amp * rng.standard_normal(nrun)
            esc = alive & (y < -y_esc)
            alive &= ~esc
            y = np.where(alive, y, -y_esc)  # freeze tipped runs
    return alive
