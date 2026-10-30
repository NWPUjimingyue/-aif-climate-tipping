"""Phase diagram of complete learning in the (lam, sigma) plane."""
import numpy as np, time, json
from core import *
from scipy.optimize import brentq
Gamma, Delta, alpha = 20.0, 0.4, 0.5
sig_vals = np.linspace(0.55, 0.98, 22)
lam_vals = np.logspace(np.log10(0.5), np.log10(200), 28)
rng = np.random.default_rng(11)
T, nrun, ell_abs = 1500, 1000, 0.01
Pl = np.zeros((len(sig_vals), len(lam_vals)))
lam_c = np.zeros(len(sig_vals)); lam_loc = np.zeros(len(sig_vals)); lam_s = np.full(len(sig_vals), np.nan)
t0 = time.time()
for i, s in enumerate(sig_vals):
    r = Delta / 2 + np.log(s / (1 - s)) / Gamma
    P = Params(Gamma, r, Delta, alpha)
    th = thresholds(P)
    lam_c[i], lam_loc[i] = th['rho_max'], th['lam_loc']
    try:
        if lyapunov(1e9, s)[0] < 0:
            lam_s[i] = brentq(lambda l: lyapunov(l, s)[0], lam_loc[i] * (1 + 1e-9) + 1e-9, 1e9)
    except ValueError:
        pass
    for j, lam in enumerate(lam_vals):
        pol = TablePolicy(P, lam)
        ell0 = ell_abs * np.where(rng.random(nrun) < 0.5, 1, -1)
        o = run_table(P, pol, 'L', ell0, T, nrun, rng)
        Pl[i, j] = np.mean(o['ell'] > 5)
    print(i, s, lam_c[i], lam_s[i], "%.0fs" % (time.time() - t0), flush=True)
# pragmatic limit column
Pinf = np.zeros(len(sig_vals))
for i, s in enumerate(sig_vals):
    r = Delta / 2 + np.log(s / (1 - s)) / Gamma
    P = Params(Gamma, r, Delta, alpha)
    pol = TablePolicy(P, 1e6, False)
    ell0 = ell_abs * np.where(rng.random(nrun) < 0.5, 1, -1)
    o = run_table(P, pol, 'L', ell0, T, nrun, rng)
    Pinf[i] = np.mean(o['ell'] > 5)
# key names: lam_loc is the spinodal lambda_s and lam_s is the attractor precision lambda_a (make_figures recomputes both)
np.savez('data/phase.npz', sig=sig_vals, lam=lam_vals, Pl=Pl, lam_c=lam_c, lam_loc=lam_loc, lam_s=lam_s, Pinf=Pinf,
         Gamma=Gamma, Delta=Delta, alpha=alpha, T=T, nrun=nrun, ell_abs=ell_abs)
print("done", time.time() - t0)
