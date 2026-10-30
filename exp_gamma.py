"""Finite policy precision: escape time from exact ignorance and survival in the window."""
import numpy as np, time
from core import *
from gamma_sim import run_gamma
P = Params(); s = P.sigma_mid
rng = np.random.default_rng(71)
gam = np.array([1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8])
# the pragmatic buyer has no epistemic value; its lambda = 20 sets the effective policy precision 20*gamma
agents = [('aif20', 20.0, True), ('aif50', 50.0, True), ('prag', 20.0, False)]
out = {'gamma': gam}
t0 = time.time()
for name, lam, epi in agents:
    tab = TablePolicy(P, lam) if epi else TablePolicy(P, 1e6, False)
    out['nu_' + name] = lundberg(lam if epi else np.inf, s)
    med, mean, sem = [], [], []
    for g in gam:
        o = run_gamma(P, tab, lam, epi, g, 'L', 0.0, 400000, 2000, rng, ell_esc=3.0)
        te = o['t_esc'].astype(float)
        assert np.all(te > 0)
        med.append(np.median(te)); mean.append(te.mean()); sem.append(te.std() / np.sqrt(len(te)))
    out['med_' + name] = np.array(med); out['mean_' + name] = np.array(mean); out['sem_' + name] = np.array(sem)
    print(name, np.round(mean, 1), '%.0fs' % (time.time() - t0), flush=True)
# survival in the window, N = 200, b = 0.525, initial log odds N(0, 0.1^2)
N, b = 200, 0.525
gs = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
out['gs'] = gs
for name, lam, epi in agents + [('aif1.4', 1.4, True)]:
    tab = TablePolicy(P, lam) if epi else TablePolicy(P, 1e6, False)
    sv = []
    for g in gs:
        v = [run_gamma(P, tab, lam, epi, g, th, rng.normal(0, 0.1, 4000), N, 4000, rng, B0=b * N)['alive'].mean() for th in ['L', 'H']]
        sv.append(np.mean(v))
    out['surv_' + name] = np.array(sv)
    print('surv', name, np.round(sv, 3), '%.0fs' % (time.time() - t0), flush=True)
np.savez('data/gamma.npz', **out)
