import numpy as np, time
from core import *
P = Params()
s = P.sigma_mid
rng = np.random.default_rng(31)
t0 = time.time()
# (a) belief density over time
T, nrun = 300, 40000
wbins = np.linspace(0, 1, 101)
dens = {}
for name, pol in [('prag', TablePolicy(P, 1e6, False)), ('aif', TablePolicy(P, 1.4))]:
    ell0 = 0.05 * np.where(rng.random(nrun) < 0.5, 1, -1)
    o = run_table(P, pol, 'L', ell0, T, nrun, rng, record=True)
    W = sig(o['L'])           # belief in the true type (code: w = P(L), truth L)
    Hs = np.array([np.histogram(W[t], bins=wbins)[0] / nrun for t in range(T + 1)])
    dens[name] = Hs
    dens[name + '_final_learn'] = np.mean(o['ell'] > 5)
    dens[name + '_final_ign'] = np.mean(np.abs(o['ell']) < 1e-3)
    print(name, dens[name + '_final_learn'], dens[name + '_final_ign'], time.time() - t0, flush=True)
np.savez('data/fig2a.npz', wbins=wbins, T=T, nrun=nrun, prag=dens['prag'], aif=dens['aif'],
         prag_learn=dens['prag_final_learn'], aif_learn=dens['aif_final_learn'])
# (b) learning probability vs initial lean
ell0s = 10.0 ** np.arange(-6, 0.01, 0.5)
res = {}
for lam in [np.inf, 50.0, 20.0, 12.0]:
    pol = TablePolicy(P, 1e6, False) if not np.isfinite(lam) else TablePolicy(P, lam)
    nu = lundberg(lam, s)
    pl = []
    for e0 in ell0s:
        n = 40000
        o = run_table(P, pol, 'L', e0 * np.where(rng.random(n) < 0.5, 1, -1), 1500, n, rng)
        pl.append(np.mean(o['ell'] > 5))
    res[str(lam)] = np.array(pl)
    res['nu' + str(lam)] = nu
    print(lam, nu, np.round(pl, 4), time.time() - t0, flush=True)
np.savez('data/fig2b.npz', ell0s=ell0s, **res)
