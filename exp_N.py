import numpy as np, time
from scipy.stats import binom
from core import *
from dp import solve_dp
P = Params(); rng = np.random.default_rng(51)
s0, bfix = 0.1, 0.525
Ns = np.array([25, 50, 100, 200, 400, 800, 1600])
agents = {'prag': TablePolicy(P, 1e6, False), 'aif1.4': TablePolicy(P, 1.4), 'aif3': TablePolicy(P, 3.0), 'aif20': TablePolicy(P, 20.0)}
SN = {k: np.zeros(len(Ns)) for k in list(agents) + ['bo', 'oracle', 'learn_prag', 'learn_aif20']}
t0 = time.time()
for i, n in enumerate(Ns):
    nrun = 4000 if n <= 400 else 2000
    for k, pol in agents.items():
        v = []; lv = []
        for th in ['L', 'H']:
            o = run_table(P, pol, th, rng.normal(0, s0, nrun), int(n), nrun, rng, B0=bfix * n)
            v.append(o['alive'].mean())
        SN[k][i] = np.mean(v)
    k_ = np.arange(n + 1)
    SN['oracle'][i] = binom.pmf(k_, n, P.p_max)[(n - P.alpha * k_) <= bfix * n + 1e-9].sum()
    if n <= 400:
        out = solve_dp(P, int(n), bfix * n, nl=401, nx=101)
        SN['bo'][i] = float(np.mean(np.interp(rng.normal(0, s0, 20000), out['ell'], out['V0'][out['nB'] - 1])))
    else:
        SN['bo'][i] = np.nan
    print(n, {k: round(float(v[i]), 4) for k, v in SN.items()}, '%.0fs' % (time.time() - t0), flush=True)
# learning probabilities without budget (long horizon) for the trapped agents, same prior
for k in ['prag', 'aif20']:
    v = []
    for th in ['L', 'H']:
        o = run_table(P, agents[k], th, rng.normal(0, s0, 20000), 3000, 20000, rng)
        v.append(np.mean(o['ell'] > 5) if th == 'L' else np.mean(o['ell'] < -5))
    SN['learn_' + k][:] = np.mean(v)
    print('learn', k, np.mean(v), flush=True)
np.savez('data/surv_N.npz', Ns=Ns, bfix=bfix, s0=s0, **SN)
