"""Survival before tipping: agents versus per-round allowance b for N = 200 (Fig. 4a)."""
import numpy as np, time
from scipy.stats import binom
from core import *
from dp import solve_dp
P = Params()
alpha = P.alpha
rng = np.random.default_rng(21)
s0 = 0.1
samp = lambda n: rng.normal(0, s0, n)

def surv_agent(pol, N, B0, nrun=4000):
    v = []
    for th in ['L', 'H']:
        o = run_table(P, pol, th, samp(nrun), N, nrun, rng, B0=B0)
        v.append(o['alive'].mean())
    return 0.5 * (v[0] + v[1])

def surv_oracle(N, B0):
    k = np.arange(N + 1)
    tot = N - alpha * k
    return binom.pmf(k, N, P.p_max)[tot <= B0 + 1e-9].sum()

def surv_bo(N, B0, nl=401, nx=101):
    out = solve_dp(P, N, B0, nl=nl, nx=nx)
    j = out['nB'] - 1
    l0 = samp(20000)
    return float(np.mean(np.interp(l0, out['ell'], out['V0'][j])))

t0 = time.time()
N = 200
bb = np.round(np.arange(0.495, 0.5701, 0.0025), 4)
agents = {'prag': TablePolicy(P, 1e6, False)}
for lam in [0.5, 1.0, 1.4, 3.0, 20.0]:
    agents['aif%g' % lam] = TablePolicy(P, lam)
S = {k: np.zeros(len(bb)) for k in list(agents) + ['bo', 'oracle']}
for i, b in enumerate(bb):
    for k, pol in agents.items():
        S[k][i] = surv_agent(pol, N, b * N)
    S['oracle'][i] = surv_oracle(N, b * N)
    S['bo'][i] = surv_bo(N, b * N)
    print('curve', b, {k: round(v[i], 4) for k, v in S.items()}, '%.0fs' % (time.time() - t0), flush=True)
np.savez('data/surv_curves.npz', bb=bb, N=N, s0=s0, **S)

print('done', time.time() - t0)
