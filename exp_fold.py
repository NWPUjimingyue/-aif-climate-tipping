import numpy as np, time
from core import *
from fold import run_fold
P = Params()
rng = np.random.default_rng(41)
N, kB, s0 = 200, 0.01, 0.1
class Fixed:
    def __init__(self, x): self.x = x
    def x_of_ell(self, ell): return np.full(np.shape(ell), self.x)
agents = {'aif1.4': TablePolicy(P, 1.4), 'prag': TablePolicy(P, 1e6, False)}
Ds = [0.0, 1e-3, 3e-3, 1e-2]
bb = np.round(np.arange(0.495, 0.6001, 0.005), 4)
nrun = 2000
out = {'bb': bb, 'Ds': np.array(Ds), 'N': N, 'kB': kB}
t0 = time.time()
for D in Ds:
    for name in ['oracle', 'aif1.4', 'prag']:
        S = np.zeros(len(bb))
        for i, b in enumerate(bb):
            v = []
            for th in ['L', 'H']:
                pol = Fixed(P.mL if th == 'L' else P.mH) if name == 'oracle' else agents[name]
                alive = run_fold(P, pol, th, rng.normal(0, s0, nrun), N, nrun, rng, b * N, kB, D)
                v.append(alive.mean())
            S[i] = np.mean(v)
        out['%s_D%g' % (name, D)] = S
        print(D, name, np.round(S, 3), '%.0fs' % (time.time() - t0), flush=True)
np.savez('data/fold.npz', **out)
