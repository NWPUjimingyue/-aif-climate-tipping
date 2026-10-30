import numpy as np, time, sys
from core import *
P = Params()
part, nparts = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(100 + part)
N, s0, nrun = 200, 0.1, 2000
lam_grid = np.logspace(np.log10(0.3), np.log10(200), 28)
bb = np.round(np.arange(0.4975, 0.5701, 0.0025), 4)
H = np.full((len(lam_grid), len(bb)), np.nan)
t0 = time.time()
for i, lam in enumerate(lam_grid):
    if i % nparts != part: continue
    pol = TablePolicy(P, lam)
    for j, b in enumerate(bb):
        v = []
        for th in ['L', 'H']:
            o = run_table(P, pol, th, rng.normal(0, s0, nrun), N, nrun, rng, B0=b * N)
            v.append(o['alive'].mean())
        H[i, j] = np.mean(v)
    print(part, i, lam, '%.0fs' % (time.time() - t0), flush=True)
np.savez('data/heat_part%d.npz' % part, lam=lam_grid, bb=bb, H=H)
