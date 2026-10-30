import numpy as np
from core import *
from fold import run_fold
P = Params(); rng = np.random.default_rng(47)
N, kB, nrun = 200, 0.01, 2000
class Fixed:
    def __init__(self, x): self.x = x
    def x_of_ell(self, ell): return np.full(np.shape(ell), self.x)
def surv(b, D):
    return np.mean([run_fold(P, Fixed(P.mL if th == 'L' else P.mH), th, np.zeros(nrun), N, nrun, rng, b * N, kB, D).mean() for th in ['L', 'H']])
res = {}
for D in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    lo, hi = 0.4975, 0.66
    for _ in range(11):
        mid = 0.5 * (lo + hi)
        if surv(mid, D) >= 0.5: hi = mid
        else: lo = mid
    b50 = 0.5 * (lo + hi)
    res[D] = b50
    print(D, b50, (b50 - 0.5) * N, flush=True)
np.savez('data/fold_dB.npz', D=np.array(list(res)), b50=np.array(list(res.values())), N=N, kB=kB)
