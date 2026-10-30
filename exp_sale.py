"""Posted-price allowance sale: distribution of posted prices over rounds (Fig. 4(d))."""
import numpy as np
from universal import allowance_sale, sig
from universal_sim import Table
M = allowance_sale(100, 20, 0.3, 0.9)
xs, pc, D = M.confounding(0.3); lc = M.ell_star(xs)
lam, T, nrun, du = 8.0, 400, 40000, 0.2
rng = np.random.default_rng(5)
tab = Table(M, lam, lc, xs)
theta = rng.random(nrun) < 0.5                     # True: type + (high cost)
ell = lc + du * np.where(rng.random(nrun) < 0.5, 1, -1)
bins = np.linspace(0.27, 0.34, 141)
H = np.zeros((T, len(bins) - 1))
for t in range(T):
    x = tab.x_of(ell)
    H[t] = np.histogram(x, bins)[0] / nrun
    ph, pl = M.pH(x), M.pL(x)
    a = rng.random(nrun) < np.where(theta, ph, pl)
    ell = ell + np.where(a, np.log(ph) - np.log(pl), np.log1p(-ph) - np.log1p(-pl))
learned = np.where(theta, ell > 9, ell < -9)
trapped = np.abs(ell - lc) < 1e-6
xp = M.policy(np.array([20.0]), lam)[0]; xm = M.policy(np.array([-20.0]), lam)[0]
print('learned %.4f trapped %.4f other %.4f  x_+ %.4f x_- %.4f' % (learned.mean(), trapped.mean(), 1 - learned.mean() - trapped.mean(), xp, xm))
np.savez('data/sale.npz', H=H, bins=bins, T=T, xc=xs, xp=xp, xm=xm, lam=lam, du=du, learned=learned.mean(), trapped=trapped.mean())
