"""Ergodic epistemic value: the expected increment of the logarithm of the information
D(q) = KL(q || uniform) = ln 2 - H(q), i.e. the time-average growth rate of information.
Ergodic expected free energy  G^erg_w(x) = -lam pbar_w(x) - E_{a|x} ln D(q'),
where q' is the posterior after response a (the term ln D(q) does not depend on x)."""
import warnings
import numpy as np
from core import sig, Hb, Policy, TablePolicy, Params
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ergodic")   # log(0) at the confounding contract is intended

def info(ell):
    """D = ln 2 - H(w) for w = sigma(ell), accurate near ell = 0 and for large |ell|."""
    ell = np.asarray(ell, float)
    m = np.tanh(0.5 * ell)
    small = np.abs(m) < 1e-3
    ms = np.where(small, m, 0.0)
    ser = ms ** 2 / 2 + ms ** 4 / 12 + ms ** 6 / 30
    return np.where(small, ser, np.log(2.0) - Hb(sig(ell)))

def G_erg(P, ell, x, lam):
    """ell: log odds of type L (as in core.py); broadcasting over ell and x."""
    w = sig(ell)
    pl, ph = P.pL(x), P.pH(x)
    pb = w * pl + (1 - w) * ph
    l1 = ell + np.log(pl) - np.log(ph)
    l0 = ell + np.log1p(-pl) - np.log1p(-ph)
    with np.errstate(divide='ignore'):
        e = pb * np.log(info(l1)) + (1 - pb) * np.log(info(l0))
    return -lam * pb - e

class ErgodicPolicy(Policy):
    def __init__(self, P, lam, ngrid=4001):
        super().__init__(P, lam, True, ngrid)

    def Gw(self, w, x):
        ell = np.log(w) - np.log1p(-w)
        return G_erg(self.P, ell, x, self.lam)

    def choose(self, w, rng=None):
        P = self.P
        w = np.asarray(w, float)
        ell = np.log(w) - np.log1p(-w)
        Gg = G_erg(P, ell[:, None], self.xg[None, :], self.lam)
        Gg = np.where(np.isfinite(Gg), Gg, np.inf)
        j = np.argmin(Gg, axis=1)
        x = self.xg[j].copy()
        dx = self.xg[1] - self.xg[0]
        lo, hi = np.clip(x - dx, 0, 1), np.clip(x + dx, 0, 1)
        h = 1e-6
        for _ in range(40):
            g0 = self.Gw(w, x); gp = self.Gw(w, np.minimum(x + h, hi)); gm = self.Gw(w, np.maximum(x - h, lo))
            d1 = (gp - gm) / (2 * h); d2 = (gp - 2 * g0 + gm) / h ** 2
            st = np.where(d2 > 1e-10, -d1 / np.where(d2 > 1e-10, d2, 1.0), 0.0)
            xn = np.clip(x + st, lo, hi)
            better = self.Gw(w, xn) <= g0 + 1e-15
            x = np.where(better & np.isfinite(self.Gw(w, xn)), xn, x)
            if np.max(np.abs(st)) < 1e-13:
                break
        return x

class ErgodicTablePolicy(TablePolicy):
    """Tabulated ergodic policy; below the smallest grid point the type test of that point is used."""
    def __init__(self, P, lam, umin=-6.0, umax=np.log10(40.0), n=3000):
        self.P, self.lam, self.epi = P, lam, True
        base = ErgodicPolicy(P, lam)
        mag = 10.0 ** np.linspace(umin, umax, n)
        self.mag = mag
        xp, xm = [], []
        for k in range(0, n, 500):
            xp.append(base.choose(sig(mag[k:k + 500])))
            xm.append(base.choose(sig(-mag[k:k + 500])))
        self.xp, self.xm = np.concatenate(xp), np.concatenate(xm)
        self.kappa = np.nan
        self.lmax = mag[-1]

# ---------------- design-dependent abatement depth (calibration, Sec. S10) ----------------
def G_erg_depth(P, ell, x, lam):
    w = sig(ell)
    pl, ph = P.pL(x), P.pH(x)
    pb = w * pl + (1 - w) * ph
    l1 = ell + np.log(pl) - np.log(ph)
    l0 = ell + np.log1p(-pl) - np.log1p(-ph)
    with np.errstate(divide='ignore'):
        e = pb * np.log(info(l1)) + (1 - pb) * np.log(info(l0))
    return -lam * P.rho(x) * pb - e

class ErgodicPolicyD(ErgodicPolicy):
    def Gw(self, w, x):
        ell = np.log(w) - np.log1p(-w)
        return G_erg_depth(self.P, ell, x, self.lam)

    def choose(self, w, rng=None):
        w = np.asarray(w, float)
        ell = np.log(w) - np.log1p(-w)
        Gg = G_erg_depth(self.P, ell[:, None], self.xg[None, :], self.lam)
        Gg = np.where(np.isfinite(Gg), Gg, np.inf)
        x = self.xg[np.argmin(Gg, axis=1)].copy()
        dx = self.xg[1] - self.xg[0]
        lo, hi = np.clip(x - dx, 0, 1), np.clip(x + dx, 0, 1)
        h = 1e-6
        for _ in range(40):
            g0 = self.Gw(w, x); gp = self.Gw(w, np.minimum(x + h, hi)); gm = self.Gw(w, np.maximum(x - h, lo))
            d1 = (gp - gm) / (2 * h); d2 = (gp - 2 * g0 + gm) / h ** 2
            st = np.where(d2 > 1e-10, -d1 / np.where(d2 > 1e-10, d2, 1.0), 0.0)
            xn = np.clip(x + st, lo, hi)
            gn = self.Gw(w, xn)
            x = np.where((gn <= g0 + 1e-15) & np.isfinite(gn), xn, x)
            if np.max(np.abs(st)) < 1e-13:
                break
        return x

class ErgodicTablePolicyD(TablePolicy):
    def __init__(self, P, lam, umin=-6.0, umax=np.log10(40.0), n=3000):
        self.P, self.lam, self.epi = P, lam, True
        base = ErgodicPolicyD(P, lam)
        mag = 10.0 ** np.linspace(umin, umax, n)
        self.mag = mag
        xp, xm = [], []
        for k in range(0, n, 500):
            xp.append(base.choose(sig(mag[k:k + 500])))
            xm.append(base.choose(sig(-mag[k:k + 500])))
        self.xp, self.xm = np.concatenate(xp), np.concatenate(xm)
        self.kappa = np.nan
        self.lmax = mag[-1]
