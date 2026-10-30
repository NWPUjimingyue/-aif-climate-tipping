"""Universal normal form of active inference at a confounding contract.
Belief log-odds ell = ln[w/(1-w)], w = q(theta = +).
Near a confounding contract x_c (p_+ = p_- = pc) chosen at the confounded belief ell_c,
the deviation u = ell - ell_c obeys u' = [1 + K (a - pc)] u with epistemic loop gain
K = (p_+' - p_-')(x_c) (dx/dell) / (pc (1 - pc)).
"""
import numpy as np
from scipy.optimize import brentq

def sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))

def Hb(p):
    p = np.clip(p, 1e-300, 1.0); q = np.clip(1.0 - p, 1e-300, 1.0)
    return -(p * np.log(p) + q * np.log(q))

# ---------------- universal functions -----------------
def Lam(K, p):
    return p * np.log(np.abs(1 + K * (1 - p))) + (1 - p) * np.log(np.abs(1 - K * p))

def nu_of(K, p):
    if Lam(K, p) >= 0:
        return 0.0
    f = lambda n: p * abs(1 + K * (1 - p)) ** n + (1 - p) * abs(1 - K * p) ** n - 1
    hi = 1.0
    while f(hi) < 0:
        hi *= 2
    return brentq(f, 1e-10, hi)

def K_a(p):
    return brentq(lambda K: Lam(K, p), 1.0 / p + 1e-9, 1e4)

# ---------------- models -----------------
class Model:
    """pH, pL: response probabilities; R(w, x): pragmatic value (default reciprocation prob)."""
    def __init__(self, name, pH, pL, R=None, xlo=0.0, xhi=1.0):
        self.name, self.pH, self.pL = name, pH, pL
        self.R = R if R is not None else (lambda w, x: w * pH(x) + (1 - w) * pL(x))
        self.xlo, self.xhi = xlo, xhi
        self.xg = np.linspace(xlo, xhi, 8001)

    def G(self, w, x, lam):
        ph, pl = self.pH(x), self.pL(x)
        pb = w * ph + (1 - w) * pl
        I = Hb(pb) - w * Hb(ph) - (1 - w) * Hb(pl)
        return -lam * self.R(w, x) - I

    def policy(self, ell, lam):
        w = sig(np.asarray(ell, float))[:, None]
        Gg = self.G(w, self.xg[None, :], lam)
        j = np.argmin(Gg, axis=1)
        dx = self.xg[1] - self.xg[0]
        x = self.xg[j].astype(float)
        lo, hi = np.maximum(x - dx, self.xlo), np.minimum(x + dx, self.xhi)
        w = w[:, 0]; h = 1e-6
        for _ in range(40):
            g0 = self.G(w, x, lam); gp = self.G(w, np.minimum(x + h, hi), lam); gm = self.G(w, np.maximum(x - h, lo), lam)
            d1 = (gp - gm) / (2 * h); d2 = (gp - 2 * g0 + gm) / h ** 2
            st = np.where(d2 > 1e-10, -d1 / np.where(d2 > 1e-10, d2, 1.0), 0.0)
            xn = np.clip(x + st, lo, hi)
            x = np.where(self.G(w, xn, lam) <= g0 + 1e-16, xn, x)
            if np.max(np.abs(st)) < 1e-14:
                break
        return x

    def confounding(self, x0):
        """x* with pH = pL near x0, pc, D = pH' - pL'."""
        xs = brentq(lambda x: self.pH(x) - self.pL(x), x0 - 0.05, x0 + 0.05, xtol=1e-15)
        pc = float(self.pH(xs)); h = 1e-6
        D = float((self.pH(xs + h) - self.pL(xs + h) - self.pH(xs - h) + self.pL(xs - h)) / (2 * h))
        return xs, pc, D

    def ell_star(self, xs):
        """belief at which x* is the pragmatic stationary point (dR/dx = 0)."""
        h = 1e-6
        f = lambda l: (self.R(sig(l), xs + h) - self.R(sig(l), xs - h)) / (2 * h)
        return brentq(f, -6, 6, xtol=1e-14)

    def theory(self, xs, pc, D, ls):
        """K_inf and lam_s from curvatures: K(lam) = K_inf lam / (lam - lam_s)."""
        w = sig(ls); h = 1e-4; e = 1e-5
        Rxx = (self.R(w, xs + h) - 2 * self.R(w, xs) + self.R(w, xs - h)) / h ** 2
        Rx = lambda l: (self.R(sig(l), xs + h) - self.R(sig(l), xs - h)) / (2 * h)
        Rxl = (Rx(ls + e) - Rx(ls - e)) / (2 * e)
        J = w * (1 - w) * D ** 2 / (pc * (1 - pc))
        return D * Rxl / (pc * (1 - pc) * abs(Rxx)), J / abs(Rxx)

# contract terms: logit acceptance sigma(Gamma (r - |x - m|)) with pc fixed by r
def logit_contract(Gamma, Delta, pc):
    r = Delta / 2 + np.log(pc / (1 - pc)) / Gamma
    mL, mH = 0.5 - Delta / 2, 0.5 + Delta / 2
    return Model(f'logit contract G={Gamma:g} D={Delta:g}',
                 lambda x: sig(Gamma * (r - np.abs(x - mH))),
                 lambda x: sig(Gamma * (r - np.abs(x - mL))))

def probit_contract(c, Delta, pc):
    from scipy.special import ndtr, ndtri
    r = Delta / 2 + ndtri(pc) / c
    mL, mH = 0.5 - Delta / 2, 0.5 + Delta / 2
    return Model(f'probit c={c:g} D={Delta:g}',
                 lambda x: ndtr(c * (r - np.abs(x - mH))),
                 lambda x: ndtr(c * (r - np.abs(x - mL))))

def gauss_contract(s, Delta, pc):
    d = Delta / 2
    p0 = pc / np.exp(-d ** 2 / (2 * s ** 2))
    mL, mH = 0.5 - Delta / 2, 0.5 + Delta / 2
    return Model(f'gauss s={s:g} D={Delta:g} p0={p0:.3f}',
                 lambda x: p0 * np.exp(-(x - mH) ** 2 / (2 * s ** 2)),
                 lambda x: p0 * np.exp(-(x - mL) ** 2 / (2 * s ** 2)))

def allowance_sale(aL, aH, xs, pc):
    """seller quotes allowance price x; buyer of cost type theta buys w.p. sigma(a_theta (b_theta - x));
    pragmatic value = expected revenue x pbar_w(x)."""
    bL = xs + np.log(pc / (1 - pc)) / aL
    bH = xs + np.log(pc / (1 - pc)) / aH
    pH = lambda x: sig(aH * (bH - x)); pL = lambda x: sig(aL * (bL - x))
    return Model(f'allowance sale aL={aL:g} aH={aH:g}', pH, pL,
                 R=lambda w, x: x * (w * pH(x) + (1 - w) * pL(x)))
