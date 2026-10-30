"""Core model with continuous design optimisation (vectorised Newton on the compromise
branch plus a coarse global grid), used for all quantitative results.

Model (dimensionless):
  contract terms x in [0,1]; preferred terms m_L = 1/2 - Delta/2, m_H = 1/2 + Delta/2
  acceptance probability p_theta(x) = sigma(Gamma (r - |x - m_theta|))
  emissions per round d = 1 - alpha a
  buyer belief w = P(theta = L), log-odds ell (minus the log odds of the Letter)
  one-step expected free energy  G_w(x) = -lam pbar_w(x) - I_w(x),  lam = alpha beta
"""
import numpy as np

def sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))

def Hb(p):
    p = np.clip(p, 1e-300, 1.0)
    q = np.clip(1.0 - p, 1e-300, 1.0)
    return -(p * np.log(p) + q * np.log(q))

class Params:
    def __init__(self, Gamma=20.0, r=0.31, Delta=0.4, alpha=0.5):
        self.Gamma, self.r, self.Delta, self.alpha = Gamma, r, Delta, alpha
        self.mL, self.mH = 0.5 - Delta / 2, 0.5 + Delta / 2

    def pL(self, x):
        return sig(self.Gamma * (self.r - np.abs(x - self.mL)))

    def pH(self, x):
        return sig(self.Gamma * (self.r - np.abs(x - self.mH)))

    @property
    def sigma_mid(self):
        return sig(self.Gamma * (self.r - self.Delta / 2))

    @property
    def p_max(self):
        return sig(self.Gamma * self.r)

    @property
    def p_far(self):
        return sig(self.Gamma * (self.r - self.Delta))

def G(P, w, x, lam, epistemic=True):
    w = np.asarray(w, float)
    pl, ph = P.pL(x), P.pH(x)
    pb = w * pl + (1 - w) * ph
    g = -lam * pb
    if epistemic:
        g = g - (Hb(pb) - w * Hb(pl) - (1 - w) * Hb(ph))
    return g

def thresholds(P):
    s = P.sigma_mid
    lam_loc = 1.0 / (2 * s - 1) if s > 0.5 else np.inf
    # global threshold rho_max = max_x I/(p_mid - pbar)
    x = np.linspace(0, 1, 20001)
    pl, ph = P.pL(x), P.pH(x)
    pb = 0.5 * (pl + ph)
    I = Hb(pb) - 0.5 * Hb(pl) - 0.5 * Hb(ph)
    dd = s - pb
    ok = dd > 1e-10
    rho = np.full_like(x, -np.inf)
    rho[ok] = I[ok] / dd[ok]
    k = np.argmax(rho)
    return dict(sigma=s, lam_loc=lam_loc, rho_max=rho[k], x_rho=x[k])

def lyapunov(lam, s):
    """Lyapunov exponent of the linearised belief map at the ignorance state."""
    K = lam / (lam * (2 * s - 1) - 1) if np.isfinite(lam) else 1.0 / (2 * s - 1)
    up = 1 + K * (1 - s)
    dn = 1 - K * s
    m = s * np.log(np.abs(up)) + (1 - s) * np.log(np.abs(dn))
    v = s * np.log(np.abs(up)) ** 2 + (1 - s) * np.log(np.abs(dn)) ** 2 - m ** 2
    return m, v, K

class Policy:
    """Deterministic (argmin) or softmax AIF policy with continuous design choice."""
    def __init__(self, P, lam, epistemic=True, ngrid=801):
        self.P, self.lam, self.epi = P, lam, epistemic
        self.xg = np.linspace(0, 1, ngrid)
        self.plg, self.phg = P.pL(self.xg), P.pH(self.xg)

    def Gw(self, w, x):
        return G(self.P, w, x, self.lam, self.epi)

    def choose(self, w, rng=None):
        """w: array (n,). Returns chosen designs (n,)."""
        P = self.P
        w = np.asarray(w, float)
        n = w.size
        # coarse global grid
        pb = w[:, None] * self.plg[None, :] + (1 - w[:, None]) * self.phg[None, :]
        Gg = -self.lam * pb
        if self.epi:
            Gg = Gg - (Hb(pb) - w[:, None] * Hb(self.plg)[None, :] - (1 - w[:, None]) * Hb(self.phg)[None, :])
        j = np.argmin(Gg, axis=1)
        xbest = self.xg[j]
        # refine by Newton inside each smooth piece (avoid kinks at mL, mH)
        x = xbest.copy()
        lo = np.where(x < P.mL, 0.0, np.where(x < P.mH, P.mL, P.mH))
        hi = np.where(x < P.mL, P.mL, np.where(x < P.mH, P.mH, 1.0))
        h = 1e-5
        for _ in range(30):
            g0 = self.Gw(w, x)
            gp = self.Gw(w, np.clip(x + h, lo, hi))
            gm = self.Gw(w, np.clip(x - h, lo, hi))
            d1 = (gp - gm) / (2 * h)
            d2 = (gp - 2 * g0 + gm) / h ** 2
            step = np.where(d2 > 1e-12, -d1 / np.where(d2 > 1e-12, d2, 1), 0.0)
            step = np.clip(step, -0.01, 0.01)
            xn = np.clip(x + step, lo + 1e-12, hi - 1e-12)
            better = self.Gw(w, xn) <= g0 + 1e-15
            x = np.where(better, xn, x)
            if np.max(np.abs(step)) < 1e-13:
                break
        # also refine the compromise branch explicitly (needed for tiny |ell|)
        return x

def run(P, policy, theta, ell0, T, nrun, rng, B0=None, record=False):
    pt = P.pL if theta == 'L' else P.pH
    ell = np.full(nrun, float(ell0)) if np.ndim(ell0) == 0 else np.array(ell0, float)
    budget = np.full(nrun, np.inf if B0 is None else float(B0))
    alive = np.ones(nrun, bool)
    tip = np.full(nrun, -1)
    rec = {}
    if record:
        rec = dict(L=np.empty((T + 1, nrun)), X=np.empty((T, nrun)), B=np.empty((T + 1, nrun)))
        rec['L'][0] = ell; rec['B'][0] = budget
    for t in range(T):
        w = sig(ell)
        x = policy.choose(w, rng)
        pl, ph = P.pL(x), P.pH(x)
        a = rng.random(nrun) < pt(x)
        inc = np.where(a, np.log(pl) - np.log(ph), np.log1p(-pl) - np.log1p(-ph))
        ell = np.where(alive, ell + inc, ell)
        budget = np.where(alive, budget - (1 - P.alpha * a), budget)
        newly = alive & (budget < -1e-9)
        tip[newly] = t + 1
        alive &= ~newly
        if record:
            rec['L'][t + 1] = ell; rec['X'][t] = x; rec['B'][t + 1] = budget
    return dict(ell=ell, alive=alive, tip=tip, **rec)

class TablePolicy:
    """Deterministic AIF policy tabulated on a signed log-spaced belief grid.
    Exact linear response is used for |ell| below the smallest grid point."""
    def __init__(self, P, lam, epistemic=True, umin=-4.0, umax=np.log10(40.0), n=4000):
        self.P, self.lam, self.epi = P, lam, epistemic
        base = Policy(P, lam, epistemic)
        mag = 10.0 ** np.linspace(umin, umax, n)
        self.mag = mag
        self.xp = base.choose(sig(mag))      # ell > 0
        self.xm = base.choose(sig(-mag))     # ell < 0
        s = P.sigma_mid
        if epistemic and np.isfinite(lam):
            den = lam * (2 * s - 1) - 1
            self.kappa = lam / (2 * P.Gamma * den) if den > 0 else np.nan
        else:
            self.kappa = 1.0 / (2 * P.Gamma * (2 * s - 1))
        self.lmax = mag[-1]

    def x_of_ell(self, ell):
        ell = np.asarray(ell, float)
        a = np.abs(ell)
        out = np.empty_like(ell)
        tiny = a < self.mag[0]
        # below the smallest grid point: linear response on the compromise branch if that
        # branch is the global optimum there, otherwise the type test at the smallest grid point
        on_branch = (abs(self.xp[0] - 0.5) < 0.05) and np.isfinite(self.kappa)
        if on_branch:
            out[tiny] = 0.5 - self.kappa * ell[tiny]
        else:
            out[tiny] = np.where(ell[tiny] >= 0, self.xp[0], self.xm[0])
        big = ~tiny
        la = np.log10(np.clip(a[big], self.mag[0], self.lmax))
        grid = np.log10(self.mag)
        k = np.clip(np.searchsorted(grid, la) - 1, 0, len(grid) - 2)
        f = (la - grid[k]) / (grid[k + 1] - grid[k])
        pos = ell[big] >= 0
        x0 = np.where(pos, self.xp[k], self.xm[k])
        x1 = np.where(pos, self.xp[k + 1], self.xm[k + 1])
        jump = np.abs(x1 - x0) > 1e-3
        xi = np.where(jump, np.where(f < 0.5, x0, x1), x0 + f * (x1 - x0))
        out[big] = xi
        return out

def run_table(P, pol, theta, ell0, T, nrun, rng, B0=None, record=False, gamma_noise=None):
    pt = P.pL if theta == 'L' else P.pH
    ell = np.full(nrun, float(ell0)) if np.ndim(ell0) == 0 else np.array(ell0, float)
    budget = np.full(nrun, np.inf if B0 is None else float(B0))
    alive = np.ones(nrun, bool)
    tip = np.full(nrun, -1)
    rec = {}
    if record:
        rec = dict(L=np.empty((T + 1, nrun)), X=np.empty((T, nrun)), B=np.empty((T + 1, nrun)))
        rec['L'][0] = ell; rec['B'][0] = budget
    for t in range(T):
        x = pol.x_of_ell(ell)
        pl, ph = P.pL(x), P.pH(x)
        a = rng.random(nrun) < pt(x)
        inc = np.where(a, np.log(pl) - np.log(ph), np.log1p(-pl) - np.log1p(-ph))
        ell = np.where(alive, ell + inc, ell)
        ell = np.clip(ell, -700, 700)
        budget = np.where(alive, budget - (1 - P.alpha * a), budget)
        newly = alive & (budget < -1e-9)
        tip[newly] = t + 1
        alive &= ~newly
        if record:
            rec['L'][t + 1] = ell; rec['X'][t] = x; rec['B'][t + 1] = budget
    return dict(ell=ell, alive=alive, tip=tip, **rec)

def lundberg(lam, s):
    """Exponent nu solving s|u+|^nu + (1-s)|u-|^nu = 1 (nu>0) for the linearised map."""
    from scipy.optimize import brentq
    m, v, K = lyapunov(lam, s)
    up, dn = abs(1 + K * (1 - s)), abs(1 - K * s)
    f = lambda nu: s * up ** nu + (1 - s) * dn ** nu - 1
    if m >= 0:
        return 0.0
    hi = 1.0
    while f(hi) < 0:
        hi *= 2
    return brentq(f, 1e-9, hi)
