"""Extension with design-dependent abatement depth.

A reciprocated commitment of design x abates the fraction alpha(x) of the emissions of the round,
    alpha(x) = alpha_c + (alpha_m - alpha_c) [phi((x - m_L)/w) + phi((x - m_H)/w)],
    phi(u) = (1 - u^2)^3 for |u| < 1 and 0 otherwise, w = Delta/4.
The depth equals alpha_c within Delta/4 of the compromise and alpha_m at each preferred design.
The profile is constant near the compromise, so the local theory (shading, belief map, Lyapunov exponent)
is unchanged. Expected free energy:
    G_w(x) = -lam rho(x) pbar_w(x) - I_w(x),  rho = alpha(x)/alpha_c,  lam = alpha_c beta.
"""
import numpy as np
from core import Params, sig, Hb, Policy


class DepthParams(Params):
    def __init__(self, alpha_c, alpha_m, Gamma=20.0, r=0.31, Delta=0.4):
        super().__init__(Gamma=Gamma, r=r, Delta=Delta, alpha=alpha_c)
        self.alpha_c, self.alpha_m, self.wd = alpha_c, alpha_m, Delta / 4

    def depth(self, x):
        x = np.asarray(x, float)
        out = np.full_like(x, self.alpha_c)
        for m in (self.mL, self.mH):
            u = (x - m) / self.wd
            out = out + (self.alpha_m - self.alpha_c) * np.where(np.abs(u) < 1, (1 - u ** 2) ** 3, 0.0)
        return out

    def rho(self, x):
        return self.depth(x) / self.alpha_c


def G_depth(P, w, x, lam, epistemic=True):
    pl, ph = P.pL(x), P.pH(x)
    pb = w * pl + (1 - w) * ph
    g = -lam * P.rho(x) * pb
    if epistemic:
        g = g - (Hb(pb) - w * Hb(pl) - (1 - w) * Hb(ph))
    return g


def thresholds_depth(P):
    """Critical precision with depth: max_x I_{1/2}(x) / (p_c - rho(x) pbar_{1/2}(x)) over designs
    with positive forgone abatement; returns inf if a design beats the compromise pragmatically."""
    s = P.sigma_mid
    x = np.linspace(0, 1, 40001)
    pl, ph = P.pL(x), P.pH(x)
    pb = 0.5 * (pl + ph)
    I = Hb(pb) - 0.5 * Hb(pl) - 0.5 * Hb(ph)
    dd = s - P.rho(x) * pb
    far = np.abs(x - 0.5) > 1e-3
    prag_ok = bool(np.all(dd[far] > 0))
    ratio = np.where(dd > 1e-12, I / np.where(dd > 1e-12, dd, 1), -np.inf)
    ratio[~far] = -np.inf
    k = int(np.argmax(ratio))
    return dict(sigma=s, lam_s=1 / (2 * s - 1), lam_c=ratio[k] if prag_ok else np.inf,
                x_c=x[k], prag_ok=prag_ok, max_prag=float(np.max(P.rho(x[far]) * pb[far])))


class PolicyD(Policy):
    """Policy of Eq. (6) in the limit of large gamma with the depth-weighted pragmatic value."""
    def __init__(self, P, lam, epistemic=True, ngrid=1601):
        super().__init__(P, lam, epistemic, ngrid)
        self.rg = P.rho(self.xg)

    def Gw(self, w, x):
        return G_depth(self.P, w, x, self.lam, self.epi)

    def choose(self, w, rng=None):
        P = self.P
        w = np.asarray(w, float)
        pb = w[:, None] * self.plg[None, :] + (1 - w[:, None]) * self.phg[None, :]
        Gg = -self.lam * self.rg[None, :] * pb
        if self.epi:
            Gg = Gg - (Hb(pb) - w[:, None] * Hb(self.plg)[None, :] - (1 - w[:, None]) * Hb(self.phg)[None, :])
        x = self.xg[np.argmin(Gg, axis=1)].copy()
        lo = np.where(x < P.mL, 0.0, np.where(x < P.mH, P.mL, P.mH))
        hi = np.where(x < P.mL, P.mL, np.where(x < P.mH, P.mH, 1.0))
        h = 1e-5
        for _ in range(40):
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
        return x


class TablePolicyD:
    """Tabulated deterministic policy with depth; linear response below the smallest grid point."""
    def __init__(self, P, lam, epistemic=True, umin=-4.0, umax=np.log10(40.0), n=3000):
        self.P = P
        base = PolicyD(P, lam, epistemic)
        mag = 10.0 ** np.linspace(umin, umax, n)
        self.mag, self.lmax = mag, mag[-1]
        self.xp = base.choose(sig(mag))
        self.xm = base.choose(sig(-mag))
        s = P.sigma_mid
        if epistemic and np.isfinite(lam):
            den = lam * (2 * s - 1) - 1
            self.kappa = lam / (2 * P.Gamma * den) if den > 0 else np.nan
        else:
            self.kappa = 1.0 / (2 * P.Gamma * (2 * s - 1))

    def x_of_ell(self, ell):
        ell = np.asarray(ell, float)
        a = np.abs(ell)
        out = np.empty_like(ell)
        tiny = a < self.mag[0]
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
        out[big] = np.where(jump, np.where(f < 0.5, x0, x1), x0 + f * (x1 - x0))
        return out


def cumulative_emissions(P, pol, theta, ell0, N, rng):
    """Cumulative emissions sum_t [1 - alpha(x_t) a_t] over N rounds without a budget."""
    pt = P.pL if theta == 'L' else P.pH
    ell = np.array(ell0, float)
    U = np.zeros_like(ell)
    for t in range(N):
        x = pol.x_of_ell(ell)
        pl, ph = P.pL(x), P.pH(x)
        a = rng.random(ell.size) < pt(x)
        ell = np.clip(ell + np.where(a, np.log(pl) - np.log(ph), np.log1p(-pl) - np.log1p(-ph)), -700, 700)
        U += 1 - P.depth(x) * a
    return U, ell


def solve_dp_depth(P, N, Bmax, q=0.005, lmax=14.0, nl=301, nx=81, rounding='floor'):
    """Bayes-optimal survival V_0(B, ell) for every budget B <= Bmax (horizon N).
    Budget on a grid of spacing q; a reciprocated design x spends 1 - alpha(x), rounded down to the
    grid (rounding='floor', upper bound on survival) or up (rounding='ceil', lower bound)."""
    nB = int(np.floor(Bmax / q + 1e-9)) + 1
    ell = np.linspace(-lmax, lmax, nl)
    dl = ell[1] - ell[0]
    w = sig(ell)
    x = np.unique(np.concatenate([np.linspace(0, 1, nx), [P.mL, P.mH, 0.5]]))
    pl, ph = P.pL(x), P.pH(x)
    up = np.log(pl) - np.log(ph)
    dn = np.log1p(-pl) - np.log1p(-ph)
    rnd = np.floor if rounding == 'floor' else np.ceil
    k_acc = rnd((1 - P.depth(x)) / q - (1e-9 if rounding == 'ceil' else -1e-9)).astype(int)
    k_ref = int(round(1 / q))

    def sw(c):
        pos = np.clip((ell + c + lmax) / dl, 0, nl - 1)
        i0 = np.minimum(np.floor(pos).astype(int), nl - 2)
        return i0, pos - i0

    Wup = [sw(c) for c in up]
    Wdn = [sw(c) for c in dn]

    def rows(Vn, k):
        out = np.zeros_like(Vn)
        if k < nB:
            out[k:] = Vn[:nB - k]
        return out

    V = np.ones((nB, nl))
    for t in range(N - 1, -1, -1):
        Vr = rows(V, k_ref)
        cache = {}
        best = np.full((nB, nl), -1.0)
        for k in range(len(x)):
            if k_acc[k] not in cache:
                cache[k_acc[k]] = rows(V, k_acc[k])
            Va = cache[k_acc[k]]
            i0, f = Wup[k]
            va = Va[:, i0] * (1 - f) + Va[:, i0 + 1] * f
            i0, f = Wdn[k]
            vr = Vr[:, i0] * (1 - f) + Vr[:, i0 + 1] * f
            pb = w * pl[k] + (1 - w) * ph[k]
            np.maximum(best, pb[None, :] * va + (1 - pb)[None, :] * vr, out=best)
        V = best
    return dict(V0=V, ell=ell, B=np.arange(nB) * q)
