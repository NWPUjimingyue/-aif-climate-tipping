from universal import *

class Table:
    def __init__(self, M, lam, ls, xs, umin=1e-7, umax=12.0, n=2500):
        self.M, self.ls, self.xs = M, ls, xs
        self.mag = np.geomspace(umin, umax, n)
        self.lg = np.log(self.mag)
        self.xp = M.policy(ls + self.mag, lam)
        self.xm = M.policy(ls - self.mag, lam)
        h = 1e-5
        x2 = M.policy(np.array([ls - h, ls + h]), lam)
        self.g = (x2[1] - x2[0]) / (2 * h)
        self.on = abs(M.policy(np.array([ls]), lam)[0] - xs) < 1e-6

    def x_of(self, ell):
        u = ell - self.ls; a = np.abs(u)
        out = np.empty_like(u)
        tiny = a < self.mag[0]
        out[tiny] = self.xs + self.g * u[tiny] if self.on else np.where(u[tiny] >= 0, self.xp[0], self.xm[0])
        b = ~tiny
        la = np.log(np.clip(a[b], self.mag[0], self.mag[-1]))
        k = np.clip(np.searchsorted(self.lg, la) - 1, 0, len(self.lg) - 2)
        f = (la - self.lg[k]) / (self.lg[k + 1] - self.lg[k])
        pos = u[b] >= 0
        x0 = np.where(pos, self.xp[k], self.xm[k]); x1 = np.where(pos, self.xp[k + 1], self.xm[k + 1])
        jump = np.abs(x1 - x0) > 1e-3
        out[b] = np.where(jump, np.where(f < 0.5, x0, x1), x0 + f * (x1 - x0))
        return out

def p_learn(M, tab, u0, nrun, rng, theta='H', T=200000, ltrap=1e-13, llearn=9.0):
    ell = np.full(nrun, tab.ls + u0)
    state = np.zeros(nrun, int)  # 0 running, 1 trapped, 2 learned
    pt = M.pH if theta == 'H' else M.pL
    for t in range(T):
        r = state == 0
        if not r.any():
            break
        e = ell[r]
        x = tab.x_of(e)
        ph, pl = M.pH(x), M.pL(x)
        a = rng.random(e.size) < pt(x)
        inc = np.where(a, np.log(ph) - np.log(pl), np.log1p(-ph) - np.log1p(-pl))
        e = e + inc
        ell[r] = e
        s = np.zeros(e.size, int)
        s[np.abs(e - tab.ls) < ltrap] = 1
        s[np.abs(e) > llearn] = 2
        st = state[r]; st[s > 0] = s[s > 0]; state[r] = st
    return np.mean(state == 2), np.mean(state == 0), t
