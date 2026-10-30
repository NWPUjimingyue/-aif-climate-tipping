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

if __name__ == '__main__':
    import sys, json, time
    rng = np.random.default_rng(1)
    cases = [('logit20', logit_contract(20, 0.4, 0.9), 0.5, [12, 20, 50, 200]),
             ('logit60', logit_contract(60, 0.2, 0.9), 0.5, [12, 20, 50, 200]),
             ('logit8', logit_contract(8, 0.6, 0.9), 0.5, [12, 20, 50, 200]),
             ('seller', allowance_sale(50, 20, 0.3, 0.9), 0.3, [4, 20]),
             ('seller2', allowance_sale(100, 20, 0.3, 0.9), 0.3, [4, 8, 20]),
             ('logit95', logit_contract(20, 0.4, 0.95), 0.5, [16, 64]),
             ('probit95', probit_contract(20, 0.4, 0.95), 0.5, [128, 512]),
             ('probit90', probit_contract(20, 0.4, 0.90), 0.5, [32])]
    which = sys.argv[1:] or [c[0] for c in cases]
    res = {}
    for name, M, x0, lams in cases:
        if name not in which: continue
        xs, pc, D = M.confounding(x0); ls = M.ell_star(xs)
        Kinf, lams_ = M.theory(xs, pc, D, ls)
        for lam in lams:
            t0 = time.time()
            tab = Table(M, lam, ls, xs)
            K = D * tab.g / (pc * (1 - pc))
            u0s = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
            P = []
            for u0 in u0s:
                pl, prun, tt = p_learn(M, tab, u0, 20000, rng)
                P.append(pl)
            P = np.array(P)
            ok = P > 0
            sl = np.polyfit(np.log(np.array(u0s)[ok][1:]), np.log(P[ok][1:]), 1)[0]
            res[f'{name}_{lam}'] = dict(K=K, pc=pc, nu_th=nu_of(K, pc), nu_sim=sl, P=P.tolist(), on=bool(tab.on))
            print(f'{name:8s} lam={lam:6.1f} on={tab.on} K={K:.4f} nu_th={nu_of(K,pc):.4f} nu_sim={sl:.4f} P={np.round(P,4)} ({time.time()-t0:.0f}s)', flush=True)
    json.dump(res, open('nu_%s.json' % '_'.join(which), 'w'), indent=1)
