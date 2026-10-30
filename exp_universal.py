"""Universality tests of the normal form (Figs. 3(a), 3(b) and Sec. S6).
  python exp_universal.py lam         -> data/universal_lam.json  (Lyapunov exponents, linear regime)
  python exp_universal.py nu <case>   -> data/universal_nu_<case>.json (learning exponents, full dynamics)
  python exp_universal.py merge       -> data/universal_nu.json (the six cases concatenated)"""
import sys, json, time
import numpy as np
from universal import *
from universal_sim import Table, p_learn

LAM_CASES = [('logit20_p90', lambda: logit_contract(20, 0.4, 0.9), 0.5),
             ('logit8_p90', lambda: logit_contract(8, 0.6, 0.9), 0.5),
             ('logit20_p95', lambda: logit_contract(20, 0.4, 0.95), 0.5),
             ('logit20_p80', lambda: logit_contract(20, 0.4, 0.80), 0.5),
             ('probit_p95', lambda: probit_contract(20, 0.4, 0.95), 0.5),
             ('probit_p90', lambda: probit_contract(20, 0.4, 0.90), 0.5),
             ('gauss_p90', lambda: gauss_contract(0.5, 0.4, 0.9), 0.5),
             ('seller_p90', lambda: allowance_sale(100, 20, 0.3, 0.9), 0.3)]
NU_CASES = {'logit20': (lambda: logit_contract(20, 0.4, 0.9), 0.5, [12, 20, 50, 200]),
            'logit8': (lambda: logit_contract(8, 0.6, 0.9), 0.5, [16, 64]),
            'logit60': (lambda: logit_contract(60, 0.2, 0.9), 0.5, [32, 128]),
            'logit95': (lambda: logit_contract(20, 0.4, 0.95), 0.5, [10, 16, 64]),
            'probit95': (lambda: probit_contract(20, 0.4, 0.95), 0.5, [128, 512]),
            'seller2': (lambda: allowance_sale(100, 20, 0.3, 0.9), 0.3, [4, 8, 20])}

def lam_collapse(rng):
    out = []
    for name, mk, x0 in LAM_CASES:
        M = mk(); xs, pc, D = M.confounding(x0); ls = M.ell_star(xs)
        Kinf, lams_ = M.theory(xs, pc, D, ls)
        for lam in [2.5, 4, 8, 16, 32, 64, 128]:
            if lam <= lams_ * 1.05 or abs(M.policy(np.array([ls]), lam)[0] - xs) > 1e-6:
                continue
            h = 1e-6; x2 = M.policy(np.array([ls - h, ls + h]), lam); g = (x2[1] - x2[0]) / (2 * h)
            K = D * g / (pc * (1 - pc))
            n, T, u, acc = 4000, 400, np.full(4000, 1e-9), []
            for t in range(T):
                x = xs + g * u; ph, pl = M.pH(x), M.pL(x)
                a = rng.random(n) < ph
                un = u + np.where(a, np.log(ph) - np.log(pl), np.log1p(-ph) - np.log1p(-pl))
                ok = (np.abs(u) < 1e-6) & (np.abs(u) > 1e-14) & (np.abs(un) > 0)
                acc.append(np.log(np.abs(un[ok] / u[ok])))
                un = np.where(np.abs(un) > 1e-6, np.sign(un) * 1e-9, un)
                u = np.where(np.abs(un) < 1e-14, 1e-9, un)
            out.append(dict(model=name, lam=lam, pc=pc, K=K, K_law=Kinf * lam / (lam - lams_),
                            Lam_sim=float(np.mean(np.concatenate(acc))), Lam_th=float(Lam(K, pc))))
            print(out[-1], flush=True)
    json.dump(out, open('data/universal_lam.json', 'w'), indent=1)

def fit_nu(u0s, P, N):
    """Slope of ln P_learn against ln|delta ell_0| by least squares weighted with the number of learned runs,
    and its standard error from binomial noise, Var(ln P_i) = (1 - P_i)/(N P_i), propagated through the
    weighted fit (sandwich form; agrees with a parametric bootstrap)."""
    x = np.log(np.asarray(u0s, float)); P = np.asarray(P, float); n = np.maximum(P * N, 1)
    sl = np.polyfit(x, np.log(np.maximum(P, 1e-9)), 1, w=np.sqrt(n))[0]
    xb = np.sum(n * x) / np.sum(n)
    se = np.sqrt(np.sum(n * (1 - P) * (x - xb) ** 2)) / np.sum(n * (x - xb) ** 2)
    return float(sl), float(se)

def refit():
    """Recompute slopes and standard errors of data/universal_nu.json from the stored learning probabilities."""
    path = 'data/universal_nu.json'; out = json.load(open(path))
    for r in out:
        r['nu_sim'], r['se'] = fit_nu(r['u0'], r['P'], NU_RUNS)
    json.dump(out, open(path, 'w'), indent=1)

NU_RUNS = 60000

def nu_case(name, rng):
    mk, x0, lams = NU_CASES[name]
    M = mk(); xs, pc, D = M.confounding(x0); ls = M.ell_star(xs)
    u0s = np.geomspace(1e-2, 1e-4, 5); out = []
    for lam in lams:
        tab = Table(M, lam, ls, xs); K = D * tab.g / (pc * (1 - pc)); N = NU_RUNS
        P = np.array([p_learn(M, tab, u0, N, rng)[0] for u0 in u0s])
        sl, se = fit_nu(u0s, P, N)
        out.append(dict(model=name, lam=lam, pc=pc, K=K, nu_th=nu_of(K, pc), nu_sim=sl, se=se,
                        u0=u0s.tolist(), P=P.tolist(), on=bool(tab.on)))
        print(out[-1], flush=True)
    json.dump(out, open('data/universal_nu_%s.json' % name, 'w'), indent=1)

def merge():
    out = []
    for name in NU_CASES:
        out += json.load(open('data/universal_nu_%s.json' % name))
    json.dump(out, open('data/universal_nu.json', 'w'), indent=1)

if __name__ == '__main__':
    rng = np.random.default_rng(7)
    if sys.argv[1] == 'lam':
        lam_collapse(rng)
    elif sys.argv[1] == 'merge':
        merge()
    elif sys.argv[1] == 'refit':
        refit()
    else:
        nu_case(sys.argv[2], np.random.default_rng(sum(map(ord, sys.argv[2]))))
