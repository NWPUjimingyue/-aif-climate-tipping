"""Ergodicity breaking and the ergodic epistemic value (Figs. 2a, 4a, 4b; Secs. S3, S5, S8).
fan:      information D_t of standard active-inference buyers (lam = 20) and of ergodic buyers,
          mean, quantiles, and D_0 plus the cumulative epistemic value
surv_b:   survival versus allowance b for N = 200 (ergodic buyers)
surv_lam: survival versus lam at b = 0.525, N = 200 (standard and ergodic buyers)
tlearn:   learning time of ergodic buyers from exact ignorance versus lam"""
import sys, time, numpy as np
from core import *
from ergodic import ErgodicTablePolicy, info
P = Params()
s0 = 0.1
mode = sys.argv[1]
t0 = time.time()

def step(pol, ell, theta, rng):
    """theta: 'L', 'H', or a boolean array (True for type L) with one entry per run."""
    x = pol.x_of_ell(ell); pl, ph = P.pL(x), P.pH(x)
    if isinstance(theta, str):
        pt = pl if theta == 'L' else ph
    else:
        pt = np.where(theta, pl, ph)
    a = rng.random(ell.size) < pt
    w = sig(ell); pb = w * pl + (1 - w) * ph
    I = Hb(pb) - w * Hb(pl) - (1 - w) * Hb(ph)
    ell = np.clip(ell + np.where(a, np.log(pl) - np.log(ph), np.log1p(-pl) - np.log1p(-ph)), -700, 700)
    return ell, I

if mode == 'fan':
    rng = np.random.default_rng(31)
    n, T, u0, lam = 40000, 150, 1e-3, 20.0
    qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    out = dict(T=T, u0=u0, lam=lam, qs=qs)
    # types drawn from the initial belief, so that averages are taken under the predictive
    # distribution of the buyer, for which E[I_{t+1}] = I_t + epistemic value holds exactly
    thetaL = rng.random(n) < sig(u0)
    for key, pol in [('aif', TablePolicy(P, lam)), ('erg', ErgodicTablePolicy(P, lam))]:
        ell = np.full(n, u0); D = info(ell)
        Dm, Dq, cumI = [D.mean()], [np.quantile(D, qs)], [D.mean()]
        for t in range(T):
            ell, I = step(pol, ell, thetaL, rng)
            D = info(ell)
            Dm.append(D.mean()); Dq.append(np.quantile(D, qs)); cumI.append(cumI[-1] + I.mean())
        out[key + '_mean'], out[key + '_q'], out[key + '_cumI'] = np.array(Dm), np.array(Dq), np.array(cumI)
        print(key, 'final mean %.3e median %.3e cum %.3e' % (Dm[-1], Dq[-1][2], cumI[-1]), '%.0fs' % (time.time() - t0), flush=True)
    np.savez('data/ergodic_fan.npz', **out)

def surv(pol, N, B0, rng, nrun=4000):
    v = []
    for th in ['L', 'H']:
        o = run_table(P, pol, th, rng.normal(0, s0, nrun), N, nrun, rng, B0=B0)
        v.append(o['alive'].mean())
    return 0.5 * (v[0] + v[1])

if mode == 'surv_b':
    rng = np.random.default_rng(32)
    bb = np.round(np.arange(0.495, 0.5701, 0.0025), 4)
    out = dict(bb=bb, N=200)
    for lam in [20.0, 1000.0]:
        pol = ErgodicTablePolicy(P, lam)
        out['erg%g' % lam] = np.array([surv(pol, 200, b * 200, rng) for b in bb])
        print(lam, np.round(out['erg%g' % lam], 3), '%.0fs' % (time.time() - t0), flush=True)
    np.savez('data/ergodic_surv_b.npz', **out)

if mode == 'surv_lam':
    rng = np.random.default_rng(33)
    lams = np.geomspace(0.5, 3000, 23)
    out = dict(lams=lams, b=0.525, N=200, aif=np.zeros(len(lams)), erg=np.zeros(len(lams)))
    for i, lam in enumerate(lams):
        out['aif'][i] = surv(TablePolicy(P, lam), 200, 0.525 * 200, rng)
        out['erg'][i] = surv(ErgodicTablePolicy(P, lam), 200, 0.525 * 200, rng)
        print('%.3g aif %.3f erg %.3f %.0fs' % (lam, out['aif'][i], out['erg'][i], time.time() - t0), flush=True)
    out['prag'] = surv(TablePolicy(P, 1e6, False), 200, 0.525 * 200, rng)
    print('prag', out['prag'])
    np.savez('data/ergodic_surv_lam.npz', **out)

if mode == 'tlearn':
    rng = np.random.default_rng(34)
    lams = np.array([3, 10, 30, 100, 300, 1000, 3000, 10000, 30000.0])
    tm, tmed = [], []
    for lam in lams:
        pol = ErgodicTablePolicy(P, lam)
        n = 6000; ell = np.zeros(n); tl = np.full(n, -1)
        for t in range(50000):
            ell, _ = step(pol, ell, 'L', rng)
            new = (tl < 0) & (np.abs(ell) > 5); tl[new] = t + 1
            if (tl > 0).all():
                break
        tm.append(tl.mean()); tmed.append(np.median(tl))
        print(lam, tm[-1], tmed[-1], np.mean(ell > 5), flush=True)
    np.savez('data/ergodic_tlearn.npz', lams=lams, tmean=np.array(tm), tmed=np.array(tmed))
