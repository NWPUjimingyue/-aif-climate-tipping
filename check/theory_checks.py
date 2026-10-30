"""Numerical checks of the theoretical statements of the Letter and SM (Theorems 2 and 5, Lemma 1)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw
from universal import Lam, nu_of, K_a, Model, sig, logit_contract, allowance_sale
ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name)

# Theorem 2(i): K < 1/pc -> Lambda < 0 and nu = 1
pg = np.linspace(0.51, 0.995, 60)
bad = 0
for p in pg:
    for K in np.linspace(0.02, 1 / p - 1e-3, 40):
        if not (Lam(K, p) < 0 and abs(nu_of(K, p) - 1) < 1e-6):
            bad += 1
check('Thm 2(i): Lambda<0 and nu=1 for K<1/pc (2400 grid points)', bad == 0)
# Theorem 2(ii): monotone in K above 1/pc, single zero, 0<nu<1 below K_a
bad = 0
for p in pg:
    Ks = np.linspace(1 / p + 1e-4, 12, 4000); L = Lam(Ks, p)
    if np.any(np.diff(L) <= 0): bad += 1
    if np.sum(np.diff(np.sign(L)) != 0) != 1: bad += 1
    ka = K_a(p)
    for K in np.linspace(1 / p + 1e-3, ka - 1e-3, 8):
        n = nu_of(K, p)
        if not (0 < n < 1): bad += 1
check('Thm 2(ii): Lambda increasing, one zero K_a, 0<nu<1 on (1/pc, K_a)', bad == 0)
check('Thm 2(iii): K_a(1/2) = 2 sqrt 2', abs(K_a(0.5 + 1e-12) - 2 * np.sqrt(2)) < 1e-6)
check('Thm 2(iii): K_a(pc -> 1) = 1 + W0(1/e)', abs(K_a(1 - 1e-7) - (1 + lambertw(np.exp(-1)).real)) < 1e-5)
check('nu continuous at K = 1/pc (nu -> 1)', abs(nu_of(1 / 0.9 + 1e-6, 0.9) - 1) < 1e-3)
# Lemma 1 / Theorem 5: gain law, and lambda_s = 1/(2pc-1) for asymmetric logit hosts
rng = np.random.default_rng(3); worst = 0
for trial in range(12):
    Gp, Gm = rng.uniform(6, 60, 2); Delta = rng.uniform(0.2, 0.6)
    mL, mH = 0.5 - Delta / 2, 0.5 + Delta / 2
    rp, rm = Delta / 2 + rng.uniform(0.02, 0.15), Delta / 2 + rng.uniform(0.02, 0.15)
    pH = lambda x, G=Gp, r=rp, m=mH: sig(G * (r - np.abs(x - m)))
    pL = lambda x, G=Gm, r=rm, m=mL: sig(G * (r - np.abs(x - m)))
    M = Model('asym', pH, pL)
    try:
        xc = brentq(lambda x: pH(x) - pL(x), mL + 1e-6, mH - 1e-6)
    except ValueError:
        continue
    pc = float(pH(xc))
    if pc <= 0.5: continue
    _, pc, D = M.confounding(xc); lc = M.ell_star(xc)
    Kinf, lams = M.theory(xc, pc, D, lc)
    worst = max(worst, abs(lams * (2 * pc - 1) - 1), abs(Kinf / lams - 1))
check('Thm 5(i,ii): lambda_s = K_inf = 1/(2pc-1) for random asymmetric logit hosts (max rel. dev. %.1e)' % worst, worst < 1e-3)
# Theorem 5(iii): K_inf = x_c lambda_s for the allowance sale
M = allowance_sale(100, 20, 0.3, 0.9); xc, pc, D = M.confounding(0.3); lc = M.ell_star(xc)
Kinf, lams = M.theory(xc, pc, D, lc)
check('Thm 5(iii): K_inf = x_c lambda_s for the sale', abs(Kinf / (xc * lams) - 1) < 1e-4)
# Lemma 1: multipliers of the exact update at small deviation, logit host and sale
for M, x0, lam in [(logit_contract(20, 0.4, 0.9), 0.5, 20.0), (allowance_sale(100, 20, 0.3, 0.9), 0.3, 8.0)]:
    xc, pc, D = M.confounding(x0); lc = M.ell_star(xc)
    h = 1e-6; x2 = M.policy(np.array([lc - h, lc + h]), lam); g = (x2[1] - x2[0]) / (2 * h)
    K = D * g / (pc * (1 - pc)); u = 1e-5
    x = M.policy(np.array([lc + u]), lam)[0]
    U1 = 1 + (np.log(M.pH(x)) - np.log(M.pL(x))) / u; U0 = 1 + (np.log1p(-M.pH(x)) - np.log1p(-M.pL(x))) / u
    check('Lemma 1 (%s): U(1), U(0) = %.5f, %.5f vs 1+K(1-pc), 1-K pc = %.5f, %.5f' % (M.name.split()[0], U1, U0, 1 + K * (1 - pc), 1 - K * pc),
          abs(U1 - 1 - K * (1 - pc)) < 1e-3 and abs(U0 - 1 + K * pc) < 1e-3)
# logit lambda_a formula and p_a
pa = brentq(lambda p: 1 / (2 * p - 1) - K_a(p), 0.6, 0.95)
pa2 = brentq(lambda p: (2 * p - 1) - np.exp(p * np.log(p) + (1 - p) * np.log(1 - p)), 0.6, 0.95)
check('p_a: K_a(p_a) = 1/(2p_a-1) agrees with 2p_a-1 = exp(-h(p_a)) (%.5f vs %.5f)' % (pa, pa2), abs(pa - pa2) < 1e-6)
p = 0.90025; la = 1 / ((2 * p - 1) - 1 / K_a(p)); K = la / (la * (2 * p - 1) - 1)
check('lambda_a = %.3f gives K(lambda_a) = K_a' % la, abs(K - K_a(p)) < 1e-9 and abs(la - 9.37) < 0.01)

# Theorem 3 (ergodicity breaking): exact identity E[D(q')] = D(q) + I_w(x) relative to any fixed reference,
# ensemble growth E U^2 = 1 + K^2 pc (1-pc), and the conserved moment E|U|^nu = 1
def kl2(w, r):
    return w * np.log(w / r) + (1 - w) * np.log((1 - w) / (1 - r))
rng = np.random.default_rng(5); worst = 0
for trial in range(2000):
    w, r = rng.uniform(0.02, 0.98, 2); ph, pl = rng.uniform(0.02, 0.98, 2)
    pb = w * ph + (1 - w) * pl
    w1, w0 = w * ph / pb, w * (1 - ph) / (1 - pb)
    hb = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))
    I = hb(pb) - w * hb(ph) - (1 - w) * hb(pl)
    worst = max(worst, abs(pb * kl2(w1, r) + (1 - pb) * kl2(w0, r) - kl2(w, r) - I))
check('Thm 3(i): E[D(q\')] = D(q) + I_w(x) for random beliefs, references, and contracts (max dev %.1e)' % worst, worst < 1e-12)
bad = 0
for p in np.linspace(0.05, 0.95, 19):
    for K in np.linspace(0.1, 6, 30):
        EU2 = p * (1 + K * (1 - p)) ** 2 + (1 - p) * (1 - K * p) ** 2
        if abs(EU2 - 1 - K ** 2 * p * (1 - p)) > 1e-12: bad += 1
        if K != 1 / p and Lam(K, p) < 0:
            n = nu_of(K, p)
            if abs(p * abs(1 + K * (1 - p)) ** n + (1 - p) * abs(1 - K * p) ** n - 1) > 1e-9: bad += 1
check('Thm 3(ii,iii): E U^2 = 1 + K^2 pc(1-pc) and E|U|^nu = 1 on a grid', bad == 0)
# epistemic value = K^2 pc (1-pc) D at linear order (logit host, lam = 20)
M = logit_contract(20, 0.4, 0.9); xc, pc, D_ = M.confounding(0.5); lc = M.ell_star(xc); lam = 20.0
h = 1e-6; x2 = M.policy(np.array([lc - h, lc + h]), lam); g = (x2[1] - x2[0]) / (2 * h); K = D_ * g / (pc * (1 - pc))
u = 1e-4; x = M.policy(np.array([lc + u]), lam)[0]; w = sig(lc + u)
pb = w * M.pH(x) + (1 - w) * M.pL(x)
hb = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))
I = hb(pb) - w * hb(M.pH(x)) - (1 - w) * hb(M.pL(x)); Dinfo = kl2(w, 0.5)
check('epistemic value I = K^2 pc(1-pc) D near ignorance (ratio %.5f)' % (I / (K ** 2 * pc * (1 - pc) * Dinfo)), abs(I / (K ** 2 * pc * (1 - pc) * Dinfo) - 1) < 1e-3)
# Theorem 6: ergodic epistemic value = 2 Lambda(z) of the one-step gain; test amplitude and forgone acceptance
from core import Params
from ergodic import ErgodicPolicy, info
P = Params(); s = P.sigma_mid; A = -P.Gamma ** 2 * s * (1 - s) * (1 - 2 * s); gg = 2 * P.Gamma
worst = 0
for z in [0.5, 1.0, 1.3, 2.0, 3.0, 5.0]:
    u = 1e-5; y = z * u / gg; x = 0.5 - y   # ell is the log odds of type L in core.py, terms shade toward m_L
    ell = u; pl, ph = P.pL(x), P.pH(x); w = sig(ell); pb = w * pl + (1 - w) * ph
    e = pb * np.log(info(ell + np.log(pl) - np.log(ph))) + (1 - pb) * np.log(info(ell + np.log1p(-pl) - np.log1p(-ph))) - np.log(info(ell))
    worst = max(worst, abs(e - 2 * Lam(z, s)))
check('Thm 6(i): ergodic epistemic value = 2 Lambda(z) at small deviation (max dev %.1e)' % worst, worst < 1e-3)
pol = ErgodicPolicy(P, 1e4, ngrid=40001); x0 = pol.choose(np.array([0.5]))[0]
y0 = abs(x0 - 0.5); forg = s - 0.5 * (P.pL(x0) + P.pH(x0))
check('Thm 6(ii): lam = 1e4: lam A y^2 = %.4f (-> 2), lam x forgone acceptance = %.4f (-> 1)' % (1e4 * A * y0 ** 2, 1e4 * forg),
      abs(1e4 * A * y0 ** 2 - 2) < 0.02 and abs(1e4 * forg - 1) < 0.02)
from ergodic import ErgodicTablePolicy
dmin = {}
for lam in [3.0, 20.0, 100.0, 1000.0]:
    tp = ErgodicTablePolicy(P, lam)
    dmin[lam] = float(min(np.abs(tp.xp - 0.5).min(), np.abs(tp.xm - 0.5).min()))
check('Thm 6(iv): ergodic terms stay away from the compromise at every belief, min |x-1/2| = %s' % {k: round(v, 4) for k, v in dmin.items()},
      all(v > 1e-3 for v in dmin.values()))
print('ALL PASS' if ok else 'SOME CHECKS FAILED')
