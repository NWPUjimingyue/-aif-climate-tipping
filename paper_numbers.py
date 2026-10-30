"""Print the theoretical values and the key simulation results quoted in the Letter and the
Supplemental Material, computed from the model functions and the stored data files in data/.
Runs in a few seconds:  python3 paper_numbers.py
"""
import os
import numpy as np
from scipy.optimize import brentq
from core import Params, Hb, thresholds, lyapunov, lundberg
from depth import DepthParams, thresholds_depth

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda name: np.load(os.path.join(HERE, 'data', name), allow_pickle=True)


def section(title):
    print('\n' + title + '\n' + '-' * len(title))


P = Params()
s = P.sigma_mid
section('Acceptance probabilities and thresholds (Gamma=20, r=0.31, Delta=0.4)')
print('p_c = %.3f   p_max = %.3f   p_far = %.3f' % (s, P.p_max, P.p_far))
th = thresholds(P)
lam_a = brentq(lambda l: lyapunov(l, s)[0], th['lam_loc'] * 1.0001, 1e4)
print('lambda_s = %.4f   lambda_c = %.4f (at x = %.3f)   lambda_a = %.4f' % (th['lam_loc'], th['rho_max'], th['x_rho'], lam_a))
p_star = brentq(lambda p: Hb(p) - np.log(1 / (2 * p - 1)), 0.51, 0.99)
print('p_a = %.4f   (2 p_a - 1 = exp[-h(p_a)])' % p_star)
L_inf = lyapunov(np.inf, s)[0]
print('pragmatic: Lambda_inf = %.4f   nu = %.3f' % (L_inf, lundberg(np.inf, s)))
for lam in (12.0, 20.0, 50.0):
    print('lambda = %4.0f: Lambda = %+.4f   nu = %.3f' % (lam, lyapunov(lam, s)[0], lundberg(lam, s)))
print('window of Eq. (12): %.4f < b < %.4f' % (1 - P.alpha * P.p_max, 1 - P.alpha * s))

section('Learning exponent of the logit host (auxiliary): unweighted fit of log P_learn versus log|ell_0| over 1e-6 to 1e-1')
b = D('fig2b.npz')
l0 = b['ell0s']
for key, lab in (('inf', 'pragmatic'), ('50.0', 'lambda=50'), ('20.0', 'lambda=20'), ('12.0', 'lambda=12')):
    y = b[key]
    ok = (y > 0) & (l0 >= 0.99e-6) & (l0 <= 1.01e-1)      # unweighted least squares over 1e-6 <= |ell_0| <= 1e-1
    nu_fit = np.polyfit(np.log(l0[ok]), np.log(y[ok]), 1)[0]
    print('%-10s nu_theory = %.3f   nu_fit = %.3f   (%d points)' % (lab, float(b['nu' + key]), nu_fit, ok.sum()))

section('Escape time at finite precision, Sec. S7: slope of log tau versus log gamma')
g = D('gamma.npz')
gam = g['gamma']
for key, lab in (('aif20', 'lambda=20'), ('aif50', 'lambda=50'), ('prag', 'pragmatic')):
    for lo, hi in ((1e5, 1e8), (1e2, 1e4)):
        m = (gam >= lo * 0.999) & (gam <= hi * 1.001)
        sl = np.polyfit(np.log(gam[m]), np.log(g['mean_' + key][m]), 1)[0]
        print('%-10s gamma in [%.0e, %.0e]: slope = %.3f   nu/2 = %.3f' % (lab, lo, hi, sl, float(g['nu_' + key]) / 2))
print('survival at b=0.525, N=200 (Table S4):')
for k in ('aif1.4', 'aif20', 'aif50', 'prag'):
    print('  %-7s' % k, ' '.join('%.3f' % v for v in g['surv_' + k]), ' for gamma =', g['gs'])

section('Survival, Fig. 4(a) and Table S5')
c = D('surv_curves.npz')
i = int(np.argmin(np.abs(c['bb'] - 0.505)))
print('b = %.3f, N = %d:' % (c['bb'][i], int(c['N'])), {k: round(float(c[k][i]), 3) for k in ('bo', 'aif1.4', 'aif3', 'aif20', 'prag', 'oracle')})
n = D('surv_N.npz')
for k in ('oracle', 'bo', 'aif1.4', 'aif3', 'aif20', 'prag'):
    print('  %-7s' % k, ' '.join('%.3f' % v for v in n[k]), ' for N =', n['Ns'])
print('  learning probability: AIF lambda=20 %.3f   pragmatic %.3f' % (n['learn_aif20'][0], n['learn_prag'][0]))

section('Noisy fold, Sec. S9')
fd = D('fold_dB.npz')
dB = int(fd['N']) * (fd['b50'] - 0.5)
kB, Nf = float(fd['kB']), int(fd['N'])
for Dv, v in zip(fd['D'], dB):
    mu = 0.02
    for _ in range(200):                                   # Kramers estimate, Eq. (S42) with t_surv = N
        mu = (0.75 * Dv * np.log(Nf * np.sqrt(mu) / np.pi)) ** (2 / 3)
    print('D = %.0e: Delta B = %.2f   Kramers estimate %.2f   excess %+.0f%%' % (Dv, v, mu / kB, 100 * (v * kB / mu - 1)))
f = D('fold.npz'); sh = dict(zip(fd['D'].tolist(), fd['b50'] - 0.5))
lo_w, hi_w = 1 - P.alpha * P.p_max, 1 - P.alpha * s
for k in ('prag', 'aif1.4', 'oracle'):                    # largest deviation of the shifted survival curves (Sec. S8) from D = 0
    dev = 0.0
    for Dv in (1e-3, 3e-3):
        xs = f['bb'] - sh[Dv]; mm = (xs > lo_w) & (xs < hi_w)
        dev = max(dev, np.max(np.abs(f['%s_D%g' % (k, Dv)][mm] - np.interp(xs[mm], f['bb'], f[k + '_D0']))))
    print('%-7s largest deviation of the shifted survival curve from D = 0 inside the window: %.3f' % (k, dev))

section('Calibration with depth, Fig. 4(c) and Table S6')
cd = D('calib_depth.npz')
PD = DepthParams(float(cd['alpha_c']), float(cd['alpha_m']))
td = thresholds_depth(PD)
xg = np.linspace(0, 1, 20001); pbg = 0.5 * (P.pH(xg) + P.pL(xg))
alg = PD.depth(xg) if hasattr(PD.depth(xg), '__len__') else np.array([PD.depth(x) for x in xg])
Psi = (alg - PD.alpha_c) / (PD.alpha_m - PD.alpha_c); okg = Psi > 1e-9
rb = 1 + (s / pbg[okg] - 1) / Psi[okg]; jb = int(np.argmin(rb))
print('alpha_c = %.3f   alpha_m = %.3f   ratio = %.3f   binding ratio (compromise still maximizes abatement) = %.3f at |x-1/2| = %.3f'
      % (PD.alpha_c, PD.alpha_m, PD.alpha_m / PD.alpha_c, rb[jb], abs(xg[okg][jb] - 0.5)))
print('lambda_c with depth = %.3f   compromise optimal under ignorance: %s' % (td['lam_c'], td['prag_ok']))
E0, N = float(cd['E0']), int(cd['N'])
lo, hi = 1 - PD.alpha_m * P.p_max, 1 - PD.alpha_c * s
print('window: %.3f < T < %.3f C   excess emissions = %.0f GtCO2' % (float(cd['T_lo']), float(cd['T_hi']), (hi - lo) * E0 * N))
T = cd['T']
m = (T >= float(cd['T_lo'])) & (T <= float(cd['T_hi']))
print('window-averaged survival:', {k: round(float(cd[k][m].mean()), 3) for k in ('oracle', 'bo', 'bo_lo', 'aif1.4', 'aif3', 'aif20', 'prag')})
print('mean warming reached: active inference %.3f C, pragmatic %.3f C' % (cd['Tend_aif'].mean(), cd['Tend_prag'].mean()))
order = list(cd['order'])
for name, v, av in zip(cd['tip_names'], cd['tip_vals'], cd['avoid']):
    print('  %-5s threshold %.1f (%.1f, %.1f):' % (name, *v), ' '.join('%s=%.3f' % (k, a) for k, a in zip(order, av)))
print('largest warming reached (all refusals): %.3f C' % float(np.interp(N * E0, [170.0, 525.0, 1055.0], [1.5, 1.7, 2.0])))

section('Epistemic loop gain: universal phase diagram, gain law, Figs. 2(b), 3(a), 3(b), Secs. S3, S4, S6')
import json
from scipy.special import lambertw
from universal import K_a, Lam, nu_of, allowance_sale
print('K_a(1/2) = %.4f (2 sqrt 2 = %.4f)   K_a(0.999999) = %.4f (1 + W(1/e) = %.4f)'
      % (K_a(0.5), 2 * np.sqrt(2), K_a(1 - 1e-6), 1 + lambertw(np.exp(-1)).real))
for p in (0.8, 0.9, s, 0.95):
    la = 1 / ((2 * p - 1) - 1 / K_a(p)) if (2 * p - 1) > 1 / K_a(p) else np.inf
    print('p_c = %.5f   1/p_c = %.4f   K_a = %.4f   logit lambda_a = 1/(2p_c-1-1/K_a) = %.3f' % (p, 1 / p, K_a(p), la))
pa = brentq(lambda p: K_a(p) - 1 / (2 * p - 1), 0.6, 0.95)
print('p_a from K_a(p) = 1/(2p-1): %.4f' % pa)
M = allowance_sale(100, 20, 0.3, 0.9)
xc, pcs, Dsl = M.confounding(0.3); lcs = M.ell_star(xc); Kinf, lss = M.theory(xc, pcs, Dsl, lcs)
print('allowance sale: x_c = %.3f p_c = %.3f w_c = %.3f K_inf = %.3f lam_s = %.3f (K_inf/(x_c lam_s) = %.4f)'
      % (xc, pcs, 1 / (1 + np.exp(-lcs)), Kinf, lss, Kinf / (xc * lss)))
ul = json.load(open(os.path.join(HERE, 'data', 'universal_lam.json')))
dl = np.array([r['Lam_sim'] - r['Lam_th'] for r in ul]); dk = np.array([abs(r['K'] / r['K_law'] - 1) for r in ul])
print('%d Lyapunov tests: max |Lam_sim - Lam_th| = %.4f, rms = %.4f, max relative deviation of K from Eq. (10) = %.1e'
      % (len(ul), np.abs(dl).max(), np.sqrt((dl ** 2).mean()), dk.max()))
for r in json.load(open(os.path.join(HERE, 'data', 'universal_nu.json'))):
    print('  %-9s p_c=%.2f lambda=%5g K=%.3f  nu_sim=%.3f +- %.3f  nu_th=%.3f' % (r['model'], r['pc'], r['lam'], r['K'], r['nu_sim'], r['se'], r['nu_th']))
sl = np.load(os.path.join(HERE, 'data', 'sale.npz'))
print('sale, Fig. 4(d): learned %.3f, within 1e-6 of ell_c %.3f, prices after learning %.3f (+) and %.3f (-)'
      % (float(sl['learned']), float(sl['trapped']), float(sl['xp']), float(sl['xm'])))

section('Ergodicity breaking and the ergodic epistemic value, Figs. 2(a), 4(a)-(c), Secs. S3, S5, S8, S10')
fan = np.load(os.path.join(HERE, 'data', 'ergodic_fan.npz'))
tt = np.arange(int(fan['T']) + 1); K20 = 20.0 / (20.0 * (2 * s - 1) - 1)
print('lambda = 20: K = %.4f, ln phi(2) = %.4f, 2 Lambda = %.4f' % (K20, np.log(1 + K20 ** 2 * s * (1 - s)), 2 * Lam(K20, s)))
print('fitted ensemble rate (t <= 40) %.4f, fitted median rate (50 <= t <= 150) %.4f'
      % (np.polyfit(tt[:41], np.log(fan['aif_mean'][:41]), 1)[0], np.polyfit(tt[50:], np.log(fan['aif_q'][50:, 2]), 1)[0]))
print('mean vs I_0 + accumulated epistemic value: max relative deviation %.3f, final %.2e vs %.2e; final median %.1e'
      % (np.max(np.abs(fan['aif_mean'] / fan['aif_cumI'] - 1)), fan['aif_mean'][-1], fan['aif_cumI'][-1], fan['aif_q'][-1, 2]))
eb = np.load(os.path.join(HERE, 'data', 'ergodic_surv_b.npz')); sc = np.load(os.path.join(HERE, 'data', 'surv_curves.npz'))
mw = (sc['bb'] >= 0.5025) & (sc['bb'] <= 0.5475)
print('window mean survival, N = 200: ergodic 20 %.3f, ergodic 1000 %.3f, Bayes-optimal %.3f, AIF 20 %.3f, pragmatic %.3f'
      % (eb['erg20'][mw].mean(), eb['erg1000'][mw].mean(), sc['bo'][mw].mean(), sc['aif20'][mw].mean(), sc['prag'][mw].mean()))
el = np.load(os.path.join(HERE, 'data', 'ergodic_surv_lam.npz'))
for l, a, e in zip(el['lams'], el['aif'], el['erg']):
    print('  b = 0.525, lambda = %8.2f: standard %.3f, ergodic %.3f' % (l, a, e))
tl = np.load(os.path.join(HERE, 'data', 'ergodic_tlearn.npz'))
print('learning time of ergodic buyers from ell = 0 (mean, median):', [(int(l), round(float(m), 1), float(md)) for l, m, md in zip(tl['lams'], tl['tmean'], tl['tmed'])])
ce = np.load(os.path.join(HERE, 'data', 'calib_ergodic.npz')); cd = np.load(os.path.join(HERE, 'data', 'calib_depth.npz'))
mT = (ce['T'] >= float(cd['T_lo'])) & (ce['T'] <= float(cd['T_hi']))
print('calibrated window mean survival: ergodic 20 %.3f, ergodic 1000 %.3f, Bayes-optimal %.3f, AIF 20 %.3f, pragmatic %.3f'
      % (ce['erg20'][mT].mean(), ce['erg1000'][mT].mean(), cd['bo'][mT].mean(), cd['aif20'][mT].mean(), cd['prag'][mT].mean()))
