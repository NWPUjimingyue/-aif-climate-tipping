"""Calibration of Sec. S10 for the ergodic buyer (Fig. 4c, Table S6): survival S(T), window mean,
and probabilities of avoiding tipping elements, with the same data and mapping as exp_calib_depth.py."""
import numpy as np, time
from exp_calib_depth import (alpha_c, alpha_m, E0, N, s0, nrun, budget, warming, TIP, threshold_cdf)
from depth import DepthParams, cumulative_emissions
from ergodic import ErgodicTablePolicyD
P = DepthParams(alpha_c, alpha_m)
rng = np.random.default_rng(2027)
Tg = np.round(np.arange(1.55, 1.951, 0.0025), 4)
Bg = budget(Tg) / E0
out = dict(T=Tg)
t0 = time.time()
for lam in [20.0, 1000.0]:
    pol = ErgodicTablePolicyD(P, lam)
    U = np.concatenate([cumulative_emissions(P, pol, th, rng.normal(0, s0, nrun), N, rng)[0] for th in ['L', 'H']])
    S = np.array([np.mean(U <= B + 1e-9) for B in Bg])
    Tend = warming(U * E0)
    lo, hi = 1 - alpha_m * P.p_max, 1 - alpha_c * P.sigma_mid
    Tlo, Thi = float(warming(lo * E0 * N)), float(warming(hi * E0 * N))
    m = (Tg >= Tlo) & (Tg <= Thi)
    out['erg%g' % lam] = S
    out['avoid_erg%g' % lam] = np.array([np.mean(1 - threshold_cdf(Tend, *TIP[k])) for k in TIP])
    print('lam=%g window mean %.3f, mean warming %.3f, %.0fs' % (lam, S[m].mean(), Tend.mean(), time.time() - t0), flush=True)
    print('  avoid', {k: round(float(v), 3) for k, v in zip(TIP, out['avoid_erg%g' % lam])}, flush=True)
np.savez('data/calib_ergodic.npz', tip_names=np.array(list(TIP)), **out)
