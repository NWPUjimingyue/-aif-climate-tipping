"""Calibration with climate data, model with design-dependent abatement depth (depth.py).

Data (all values entered by hand from the cited tables):
  [GCB25]  Friedlingstein et al., Global Carbon Budget 2025, Earth Syst. Sci. Data 18, 3211 (2026),
           doi:10.5194/essd-18-3211-2026; data doi:10.18160/GCP-2025.
           Projected 2025 CO2 emissions: fossil 38.1 GtCO2, land-use change 4.1 GtCO2.
           Remaining budgets from January 2026 (50% likelihood): 170, 525, 1055 GtCO2 for 1.5, 1.7, 2.0 C.
  [EGR25]  UNEP, Emissions Gap Report 2025: Off Target (Nairobi, 2025), doi:10.59117/20.500.11822/48854.
           Table 2.1: global GHG emissions in 2024, 57.7 GtCO2e.
           Table ES.2, median global GHG emissions (GtCO2e) in 2030, 2035, 2050:
             below 2.0 C (66%):                         41, 36, 20
             around 1.5 C (50%, no or limited overshoot): 33, 25,  8
  [AMK22]  Armstrong McKay et al., Science 377, eabn7950 (2022), Table 1: tipping thresholds.

Mapping:
  one round = one year, 2026 to 2049 (N = 24); emissions end after 2049 (exogenous decarbonization).
  unabated round: E0 = 38.1 + 4.1 = 42.2 GtCO2 [GCB25].
  depth of the compromise alpha_c: mean fractional cut below the 2024 level over 2026-2049 on the 2 C
  pathway [EGR25], with linear interpolation through (2024, 57.7), (2030, .), (2035, .), (2050, .);
  depth of a preferred design alpha_m: the same for the 1.5 C pathway. Fractional cuts of total GHG
  emissions are applied to CO2.
  budget B(T): piecewise linear through the three [GCB25] points, end slopes outside; b(T) = B/(E0 N).
"""
import numpy as np, time
from scipy.stats import binom, triang
from depth import DepthParams, TablePolicyD, cumulative_emissions, solve_dp_depth, thresholds_depth

E24 = 57.7
PATH = {'2C': (41.0, 36.0, 20.0), '1.5C': (33.0, 25.0, 8.0)}
YRS = np.arange(2026, 2050)
def mean_cut(v):
    E = np.interp(YRS, [2024, 2030, 2035, 2050], [E24, *v])
    return 1 - E.mean() / E24
alpha_c, alpha_m = mean_cut(PATH['2C']), mean_cut(PATH['1.5C'])
E0, N, s0, nrun = 38.1 + 4.1, 24, 0.1, 40000
Tk = np.array([1.5, 1.7, 2.0]); Bk = np.array([170.0, 525.0, 1055.0])
def budget(T):
    T = np.asarray(T, float)
    lo = Bk[0] + (Bk[1] - Bk[0]) / (Tk[1] - Tk[0]) * (T - Tk[0])
    hi = Bk[1] + (Bk[2] - Bk[1]) / (Tk[2] - Tk[1]) * (T - Tk[1])
    return np.where(T <= Tk[1], lo, hi)
def warming(B):
    B = np.asarray(B, float)
    return np.where(B <= Bk[1], Tk[0] + (B - Bk[0]) * (Tk[1] - Tk[0]) / (Bk[1] - Bk[0]),
                    Tk[1] + (B - Bk[1]) * (Tk[2] - Tk[1]) / (Bk[2] - Bk[1]))
TIP = {  # [AMK22] Table 1: central estimate, minimum, maximum (degC)
    'GrIS': (1.5, 0.8, 3.0), 'WAIS': (1.5, 1.0, 3.0), 'LABC': (1.8, 1.1, 3.8), 'EASB': (3.0, 2.0, 6.0),
    'AMAZ': (3.5, 2.0, 6.0), 'PFTP': (4.0, 3.0, 6.0), 'AMOC': (4.0, 1.4, 8.0), 'AWSI': (6.3, 4.5, 8.7),
    'EAIS': (7.5, 5.0, 10.0), 'REEF': (1.5, 1.0, 2.0), 'PFAT': (1.5, 1.0, 2.3), 'BARI': (1.6, 1.5, 1.7),
    'GLCR': (2.0, 1.5, 3.0), 'SAHL': (2.8, 2.0, 3.5), 'BORF': (4.0, 1.4, 5.0), 'TUND': (4.0, 1.5, 7.2)}


def threshold_cdf(T, c, a, bmax):
    """Distribution function of a triangular threshold density on [a, bmax] with mode c."""
    return triang.cdf(T, (c - a) / (bmax - a), loc=a, scale=bmax - a)

OUTFILE = 'data/calib_depth.npz'

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        OUTFILE = sys.argv[1]
    P = DepthParams(alpha_c, alpha_m)
    th = thresholds_depth(P)
    print('alpha_c=%.4f alpha_m=%.4f lam_c(depth)=%.3f' % (alpha_c, alpha_m, th['lam_c']), flush=True)
    rng = np.random.default_rng(2026)
    Tg = np.round(np.arange(1.55, 1.951, 0.0025), 4)
    Bg = budget(Tg) / E0                        # budget in units of one unabated round
    out = dict(T=Tg, b=Bg / N, E0=E0, N=N, alpha_c=alpha_c, alpha_m=alpha_m, lam_c=th['lam_c'])
    S, Tend, Ucum = {}, {}, {}
    agents = {'aif1.4': (1.4, True), 'aif3': (3.0, True), 'aif20': (20.0, True), 'prag': (1e6, False)}
    for k, (lam, epi) in agents.items():
        pol = TablePolicyD(P, lam, epi)
        U = np.concatenate([cumulative_emissions(P, pol, th_, rng.normal(0, s0, nrun), N, rng)[0] for th_ in ['L', 'H']])
        Ucum[k] = U
        S[k] = np.array([np.mean(U <= B + 1e-9) for B in Bg])
        Tend[k] = warming(U * E0)
        print(k, 'mean warming reached %.3f' % Tend[k].mean(), flush=True)
    kk = np.arange(N + 1); pk = binom.pmf(kk, N, P.p_max)
    S['oracle'] = np.array([pk[(N - alpha_m * kk) <= B + 1e-9].sum() for B in Bg])
    ell0 = rng.normal(0, s0, 20000)
    q = 0.0025
    Sbo_fine = {}
    for rd, key in [('floor', 'bo'), ('ceil', 'bo_lo')]:
        t0 = time.time()
        o = solve_dp_depth(P, N, Bg.max() + 0.01, q=q, nl=301, nx=81, rounding=rd)
        Vb = np.array([np.mean(np.interp(ell0, o['ell'], row)) for row in o['V0']])   # survival at every grid budget
        Sbo_fine[key] = (o['B'], Vb)
        S[key] = np.interp(Bg, o['B'], Vb)
        print('DP %s %.0fs' % (rd, time.time() - t0), flush=True)
    out.update(S)
    lo, hi = 1 - alpha_m * P.p_max, 1 - alpha_c * P.sigma_mid
    Tlo, Thi = float(warming(lo * E0 * N)), float(warming(hi * E0 * N))
    out.update(T_lo=Tlo, T_hi=Thi)
    m = (Tg >= Tlo) & (Tg <= Thi)
    print('window %.3f-%.3f C; excess %.0f GtCO2' % (Tlo, Thi, (hi - lo) * E0 * N))
    print('window mean', {k: round(float(v[m].mean()), 3) for k, v in S.items()})
    for T in [1.65, 1.7, 1.75]:
        print('T=%.2f' % T, {k: round(float(np.interp(T, Tg, v)), 3) for k, v in S.items()})
    # probability of avoiding each element, P_k = E[1 - F_k(T_reached)] for the budget-independent buyers
    # (exact over the sampled runs and the binomial law), and the integral of f_k S on the fine budget grid
    # for the Bayes-optimal buyer, which plans for the budget of the drawn threshold
    order = ['oracle', 'bo', 'bo_lo', 'aif1.4', 'aif3', 'aif20', 'prag']
    T_or = warming((N - alpha_m * kk) * E0)
    avoid = []
    for name, (c, a, bmax) in TIP.items():
        row = []
        for k in order:
            if k == 'oracle':
                row.append(float(np.sum(pk * (1 - threshold_cdf(T_or, c, a, bmax)))))
            elif k in ('bo', 'bo_lo'):
                Bf, Vb = Sbo_fine[k]
                Tf = warming(Bf * E0)
                F = threshold_cdf(Tf, c, a, bmax)
                # P_k = int f_k(T) S(T) dT = sum over grid cells of S times the threshold mass in the cell,
                # plus the mass above the largest budget, where S = 1
                row.append(float(np.sum(0.5 * (Vb[1:] + Vb[:-1]) * np.diff(F)) + (1 - F[-1])))
            else:
                row.append(float(np.mean(1 - threshold_cdf(Tend[k], c, a, bmax))))
        avoid.append(row); print(name, [round(v, 3) for v in row], flush=True)
    np.savez(OUTFILE, **out, order=np.array(order), tip_names=np.array(list(TIP)),
             tip_vals=np.array([TIP[k] for k in TIP]), avoid=np.array(avoid),
             Tend_prag=Tend['prag'], Tend_aif=Tend['aif1.4'])
