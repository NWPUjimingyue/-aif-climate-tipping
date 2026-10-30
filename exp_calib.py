"""Calibration with climate data.
One round = one year from January 2026; decarbonization completes after N = 24 rounds (2050).
Unabated CO2 emissions per round E0 = 38.1 (fossil) + 4.1 (land use) = 42.2 GtCO2 (GCB 2025 projection for 2025).
Remaining budgets from January 2026 (GCB 2025, 50% likelihood): 170, 525, 1055 GtCO2 for 1.5, 1.7, 2.0 C.
Budget as a function of the warming level T: piecewise linear through these three points, extended linearly
beyond them with the end slopes. Allowance b(T) = B(T) / (E0 N).
Tipping thresholds (central, min, max) from Armstrong McKay et al., Science 377, eabn7950 (2022), Table 1."""
import numpy as np, time
from scipy.stats import binom
from core import *
from dp import solve_dp
P = Params(); rng = np.random.default_rng(81)
E0, N, s0 = 38.1 + 4.1, 24, 0.1
Tk = np.array([1.5, 1.7, 2.0]); Bk = np.array([170.0, 525.0, 1055.0])
def budget(T):
    T = np.asarray(T, float)
    lo = Bk[0] + (Bk[1] - Bk[0]) / (Tk[1] - Tk[0]) * (T - Tk[0])
    hi = Bk[1] + (Bk[2] - Bk[1]) / (Tk[2] - Tk[1]) * (T - Tk[1])
    return np.where(T <= Tk[1], lo, hi)
def allowance(T):
    return budget(T) / (E0 * N)
TIP = {  # name: (central, min, max) in degC
    'GrIS': (1.5, 0.8, 3.0), 'WAIS': (1.5, 1.0, 3.0), 'LABC': (1.8, 1.1, 3.8), 'REEF': (1.5, 1.0, 2.0),
    'PFAT': (1.5, 1.0, 2.3), 'BARI': (1.6, 1.5, 1.7), 'GLCR': (2.0, 1.5, 3.0), 'AMOC': (4.0, 1.4, 8.0),
    'SAHL': (2.8, 2.0, 3.5), 'EASB': (3.0, 2.0, 6.0), 'AMAZ': (3.5, 2.0, 6.0)}
agents = {'aif1.4': TablePolicy(P, 1.4), 'aif20': TablePolicy(P, 20.0), 'prag': TablePolicy(P, 1e6, False)}
Tg = np.round(np.arange(1.40, 2.401, 0.005), 3)
out = {'T': Tg, 'b': allowance(Tg), 'E0': E0, 'N': N}
S = {k: np.zeros(len(Tg)) for k in list(agents) + ['bo', 'oracle']}
t0 = time.time()
for i, T in enumerate(Tg):
    b = float(allowance(T))
    if b <= 0:
        continue
    B0 = b * N
    for k, pol in agents.items():
        S[k][i] = np.mean([run_table(P, pol, th, rng.normal(0, s0, 20000), N, 20000, rng, B0=B0)['alive'].mean() for th in ['L', 'H']])
    kk = np.arange(N + 1); S['oracle'][i] = binom.pmf(kk, N, P.p_max)[(N - P.alpha * kk) <= B0 + 1e-9].sum()
    o = solve_dp(P, N, B0, nl=401, nx=101); j = o['nB'] - 1
    S['bo'][i] = float(np.mean(np.interp(rng.normal(0, s0, 20000), o['ell'], o['V0'][j])))
    if i % 20 == 0:
        print(T, round(b, 3), {k: round(float(v[i]), 3) for k, v in S.items()}, '%.0fs' % (time.time() - t0), flush=True)
out.update(S)
# probability of avoiding each tipping element: triangular threshold distribution (min, central, max)
from scipy.stats import triang
Tf = np.linspace(0.5, 10.0, 20001)
avoid = {}
for name, (c, a, bmax) in TIP.items():
    pdf = triang.pdf(Tf, (c - a) / (bmax - a), loc=a, scale=bmax - a)
    w = pdf / np.trapezoid(pdf, Tf)
    row = {}
    for k in S:
        sT = np.interp(Tf, Tg, S[k], left=0.0, right=1.0)
        row[k] = float(np.trapezoid(w * sT, Tf))
    avoid[name] = row
    print(name, {k: round(v, 3) for k, v in row.items()}, flush=True)
np.savez('data/calib.npz', **out, tip_names=np.array(list(TIP)), tip_vals=np.array([TIP[k] for k in TIP]),
         avoid=np.array([[avoid[n][k] for k in ['oracle', 'bo', 'aif1.4', 'aif20', 'prag']] for n in TIP]))
print('T=1.7:', {k: round(float(np.interp(1.7, Tg, v)), 3) for k, v in S.items()}, 'b(1.7)=', float(allowance(1.7)))
