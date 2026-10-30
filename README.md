# Ergodicity breaking of active inference in carbon trading before tipping

Code and data for the Letter

> M. Ji and H. Xing, *Ergodicity Breaking of Active Inference in Carbon Trading before Tipping*, to be submitted to Physical Review Letters (2026).

A buyer proposes carbon-trading contracts to a host whose abatement cost is private, while a finite carbon budget shrinks toward an irreversible tipping point. Near a confounding contract, which every cost type accepts with the same probability, the belief of an active-inference buyer follows a random multiplicative map with multiplier 1 + K(a - p_c), where K is the epistemic loop gain. The expected information gain of active inference equals the ensemble growth of the information of the buyer, whereas learning follows its time-average growth, twice the Lyapunov exponent of this map. Below a universal gain the two have opposite signs, and ignorance becomes an attractor. The loop gain obeys the Curie-Weiss law K = K_inf T_s/(T_s - T) in the epistemic temperature T = 1/lambda. An ergodic epistemic value, the expected growth of the logarithm of the information, removes the attractor at every precision (`ergodic.py`). The repository reproduces every figure and table of the Letter and its Supplemental Material, tests the universality of the normal form on several acceptance models and on a posted-price allowance sale, and `paper_numbers.py` prints the main quoted theoretical values and simulation results.


## Archive and citation

Version 1.3.2 of this code is the GitHub release [v1.3.2](https://github.com/NWPUjimingyue/-aif-climate-tipping/releases/tag/v1.3.2) of https://github.com/NWPUjimingyue/-aif-climate-tipping. Please cite it together with the Letter; `CITATION.cff` holds the metadata, and GitHub shows it under "Cite this repository".

A DOI can be added through Zenodo: log in to zenodo.org with the GitHub account, switch this repository on under *GitHub* in the account menu, and publish a new release. Zenodo then archives the release, reads `.zenodo.json` for title, authors, licence and keywords, and assigns a DOI.

## Changes

- 1.3.2: stored data replaced by the output of a complete rerun of `reproduce_all.sh` (bit-identical except for the earlier separately recomputed lambda = 1.4 entries, which change within sampling error and change no quoted number); figure PDFs without time stamps; corrected standard errors of the learning exponents (`exp_universal.fit_nu`; the slopes are unchanged, Table S3 and Fig. 3(b) updated, `python3 exp_universal.py refit` recomputes them from the stored data); Fig. 2(b) arrows drawn from the computed gains; failures of background jobs now stop `reproduce_all.sh`, which also runs `exp_phase.py`; stale cross-references in comments corrected.
- 1.3.1: first public release.

## Requirements

Python 3.10 or newer with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The code was run with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11. No LaTeX installation is needed for the figures.

## Quick start

```
python3 paper_numbers.py      # main quoted theory values and key results from the stored data (seconds)
python3 make_figures.py       # Figs. 1-4 from the stored data, written to figures/
python3 check/theory_checks.py  # checks of the theorems against the exact functions (about half a minute)
```

The data files used for the manuscript are included in `data/`. To regenerate them from scratch:

```
bash reproduce_all.sh         # about 3 h on two cores
```

Monte Carlo results reproduce within sampling error; the dynamic programs and all theoretical values reproduce exactly.

Notes on reproduction:

- Every script fixes its random seed. The files in `data/` are the output of `reproduce_all.sh` with the released scripts: a complete rerun from an empty `data/` folder in a fresh environment (Python 3.13, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2) reproduces every file bit for bit, and `make_figures.py` then reproduces the figures of the Letter.
- At finite policy precision the pragmatic buyer (no epistemic value) uses lambda = 20 in the policy posterior, so its effective precision is 20 gamma (`exp_gamma.py`).
- The learning exponents of Fig. 3(b) and Table S3 are weighted least-squares fits over 1e-4 <= |delta ell_0| <= 1e-2 with 6e4 runs per point (`exp_universal.py`); `paper_numbers.py` also prints an unweighted fit for the logit host over 1e-6 <= |ell_0| <= 1e-1 from `fig2b.npz`.
- `exp_heat.py` takes two arguments, the part index and the number of parts; `reproduce_all.sh` runs both parts in parallel.

## Contents

| File | Content | Used in |
|---|---|---|
| `core.py` | model, expected free energy, policies (direct minimization and tabulated), closed-loop simulation, Lyapunov and learning exponents | all |
| `universal.py` | general confounding contracts (logit, probit, Gaussian acceptance; allowance sale), gain law, universal functions Lambda(K, p_c), nu(K, p_c), K_a(p_c) | Figs. 2(b) and 3, Secs. S3, S4, S6 |
| `universal_sim.py` | full nonlinear dynamics of the general models with tabulated policies, learning probabilities | Fig. 3(b), Fig. 4(d), Table S3 |
| `dp.py` | Bayes-optimal policy by dynamic programming over (round, budget, log odds) | Fig. 4(a), Table S5 |
| `gamma_sim.py` | closed loop at finite policy precision (Laplace sampling of the policy posterior) | Sec. S7, Table S4 |
| `fold.py` | noisy saddle-node tipping element driven by the budget | Sec. S9 |
| `depth.py` | design-dependent abatement depth, its policies, and the dynamic program with depth | Fig. 4(c), Table S6 |
| `exp_universal.py` | Lyapunov exponents (`lam`), learning exponents (`nu <case>`), and their merge (`merge`) | Fig. 3(a), 3(b), Table S3 |
| `exp_sale.py` | posted prices of the allowance sale | Fig. 4(d) |
| `exp_gamma.py` | escape time and survival at finite policy precision | Sec. S7, Table S4 |
| `exp_survival_curves.py` | survival versus allowance, N = 200 | Fig. 4(a) |
| `exp_heat.py` | survival in the plane of preference precision and allowance (run with arguments `0 2` and `1 2`; auxiliary) | none |
| `exp_N.py` | survival versus number of rounds inside the window | Table S5 |
| `exp_fold.py`, `exp_fold3.py` | survival with the noisy fold and the noise-induced budget loss | Sec. S9 |
| `exp_calib_depth.py` | calibration with emissions, mitigation pathways, carbon budgets and tipping thresholds | Fig. 4(c), Sec. S10, Table S6 |
| `exp_calib.py` | the same calibration with equal depths (auxiliary; the equal-depth window quoted in Sec. S10 is analytic) | none |
| `exp_fig2.py` | belief densities and learning probability versus initial lean for the logit host (auxiliary) | `paper_numbers.py` |
| `exp_phase.py` | learning probability in the plane of preference precision and acceptance probability (auxiliary) | none |
| `check/asym.py` | check of asymmetric logit hosts | Sec. S4 |
| `check/theory_checks.py` | numerical checks of Lemma 1 and Theorems 2, 3, 5, and 6 of the SM (phase boundaries, limits of K_a, nu, information identity and growth rates, gain law, p_a, lambda_a, ergodic epistemic value) | Secs. S3 to S5 |
| `ergodic.py` | ergodic epistemic value and its tabulated policies, with and without abatement depth | Fig. 2(a), Fig. 4, Sec. S5 |
| `exp_ergodic.py` | information of standard and ergodic buyers (`fan`), survival versus allowance (`surv_b`) and versus precision (`surv_lam`), learning times (`tlearn`) | Figs. 2(a), 4(a), 4(b), Table S2 |
| `exp_calib_ergodic.py` | calibration for the ergodic buyer | Fig. 4(c), Table S6 |
| `make_figures.py` | Figs. 1 to 4 from the data files, drawn to the APS figure guidelines (one column, 8.6 cm; lettering of at least 2 mm) | Figs. 1 to 4 |
| `brain.py` | vector brain glyph of the active-inference buyer in Fig. 1(a), with gyri from a fixed-seed labyrinth pattern | Fig. 1(a) |
| `paper_numbers.py` | the main numbers quoted in the text, recomputed | text, Tables S2 to S6 |

## Notation in the code

| Paper | Code |
|---|---|
| host cost types theta = +, - with preferred terms m_+ = (1 + Delta)/2, m_- = (1 - Delta)/2 | types `'H'` and `'L'`, attributes `mH`, `mL` |
| belief w = q(theta = +) and its log odds ell | in `core.py`, `w` is the probability of type `'L'` and `ell` its log odds, i.e. minus the paper's ell; in `universal.py`, `w` and `ell` follow the paper |
| acceptance probability p_c of the compromise; p_max, p_far of the preferred terms by the matching and by the other type | `sigma_mid`, `p_max`, `p_far` |
| spinodal lambda_s and critical precision lambda_c, Eq. (4) | `thresholds(P)['lam_loc']`, `thresholds(P)['rho_max']` |
| epistemic loop gain K, pragmatic gain K_inf, Eqs. (5) and (10); confounding contract x_c and confounded log odds ell_c | `D * g / (pc * (1 - pc))` in `universal_sim.Table`, `Model.theory()`, `Model.confounding()`, `Model.ell_star()` |
| allowance sale with elasticities Gamma_-, Gamma_+ and confounding price x_c | `allowance_sale(aL=Gamma_-, aH=Gamma_+, xs=x_c, pc)` |
| pragmatic buyer (lambda -> infinity, no epistemic value) | `TablePolicy(P, 1e6, epistemic=False)` |
| depths alpha_c, alpha_m and alpha(x), Eq. (S43) | `DepthParams(alpha_c, alpha_m)`, `.depth(x)` |
| sensitivity g_B of the tipping element to the budget, noise intensity D_n | `kB`, `D` |

## External data

All values are entered in `exp_calib_depth.py` and `exp_calib.py` from the cited tables; nothing is downloaded.

| Quantity | Value | Source |
|---|---|---|
| Fossil CO2 emissions, 2025 (projection) | 38.1 GtCO2 | P. Friedlingstein et al., Global Carbon Budget 2025, Earth Syst. Sci. Data 18, 3211 (2026), doi:10.5194/essd-18-3211-2026; data: Global Carbon Project, doi:10.18160/GCP-2025 |
| Land-use change CO2 emissions, 2025 (projection) | 4.1 GtCO2 | same |
| Remaining carbon budgets from January 2026, 50% likelihood | 170, 525, 1055 GtCO2 for 1.5, 1.7, 2.0 C | same |
| Global greenhouse-gas emissions, 2024 | 57.7 GtCO2e | UNEP, Emissions Gap Report 2025: Off Target (Nairobi, 2025), Table 2.1, doi:10.59117/20.500.11822/48854 |
| Median emissions of pathways below 2 C (66%), 2030, 2035, 2050 | 41, 36, 20 GtCO2e | same, Table ES.2 |
| Median emissions of pathways around 1.5 C (no or limited overshoot), 2030, 2035, 2050 | 33, 25, 8 GtCO2e | same, Table ES.2 |
| Tipping thresholds (central, minimum, maximum) | Table 1 | D. I. Armstrong McKay et al., Science 377, eabn7950 (2022), doi:10.1126/science.abn7950 |

## License

MIT, see `LICENSE`. Please cite the Letter when using the code (`CITATION.cff`).
