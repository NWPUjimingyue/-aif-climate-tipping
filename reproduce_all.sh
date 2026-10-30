#!/usr/bin/env bash
# Regenerate every data file in data/ and the figures in figures/.
# Run from the repository root. About two hours on two cores.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data figures
python3 exp_fig2.py
python3 exp_gamma.py
python3 exp_survival_curves.py
python3 exp_heat.py 0 2 & p0=$!; python3 exp_heat.py 1 2 & p1=$!; wait $p0; wait $p1
python3 exp_N.py
python3 exp_fold.py
python3 exp_fold3.py
python3 exp_calib.py
python3 exp_calib_depth.py
python3 check/asym.py
python3 check/theory_checks.py
python3 exp_universal.py lam
pids=(); for c in logit20 logit8 logit60 logit95 probit95 seller2; do python3 exp_universal.py nu $c & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
python3 exp_universal.py merge
python3 exp_sale.py
for m in fan surv_b surv_lam tlearn; do python3 exp_ergodic.py $m; done
python3 exp_calib_ergodic.py
python3 exp_phase.py
python3 make_figures.py figures
python3 paper_numbers.py
