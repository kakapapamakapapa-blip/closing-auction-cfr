# Closing-auction experiments

Code for Chapter 7 of the thesis "Solving Discrete Double Auctions using Imperfect-Recall Abstractions in
CFR: Accuracy–Complexity Trade-offs". Requires Python 3 with NumPy (and matplotlib for the figures).
Every command runs on one core and writes a JSON file; `figures.py` draws the figures from these files
into `../thesis/figures`.

## Games and solvers

| File | Contents |
| --- | --- |
| `auction.py` | instance A (buyer against a liquidity provider; hidden cancellable LOC orders; signal, exogenous interest, imbalance indicator) and its model game |
| `auction_cont.py` | instance B: instance A plus a continuous market open before and after the cutoff |
| `auction_c.py` | instance C: two buyers race for priority with market-on-close orders; optional ban on cancellation |
| `market.py` | parametrized instances A and B, including disclosure of revocable interest before the cutoff |
| `cfr.py` | vectorized vanilla CFR on bucket abstractions for the zero-sum game Γ₀, exact perfect-recall best responses, compiled tables, CFR mixtures, drift |
| `general_sum.py` | the same for native (general-sum) payoffs |
| `mccfr.py` | external-sampling MCCFR on buckets |

## Experiments

| Script | Command | Output | Thesis |
| --- | --- | --- | --- |
| `cancel_dev.py` | `python3 cancel_dev.py 3 2000 GRID _grid` | `cancel_dev_K3_grid.json` | Table 7.3 |
| `cancel_value.py` | `python3 cancel_value.py 2 2000 GRID _grid` (also K = 3) | `cancel_value_K*_grid.json` | Section 7.2 |
| `exact_c4.py` | `python3 exact_c4.py 4 300 0.6 0.6` | `exact_C_K4.json` | Section 7.2 |
| `gs_cont.py` | `python3 gs_cont.py 3 10000` | `general_sum_cont_K3.json` | Table 7.4 |
| `gamma0_cont.py` | `python3 gamma0_cont.py 3 10000` | `gamma0_cont_K3.json` | Table 7.4, Section 7.7 |
| `frontier.py` | `python3 frontier.py B 3 2000`; `python3 frontier.py C 3 2000 0.6 0.6` | `frontier_*_K3.json` | Figure 7.1 |
| `experiment.py` | `python3 experiment.py 3 10000 5` (also K = 2) | `results_K*.json` | Table 7.5, Figure 7.2 |
| `general_sum.py` | `python3 general_sum.py 2 3000` | `general_sum_K2.json` | Section 7.1 |
| `certificate.py` | `python3 certificate.py 3 10000` | `certificate_K3.json` | Table 7.6 |
| `apriori.py` | `python3 apriori.py` | `apriori_K3.json` | Table 7.6 (worst-case column) |
| `model_check.py` | `python3 model_check.py 3 10000 H_s0` (also `H_s0m`) | `model_check_K3_*.json` | Table 7.7 |
| `deploy_check.py` | `python3 deploy_check.py 3 10000 H_s0m,H_s0` | `deploy_check_K3.json` | Table 7.8 |
| `scaling.py` | `python3 scaling.py` | `scaling_A.json` | Figure 7.3 |
| `mccfr.py` | `python3 mccfr.py K B 10 full,H _seeds10` with (K, B) = (2, 1e7), (3, 3.2e7), (4, 1.5e8) | `mccfr_K*_seeds10.json` | Table 7.9, Figure 7.4 |
| `mccfr_stats.py` | `python3 mccfr_stats.py 2 3 4` | `mccfr_stats.json` | Table 7.9 |
| `vanilla_curve.py` | `python3 vanilla_curve.py 4 1000 full,H,A` | `vanilla_K4.json` | Figure 7.4 |
| `timing.py` | `python3 timing.py 2e7 3` | `timing.json` | Table 7.9 |
| `sweep.py` | `python3 sweep.py 0 40 2000 sweep_40.json` | `sweep_40.json` | Figure 7.5, Section 7.9 |
| `sweep_analysis.py` | `python3 sweep_analysis.py sweep_40.json` | `sweep_summary.json` | Section 7.9 |
| `disclose_curve.py` | `python3 disclose_curve.py 10000` | `disclose_curve.json` | Section 7.9 |
| `resolve.py` | `python3 resolve.py 3 2000 3e6 300 1000`; `python3 resolve.py 4 2000 1.5e7 300 1000` (uses `staged.py`) | `resolve_K*.json` | Table 7.10 |
| `hawkes_check.py` | `python3 hawkes_check.py 1000000 2` | `hawkes_check.json` | Table 7.11 |
| `figures.py` | `python3 figures.py` | `../thesis/figures/*.pdf` | Figures 7.1–7.5 |

GRID is a JSON list of settings, e.g. `'[{"q": 0.3, "lam": 0.3}, {"q": 0.3, "lam": 0.45}]'`. The grids used
are q ∈ {0.3, 0.6} × λ ∈ {0.3, 0.45, 0.6, 0.75} for `cancel_dev.py` (K = 2, 3); q ∈ {0.2, 0.4, 0.6, 0.8} ×
λ ∈ {0.3, 0.6, 0.9} for `cancel_value.py` at K = 2, and q = 0.6 × λ ∈ {0.3, 0.45, 0.6, 0.75} at K = 3.

All exploitabilities are exact: best responses are computed over the perfect-recall information sets of
the full tree. The tree is built for this evaluation; the φ_H keys are computed from the simulator state.
