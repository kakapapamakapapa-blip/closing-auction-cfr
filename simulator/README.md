# Nasdaq Closing Cross: simulator + CFR

`ncc.c` - simulator with the real Closing Cross timeline (Nasdaq FAQ 2025): dark window before 15:50 (MOC can be
submitted and cancelled, nothing is published), 330 NOII publications (every 10 s 15:50-15:55, every 1 s
15:55-16:00), MOC until 15:55, LOC until 15:58, IO until 16:00, no cancellation after 15:50, cross at 16:00
(maximum volume, then minimum imbalance, then closest to the midpoint).
Players: agent (must buy Q=2 lots, penalty LAM per unfilled lot), strategic liquidity provider (IO orders).
Exogenous: hidden index MOC flow, late MOC, offsetting liquidity attracted by the published imbalance,
continuous book (1 lot per tick), midpoint drift. The flow is stylized, NOT calibrated to market data.
Solver: outcome-sampling MCCFR, RM+, linear averaging, VR-MCCFR action baselines.
Evaluation: Monte Carlo; best responses learned by MCCFR on key variants 0 and 1 (the better is kept),
so exploitability (nashconv_lb) is a LOWER bound.

Build: `gcc -O2 -march=native -o ncc ncc.c -lm`
- `./ncc train VAR_A VAR_L ITERS D SEED BR_ITERS NEVAL` (env `LAM=1` changes the penalty)
- `./ncc heur H D SEED BR_ITERS NEVAL` (H=1 MOC early, 2 MOC at end of dark window, 3 imbalance rule)
- `./ncc speed D`
Key variants: 0 time+own state; 1 +signal count +current NOII; 2 +previous NOII +midpoint;
3 perfect recall at decision points with phi_H (own pre-cutoff actions -> resting set);
4 full perfect recall at decision points; 5 perfect recall of all 330 publications.
D = number of post-cutoff decision points (330 = decide at every NOII).
`python3 sizes.py` - game sizes at real resolution. `python3 summarize.py res2` - result tables.
Utilities are in ticks relative to the 15:50 midpoint for 2 lots (u1 < 0 is a cost).

## v2: scalable engine (`ncc2.c`)

Build: `gcc -O3 -march=native -pthread -o ncc3 ncc2.c -lm`. Environment:
`THREADS` (Hogwild threads), `LOGT_A`/`LOGT_L`/`LOGT_BR` (log2 table sizes; an entry is 68 bytes),
`CONTBUY=1` (agent may also buy in the continuous book at mid+1 at any decision), `LAM`.
Modes:
- `run VA VL D SEED CKPTS BR_ITERS NEVAL BR_VMASK [SAVE_PREFIX]` - self-play with checkpoints (e.g. `4e6,16e6,64e6`);
  learned best responses at the last checkpoint (key variants in BR_VMASK; lower bound clipped at 0)
- `cross D NEVAL pref:va:vl ...` - tournament: every saved agent against every saved LP (+ MOC-early heuristic)
- `voi VAR D ITERS LPHEUR NEVAL` - single-agent learning against a fixed LP (heuristic, or `LPFILE=... LPVAR=...`)
Extra agent keys: 7 = time+own state+hidden flow type (oracle), 8 = time+own state+current imbalance,
9 = oracle + exact imbalance. Results: `res3/` (original model), `res4/` (CONTBUY=1), D=330 everywhere.
Scripts: `scale.sh` (res3), `scale4.sh` (res4).
