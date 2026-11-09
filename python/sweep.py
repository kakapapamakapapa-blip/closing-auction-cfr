"""Section 6.10: robustness over random instances of instance B (K = 2).

For each random parameter draw:
  * native general-sum game and auxiliary zero-sum game Gamma_0: NashConv of the compiled tables
    and of the CFR mixtures for full, phi_H, forget s0 after s1, drop s0, drop chi;
  * disclose-before-freeze regime in Gamma_0: NashConv of full and phi_H.
Usage: python3 sweep.py FIRST_SEED COUNT N OUT.json
"""
import json
import sys
import numpy as np
from market import Market, Params
from cfr import Solver
from general_sum import GSolver

VARS = ["full", "H", "H_s0m", "H_s0", "H_chi"]


def draw(rng):
    while True:
        p = dict(lam=rng.uniform(0.5, 3.0), stay=rng.uniform(0.5, 0.9),
                 pp0=rng.uniform(0.05, 0.3), pp1=rng.uniform(0.1, 0.5),
                 pm0=rng.uniform(0.3, 0.7), pm1=rng.uniform(0.1, 0.5),
                 askD=rng.uniform(100.2, 100.8), ask0=rng.uniform(100.2, 100.8), aslope=rng.uniform(0.2, 0.8))
        p["askU"] = p["askD"] + rng.uniform(0.5, 1.5)
        lo_pm = p["pm0"] - p["pm1"]
        if (p["pp0"] + p["pp1"] <= 0.9 and lo_pm >= 0.02 and p["pp0"] + p["pm0"] <= 0.98
                and p["pp0"] + p["pp1"] + lo_pm <= 0.98):
            return {k: float(v) for k, v in p.items()}


def run_native(T, v, N):
    S = GSolver(T, v); S.setup()
    for _ in range(N):
        S.step()
    tab, ids = S.compiled(); tabm, idsm = S.mixture()
    _, g, _ = S.evaluate(tab, ids); _, gm, _ = S.evaluate(tabm, idsm)
    return {"nc_table": sum(g.values()), "nc_mixture": sum(gm.values())}


def run_zero_sum(T, v, N):
    S = Solver(T, v)
    for _ in range(N):
        S.step()
    tab, ids = S.compiled(); tabm, idsm = S.mixture()
    f = S.edge_factors(tab, ids)
    return {"nc_table": float(S.nashconv(tab, ids)[0]), "nc_mixture": float(S.nashconv(tabm, idsm)[0]),
            "v_hi": float(S.br(1, f)), "v_lo": float(-S.br(2, f))}


if __name__ == "__main__":
    first, count, N, outp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    out = []
    for seed in range(first, first + count):
        rng = np.random.default_rng(1000 + seed)
        prm = draw(rng)
        T = Market(Params(K=2, cont=True, **prm))
        rec = {"seed": seed, "params": prm, "native": {}, "gamma0": {}, "disclose_gamma0": {}}
        for v in VARS:
            rec["native"][v] = run_native(T, v, N)
            rec["gamma0"][v] = run_zero_sum(T, v, N)
        Td = Market(Params(K=2, cont=True, disclose=True, **prm))
        for v in ("full", "H"):
            rec["disclose_gamma0"][v] = run_zero_sum(Td, v, N)
        out.append(rec)
        print(seed, json.dumps({m: {v: round(rec[m][v]["nc_table"], 5) for v in rec[m]} for m in ("native", "gamma0", "disclose_gamma0")}), flush=True)
        json.dump(out, open(outp, "w"), indent=1)
