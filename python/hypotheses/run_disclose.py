"""H3: does disclosing revocable on-close interest before the cutoff change the zero-sum value of Gamma_0?

Usage: python3 run_disclose.py lp    -> out/disclose_lp.json   (exact sequence-form LP values)
       python3 run_disclose.py cfr   -> out/disclose_cfr.json  (CFR N = 2000: certified intervals,
                                         phi_H NashConv in Gamma_0, native general-sum payoffs)
Instances (K = 2): default A (cont=False), default B (cont=True), the 40 instance-B draws of
sweep_40.json, and 10 new instance-A draws (sweep.draw with seeds 5000..5009).
Mechanism check (lp): value of the hidden regime when player 1 (resp. 2) may not place on-close
orders before the last pre-cutoff period (t < K - 1): if both bans leave the value unchanged, each
player has an optimal strategy that discloses nothing, so the disclosure regime has the same value.
"""
import json
import os
import sys
import time
sys.path.insert(0, "/home/claude/exp/hypotheses"); sys.path.insert(0, "/home/claude/exp")
import numpy as np
from market import Market, Params
from cfr import Solver
from general_sum import GSolver
from sweep import draw
from seqlp import game_value, value_both_sides

K = 2


def instances():
    out = [("A_default", {"cont": False}), ("B_default", {"cont": True})]
    for r in json.load(open("/home/claude/exp/sweep_40.json")):
        out.append((f"B_sweep{r['seed']}", dict(cont=True, **r["params"])))
    for s in range(10):
        out.append((f"A_draw{s}", dict(cont=False, **draw(np.random.default_rng(5000 + s)))))
    return out


def early_order(T, K):
    def pred(n, a):
        return T.keys[n]["full"][1] < K - 1 and T.acts[n][a] in ("lo", "hi")
    return pred


def run_lp():
    res = []
    for name, prm in instances():
        rec = {"name": name, "params": prm}
        t0 = time.time()
        for cancel in (True, False):
            for disclose in (False, True):
                T = Market(Params(K=K, cancel=cancel, disclose=disclose, **prm))
                v1, v2 = value_both_sides(T)
                rec[f"v_{'dis' if disclose else 'hid'}_{'c' if cancel else 'nc'}"] = v1
                rec[f"lp_side_gap_{'dis' if disclose else 'hid'}_{'c' if cancel else 'nc'}"] = abs(v1 - v2)
                if cancel and not disclose:
                    pred = early_order(T, K)
                    rec["v_hid_c_ban_early_p1"] = game_value(T, ban1=pred)[0]
                    rec["v_hid_c_ban_early_p2"] = game_value(T, ban2=pred)[0]
                if cancel and disclose:
                    pred = early_order(T, K)
                    rec["v_dis_c_ban_early_p1"] = game_value(T, ban1=pred)[0]
                    rec["v_dis_c_ban_early_p2"] = game_value(T, ban2=pred)[0]
        rec["seconds"] = time.time() - t0
        res.append(rec)
        print(f"{name:12s} hid_c={rec['v_hid_c']:.9f} dis_c={rec['v_dis_c']:.9f} d={rec['v_dis_c']-rec['v_hid_c']:+.2e} | "
              f"hid_nc={rec['v_hid_nc']:.9f} dis_nc={rec['v_dis_nc']:.9f} d={rec['v_dis_nc']-rec['v_hid_nc']:+.2e} | "
              f"ban1={rec['v_hid_c_ban_early_p1']-rec['v_hid_c']:+.2e} ban2={rec['v_hid_c_ban_early_p2']-rec['v_hid_c']:+.2e} "
              f"{rec['seconds']:.1f}s", flush=True)
    return res


def run_cfr(N=2000):
    res = []
    for name, prm in instances():
        if name.startswith("B_sweep"):
            continue                      # CFR results for these draws are already in sweep_40.json
        rec = {"name": name, "params": prm, "N": N}
        t0 = time.time()
        for disclose in (False, True):
            T = Market(Params(K=K, disclose=disclose, **prm))
            tag = "dis" if disclose else "hid"
            for v in ("full", "H"):
                S = Solver(T, v)
                for _ in range(N):
                    S.step()
                tab, ids = S.compiled(); f = S.edge_factors(tab, ids)
                vhi, vlo = float(S.br(1, f)), float(-S.br(2, f))
                rec[f"g0_{tag}_{v}"] = {"v_lo": vlo, "v_hi": vhi, "nashconv": vhi - vlo}
                G = GSolver(T, v); G.setup()
                for _ in range(N):
                    G.step()
                tab, ids = G.compiled()
                u, gaps, _ = G.evaluate(tab, ids)
                rec[f"native_{tag}_{v}"] = {"u1": u[1], "u2": u[2], "nashconv": float(sum(gaps.values()))}
        rec["seconds"] = time.time() - t0
        res.append(rec)
        print(name, json.dumps({k: ({kk: round(vv, 6) for kk, vv in val.items()} if isinstance(val, dict) else val)
                                for k, val in rec.items() if k.startswith(("g0", "native"))}), f"{rec['seconds']:.1f}s", flush=True)
    return res


if __name__ == "__main__":
    os.makedirs("/home/claude/exp/hypotheses/out", exist_ok=True)
    mode = sys.argv[1]
    t0 = time.time()
    out = run_lp() if mode == "lp" else run_cfr()
    json.dump(out, open(f"/home/claude/exp/hypotheses/out/disclose_{mode}.json", "w"), indent=1)
    print(f"total {time.time() - t0:.1f}s")
