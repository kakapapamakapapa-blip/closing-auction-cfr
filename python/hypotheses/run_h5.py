"""H5: does phi_H_nosig (forget every price signal) reach full-game accuracy in instance C?

Usage: python3 run_h5.py k2 | k3   -> out/h5_k2.json / out/h5_k3.json
General-sum CFR (native payoffs, N = 2000, compiled tables), exact NashConv in the full game.
k2: q in {0.2, 0.4, 0.6, 0.8} x lam in {0.3, 0.6, 0.9} x persist in {0.5, 0.7, 0.9}; variants full, H, H_nosig.
k3: q in {0.3, 0.6} x lam in {0.3, 0.6, 0.9}, persist 0.7; variants full, H_nosig.
"""
import json
import os
import sys
import time
sys.path.insert(0, "/home/claude/exp/hypotheses"); sys.path.insert(0, "/home/claude/exp")
from cx import TreeCX, train


def run(K, q, lam, persist, variants, N=2000):
    T = TreeCX(K, q=q, lam=lam, persist=persist)
    rec = {"K": K, "q": q, "lam": lam, "persist": persist, "N": N, "nodes": int(T.M)}
    for v in variants:
        t0 = time.time()
        S = train(T, v, N)
        tab, ids = S.compiled()
        u, gaps, f = S.evaluate(tab, ids)
        rec[v] = {"entries": int(S.bmask.sum()), "nashconv": float(sum(gaps.values())), "u1": u[1], "u2": u[2],
                  "p_cancel_p1": float(S.values_p(f, T.cycled)[0]), "seconds": time.time() - t0}
    return rec


if __name__ == "__main__":
    chunk = sys.argv[1]
    os.makedirs("/home/claude/exp/hypotheses/out", exist_ok=True)
    jobs = ([(2, q, lam, p, ("full", "H", "H_nosig")) for p in (0.5, 0.7, 0.9) for q in (0.2, 0.4, 0.6, 0.8)
             for lam in (0.3, 0.6, 0.9)] if chunk == "k2" else
            [(3, q, lam, 0.7, ("full", "H_nosig")) for q in (0.3, 0.6) for lam in (0.3, 0.6, 0.9)])
    out, t0 = [], time.time()
    for K, q, lam, p, vs in jobs:
        rec = run(K, q, lam, p, vs)
        out.append(rec)
        a, b = rec["full"], rec["H_nosig"]
        print(f"K={K} q={q} lam={lam} p={p}: full nc={a['nashconv']:.2e} u1={a['u1']:.4f} | H_nosig nc={b['nashconv']:.2e} "
              f"u1={b['u1']:.4f} ratio={b['nashconv'] / a['nashconv']:.2f} | entries {a['entries']}->{b['entries']} "
              f"{sum(rec[v]['seconds'] for v in vs):.1f}s", flush=True)
        json.dump(out, open(f"/home/claude/exp/hypotheses/out/h5_{chunk}.json", "w"), indent=1)
    print(f"total {time.time() - t0:.1f}s")
