"""Disclosure in instance C (native payoffs, general sum): hidden vs disclose-before-freeze.

Usage: python3 run_disclose_c.py K   -> out/disclose_c_K{K}.json
Grid: q in {0.3, 0.6} x lam in {0.3, 0.45, 0.6, 0.75}, persist 0.7, N = 2000. For each regime:
full-game CFR (u_i, NashConv, private value of cancelling, cancel probability) and phi_H's NashConv.
"""
import json
import os
import sys
import time
sys.path.insert(0, "/home/claude/exp/hypotheses"); sys.path.insert(0, "/home/claude/exp")
from cx import train, cancel_stats
from cx_disclose import TreeCD

K = int(sys.argv[1])
os.makedirs("/home/claude/exp/hypotheses/out", exist_ok=True)
out, t00 = [], time.time()
for q in (0.3, 0.6):
    for lam in (0.3, 0.45, 0.6, 0.75):
        rec = {"K": K, "q": q, "lam": lam, "persist": 0.7, "N": 2000}
        for disclose in (False, True):
            t0 = time.time()
            T = TreeCD(K, q=q, lam=lam, disclose=disclose)
            S = train(T, "full", 2000)
            tab, ids = S.compiled()
            u, gaps, f = S.evaluate(tab, ids)
            cs = cancel_stats(T, S, f)
            H = train(T, "H", 2000)
            th, ih = H.compiled()
            uh, gh, _ = H.evaluate(th, ih)
            rec["dis" if disclose else "hid"] = {
                "nodes": int(T.M), "u1": u[1], "u2": u[2], "nashconv": float(sum(gaps.values())),
                "pv1": cs[1]["pv_exact"], "pv2": cs[2]["pv_exact"], "p_cancel1": cs[1]["p_cancel"],
                "submit_by_t1": cs[1]["submit_by_t"], "H_nashconv": float(sum(gh.values())), "H_u1": uh[1],
                "entries_full": int(S.bmask.sum()), "entries_H": int(H.bmask.sum()), "seconds": time.time() - t0}
        out.append(rec)
        h, d = rec["hid"], rec["dis"]
        print(f"K={K} q={q} lam={lam}: HIDDEN u1={h['u1']:.4f} pv1={h['pv1']:.4f} pc1={h['p_cancel1']:.3f} nc={h['nashconv']:.1e} "
              f"Hnc={h['H_nashconv']:.1e} || DISCLOSE u1={d['u1']:.4f} u2={d['u2']:.4f} pv1={d['pv1']:.4f} pc1={d['p_cancel1']:.3f} "
              f"nc={d['nashconv']:.1e} Hnc={d['H_nashconv']:.1e} | du1={d['u1'] - h['u1']:+.4f} "
              f"{h['seconds'] + d['seconds']:.1f}s", flush=True)
        json.dump(out, open(f"/home/claude/exp/hypotheses/out/disclose_c_K{K}.json", "w"), indent=1)
print(f"total {time.time() - t00:.1f}s")
