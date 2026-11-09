"""H1 (option value of cancelling vs signal informativeness and K) and H2 (pro rata vs time priority).

Usage: python3 run_cancel.py CHUNK   with CHUNK in k2, k3a, k3b, k3c   -> out/cancel_CHUNK.json
Each record: CFR (native payoffs, N = 2000, full game, compiled tables) on instance C with
cancellation allowed; exact private value of the right to cancel for each player against the
opponent's CFR strategy; cancel / submission probabilities; NashConv; seconds.
"""
import json
import os
import sys
import time
sys.path.insert(0, "/home/claude/exp/hypotheses")
from cx import solve_and_measure

GRID8 = [(q, lam) for q in (0.3, 0.6) for lam in (0.3, 0.45, 0.6, 0.75)]
GRID3 = [(0.6, 0.3), (0.6, 0.6), (0.3, 0.75)]
PERSISTS = (0.5, 0.6, 0.7, 0.8, 0.9)


def jobs(chunk):
    J = []
    if chunk == "k2":
        for q, lam in GRID8:
            for p in PERSISTS:
                J.append(dict(K=2, q=q, lam=lam, persist=p, slope=0.6, alloc="time", nocancel=True, tag="H1_persist"))
            for p in (0.5, 0.9):
                J.append(dict(K=2, q=q, lam=lam, persist=p, slope=0.0, alloc="time", tag="H1_flat"))
            for s in (0.3, 0.9):
                J.append(dict(K=2, q=q, lam=lam, persist=0.7, slope=s, alloc="time", tag="H1_slope"))
            for p in (0.5, 0.7, 0.9):
                J.append(dict(K=2, q=q, lam=lam, persist=p, slope=0.6, alloc="prorata", nocancel=True, tag="H2_prorata"))
    elif chunk == "k3a":
        for q, lam in GRID3:
            for p in PERSISTS:
                J.append(dict(K=3, q=q, lam=lam, persist=p, slope=0.6, alloc="time", tag="H1_persist"))
    elif chunk == "k3b":
        for q, lam in GRID3:
            for p in (0.5, 0.9):
                J.append(dict(K=3, q=q, lam=lam, persist=p, slope=0.0, alloc="time", tag="H1_flat"))
            for s in (0.3, 0.9):
                J.append(dict(K=3, q=q, lam=lam, persist=0.7, slope=s, alloc="time", tag="H1_slope"))
    elif chunk == "k3c":
        for q, lam in GRID8:
            J.append(dict(K=3, q=q, lam=lam, persist=0.7, slope=0.6, alloc="prorata", tag="H2_prorata"))
            if (q, lam) not in GRID3:
                J.append(dict(K=3, q=q, lam=lam, persist=0.7, slope=0.6, alloc="time", tag="H2_time"))
    return J


if __name__ == "__main__":
    chunk = sys.argv[1]
    os.makedirs("/home/claude/exp/hypotheses/out", exist_ok=True)
    path = f"/home/claude/exp/hypotheses/out/cancel_{chunk}.json"
    out, t0 = [], time.time()
    for j in jobs(chunk):
        tag = j.pop("tag")
        rec = solve_and_measure(N=2000, **j)
        rec["tag"] = tag
        out.append(rec)
        P1, P2 = rec["players"][1], rec["players"][2]
        print(f"{tag:11s} K={rec['K']} q={rec['q']} lam={rec['lam']} p={rec['persist']} s={rec['slope']} {rec['alloc']:7s} "
              f"nc={rec['nashconv']:.1e} pv1={P1['pv_exact']:.4f} pv2={P2['pv_exact']:.4f} "
              f"pc1={P1['p_cancel']:.3f} pc2={P2['p_cancel']:.3f} pre1={P1['n_submit_pre']:.3f} u1={rec['u1']:.4f} "
              f"{rec['seconds']:.1f}s", flush=True)
        json.dump(out, open(path, "w"), indent=1)
    print(f"total {time.time() - t0:.1f}s", flush=True)
