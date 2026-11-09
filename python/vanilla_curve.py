"""Vanilla CFR exploitability against iterations, node visits (M per iteration) and seconds."""
import json
import sys
import time
from auction import Tree
from cfr import Solver

K, N = int(sys.argv[1]), int(sys.argv[2])
variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["full"]
T = Tree(K)
marks = sorted({int(round(x)) for x in [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000] if x <= N} | {N})
out = {"K": K, "M": int(T.M), "runs": []}
for v in variants:
    S, curve, t0, spent = Solver(T, v), [], time.time(), 0.0
    for n in range(1, N + 1):
        t1 = time.time(); S.step(); spent += time.time() - t1
        if n in marks:
            tab, ids = S.compiled()
            curve.append({"n": n, "visits": n * T.M, "seconds": spent, "nc": float(S.nashconv(tab, ids)[0])})
    out["runs"].append({"variant": v, "curve": curve})
    print(v, [(c["n"], round(c["nc"], 4), round(c["seconds"], 1)) for c in curve], flush=True)
json.dump(out, open(f"vanilla_K{K}.json", "w"), indent=1)
