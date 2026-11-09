"""Accuracy-complexity frontier: table entries vs NashConv (native payoffs) for many abstractions.
Instance B (auction_cont, continuous market) and instance C (auction_c, MOC race)."""
import json, sys, time
import numpy as np
from general_sum import GSolver

inst, K, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
variants = ["full", "H", "H_s0m", "H_s0", "H_last", "H_chi", "A", "H_nosig"]
if inst == "B":
    from auction_cont import TreeC as TB
    T = TB(K)
else:
    from auction_c import TreeC
    T = TreeC(K, q=float(sys.argv[4]), lam=float(sys.argv[5]))
    variants = [v for v in variants if v != "H_s0m"]
out = {"instance": inst, "K": K, "N": N, "nodes": int(T.M), "variants": {}}
for v in variants:
    t0 = time.time()
    try:
        S = GSolver(T, v); S.setup()
    except (KeyError, AssertionError) as e:
        print(v, "skipped", e); continue
    for _ in range(N):
        S.step()
    tab, ids = S.compiled()
    u, gaps, f = S.evaluate(tab, ids)
    tabm, idsm = S.mixture(); fm = S.edge_factors(tabm, idsm)
    cce = max(float(S.br_p(i, fm) - S.util_sum[i] / N) for i in (1, 2))
    rec = {"entries": int(S.bmask.sum()), "entries_p1": int(S.bmask[S.bowner == 1].sum()),
           "nashconv": float(sum(gaps.values())), "cce_gap": cce, "u1": u[1], "u2": u[2], "seconds": time.time() - t0}
    out["variants"][v] = rec
    print(v, json.dumps(rec), flush=True)
json.dump(out, open(f"frontier_{inst}_K{K}.json", "w"), indent=1)
