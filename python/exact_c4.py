"""Instance C at c = 4: phi_H against the full game under general-sum CFR (native payoffs).

Runs CFR on both abstractions in lockstep, records the largest difference between their current
strategies, and compares the compiled tables' payoffs and NashConv.
Usage: python3 exact_c4.py K N q lam  -> exact_C_K{K}.json
"""
import json
import sys
import time
import numpy as np
from auction_c import TreeC
from general_sum import GSolver

K, N, q, lam = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
t0 = time.time()
T = TreeC(K, q=q, lam=lam, cancel=True)
A, B = GSolver(T, "full"), GSolver(T, "H")
A.setup(); B.setup()
gap = 0.0
for n in range(1, N + 1):
    A.step(); B.step()
    if n == 1 or n % 25 == 0:
        sa, ia = A.current(); sb, ib = B.current(); d = T.dec
        gap = max(gap, float(np.abs(sa[ia[d]] - sb[ib[d]]).max()))
res = {"K": K, "N": N, "q": q, "lam": lam, "nodes": int(T.M), "iterate_gap": gap}
for name, S in (("full", A), ("H", B)):
    tab, ids = S.compiled()
    u, gaps, f = S.evaluate(tab, ids)
    res[name] = {"entries_p1": int(S.bmask[S.bowner == 1].sum()), "u1": u[1], "u2": u[2],
                 "nashconv": sum(gaps.values()), "p_cancel": float(S.values_p(f, T.cycled)[0])}
res["seconds"] = time.time() - t0
print(json.dumps(res), flush=True)
json.dump(res, open(f"exact_C_K{K}.json", "w"), indent=1)
