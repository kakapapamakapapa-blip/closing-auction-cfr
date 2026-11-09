"""Section 10.6 runner: general-sum game with a continuous market (auction_cont.TreeC)."""
import json, sys
import numpy as np
from auction_cont import TreeC, VARIANTS
from general_sum import GSolver, fill_rate
from cfr import Solver

K, N = int(sys.argv[1]), int(sys.argv[2])
T = TreeC(K)
out = {"K": K, "N": N, "nodes": int(T.M), "variants": {}}
for v in VARIANTS:
    S = GSolver(T, v); S.setup()
    curve = []
    for n in range(1, N + 1):
        S.step()
        if n in (100, 1000, 3000, 10000, N):
            tab, ids = S.compiled(); u, gaps, f = S.evaluate(tab, ids)
            curve.append({"n": n, "nashconv": sum(gaps.values())})
    tab, ids = S.compiled()
    u, gaps, f = S.evaluate(tab, ids)
    tabm, idsm = S.mixture()
    fm = S.edge_factors(tabm, idsm)
    cce = {i: float(S.br_p(i, fm) - S.util_sum[i] / N) for i in (1, 2)}
    gap_mix = {i: float(S.br_p(i, fm) - S.values_p(fm, S.U[i])[0]) for i in (1, 2)}
    # probability that the buyer takes liquidity in the continuous market, before and after the cutoff
    take_pre = take_post = 0.0
    pi = S.reaches(f)
    e1 = S.pe[1]
    for e in e1:
        h = T.parent[e]
        if T.nact[h] == 4 and T.eact[e] == 3:
            w = pi[1][h] * pi[2][h] * pi[0][h] * f[e]
            if T.keys[h]["full"][1] < K: take_pre += w
            else: take_post += w
    rec = {"entries_p1": int(S.bmask[S.bowner == 1].sum()), "u_table": u, "nashconv_table": sum(gaps.values()),
           "nashconv_mixture": sum(gap_mix.values()), "cce_gap": max(cce.values()),
           "take_pre": float(take_pre), "take_post": float(take_post), "curve": curve}
    out["variants"][v] = rec
    print(v, json.dumps({k: rec[k] for k in rec if k != "curve"}), [ (c["n"], round(c["nashconv"],5)) for c in curve], flush=True)
# iterate equivalence of phi_H and the full game in this instance (Assumption 2.2 fails here)
A_, B_ = GSolver(T, "full"), GSolver(T, "H"); A_.setup(); B_.setup()
mx = 0.0
for n in range(1, 2001):
    A_.step(); B_.step()
    if n % 100 == 0:
        sa, ia = A_.current(); sb, ib = B_.current(); d = T.dec
        mx = max(mx, float(np.abs(sa[ia[d]] - sb[ib[d]]).max()))
out["iterate_gap_H_vs_full"] = mx
print("iterate gap", mx)
json.dump(out, open(f"general_sum_cont_K{K}.json", "w"), indent=1)
