"""Instance B in the auxiliary zero-sum game Gamma_0: NashConv of compiled tables and mixtures.

Usage: python3 gamma0_cont.py K N   -> gamma0_cont_K{K}.json
"""
import json
import sys
from auction_cont import TreeC, VARIANTS
from cfr import Solver

K, N = int(sys.argv[1]), int(sys.argv[2])
T = TreeC(K)
out = {"K": K, "N": N, "nodes": int(T.M), "variants": {}}
for v in VARIANTS:
    S = Solver(T, v)
    for _ in range(N):
        S.step()
    tab, ids = S.compiled(); tabm, idsm = S.mixture()
    rec = {"entries_p1": int(S.bmask[S.bowner == 1].sum()),
           "nashconv_table": float(S.nashconv(tab, ids)[0]), "nashconv_mixture": float(S.nashconv(tabm, idsm)[0])}
    out["variants"][v] = rec
    print(v, json.dumps(rec), flush=True)
json.dump(out, open(f"gamma0_cont_K{K}.json", "w"), indent=1)
