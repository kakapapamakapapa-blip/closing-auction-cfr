"""Deployment check for coarse layers: compiled table vs CFR mixture against random opponents (game Gamma_0)."""
import json, sys
import numpy as np
from auction import Tree
from cfr import Solver

K, N = int(sys.argv[1]), int(sys.argv[2])
T = Tree(K)
rng = np.random.default_rng(0)
out = {}
for v in sys.argv[3].split(","):
    S = Solver(T, v)
    for _ in range(N):
        S.step()
    tab, ids = S.compiled(); tabm, idsm = S.mixture()
    gaps = []
    for _ in range(20):
        X = rng.random((S.nf, S.fmask.shape[1])) * S.fmask; R2 = X / X.sum(1, keepdims=True)
        gaps.append(abs(S.value_pair(tab, ids, R2, S.fid) - S.value_pair(tabm, idsm, R2, S.fid)))
        gaps.append(abs(S.value_pair(R2, S.fid, tab, ids) - S.value_pair(R2, S.fid, tabm, idsm)))
    out[v] = {"max_gap": float(max(gaps)), "mean_gap": float(np.mean(gaps)),
              "nc_table": float(S.nashconv(tab, ids)[0]), "nc_mixture": float(S.nashconv(tabm, idsm)[0])}
    print(v, json.dumps(out[v]), flush=True)
json.dump(out, open(f"deploy_check_K{K}.json", "w"), indent=1)
