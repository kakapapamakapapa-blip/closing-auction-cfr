import json, sys, time
import numpy as np
from auction import Tree, VARIANTS
from cfr import Solver, rm

K = int(sys.argv[1]); N = int(sys.argv[2]); TRACK = int(sys.argv[3]) if len(sys.argv) > 3 else 1
checkpoints = sorted(set([c for c in [10, 30, 100, 300, 1000, 3000, 10000, 30000] if c <= N] + [N]))
T = Tree(K)
rng = np.random.default_rng(0)
out = {"K": K, "N": N, "nodes": int(T.M), "terminals": int(len(T.term)), "variants": {}}
full_ref = Solver(T, "full")
nf, fmask, fowner = full_ref.nf, full_ref.fmask, full_ref.fowner
# random opponent strategies on base information sets (for payoff-equivalence tests)
rand_tabs = []
for _ in range(20):
    X = rng.random((nf, 3)) * fmask
    rand_tabs.append(X / X.sum(1, keepdims=True))

solvers = {}
for v in VARIANTS:
    S = Solver(T, v, track_every=TRACK)
    solvers[v] = S
    rec = {"buckets": {str(i): int((S.bowner == i).sum()) for i in (1, 2)},
           "entries": {str(i): int(S.bmask[S.bowner == i].sum()) for i in (1, 2)},
           "base_infosets": {str(i): int((fowner == i).sum()) for i in (1, 2)},
           "base_entries": {str(i): int(fmask[fowner == i].sum()) for i in (1, 2)},
           "curve": []}
    t0 = time.time()
    for n in range(1, N + 1):
        S.step()
        if n in checkpoints:
            tab, ids = S.compiled(); nc_t, _ = S.nashconv(tab, ids)
            tabm, idsm = S.mixture(); nc_m, fm = S.nashconv(tabm, idsm)
            rec["curve"].append({"n": n, "nc_table": float(nc_t), "nc_mixture": float(nc_m)})
    rec["seconds"] = time.time() - t0
    tabm, idsm = S.mixture(); _, fm = S.nashconv(tabm, idsm)
    rec["value_mixture"] = float(S.value(fm))
    rec["drift_pre"] = {str(i): S.drift("pre")[i] for i in (1, 2)}
    rec["drift_post"] = {str(i): S.drift("post")[i] for i in (1, 2)}
    rec["member_spread"] = S.member_spread()
    # payoff equivalence of compiled table vs mixture against random opponents (each side)
    tab, ids = S.compiled()
    gaps = []
    for R2 in rand_tabs:
        g1 = S.value_pair(tab, ids, R2, S.fid) - S.value_pair(tabm, idsm, R2, S.fid)
        g2 = S.value_pair(R2, S.fid, tab, ids) - S.value_pair(R2, S.fid, tabm, idsm)
        gaps += [abs(g1), abs(g2)]
    rec["deploy_gap_random_opponents"] = float(max(gaps))
    out["variants"][v] = rec
    print(v, json.dumps({k: rec[k] for k in ["buckets", "entries", "seconds", "value_mixture", "drift_pre", "drift_post", "member_spread", "deploy_gap_random_opponents"]}), flush=True)
    print("   ", [(c["n"], round(c["nc_table"], 5), round(c["nc_mixture"], 5)) for c in rec["curve"]], flush=True)

# iterate equivalence: CFR on phi_H vs CFR on full game, current strategies, fresh runs
A, B = Solver(T, "full"), Solver(T, "H")
mx = 0.0
for n in range(1, min(N, 2000) + 1):
    A.step(); B.step()
    if n % 100 == 0 or n == 1:
        sa, ia = A.current(); sb, ib = B.current()
        d = T.dec
        mx = max(mx, float(np.abs(sa[ia[d]] - sb[ib[d]]).max()))
out["iterate_gap_H_vs_full"] = mx
print("iterate gap H vs full (max |sigma diff| over decision nodes):", mx)
json.dump(out, open(f"results_K{K}.json", "w"), indent=1)
