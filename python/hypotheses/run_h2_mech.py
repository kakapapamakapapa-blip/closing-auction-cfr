"""H2 mechanism check: is cancelling replaceable by delaying submission?

Against the opponent's CFR strategy (instance C, persist 0.7, N = 2000), player 1's perfect-recall
best-response values when
  V_all  : every action allowed;
  V_nc   : cancelling forbidden;
  V_late : cancelling forbidden and no MOC submission before the last pre-cutoff period t = K - 1.
Replication argument: with a hidden book and pro-rata allocation the final book (hence fills and the
imbalance indicator) does not depend on when an order was submitted, so any cancel strategy is
replicated by submitting at t = K - 1 exactly when the cancel strategy would still hold an order:
V_all = V_nc = V_late. Under time priority delaying costs priority, so V_late < V_nc <= V_all.
Usage: python3 run_h2_mech.py -> out/h2_mech.json
"""
import json
import sys
import time
sys.path.insert(0, "/home/claude/exp/hypotheses"); sys.path.insert(0, "/home/claude/exp")
import numpy as np
from cx import TreeCX, train, cancel_stats
from cancel_dev import br_masked

jobs = [(2, q, lam) for q in (0.3, 0.6) for lam in (0.3, 0.45, 0.6, 0.75)] + [(3, 0.6, 0.3), (3, 0.6, 0.6)]
out, t00 = [], time.time()
for K, q, lam in jobs:
    for alloc in ("time", "prorata"):
        t0 = time.time()
        T = TreeCX(K, q=q, lam=lam, persist=0.7, alloc=alloc)
        S = train(T, "full", 2000)
        tab, ids = S.compiled()
        u, gaps, f = S.evaluate(tab, ids)
        cs = cancel_stats(T, S, f)
        nc = np.ones_like(S.fmask); late = np.ones_like(S.fmask)
        for n in T.dec:
            if T.player[n] != 1:
                continue
            t, order = T.keys[n]["H"][1], T.keys[n]["H"][3]
            if order is not None and t < K and T.nact[n] == 2:
                nc[S.fid[n], 1] = False; late[S.fid[n], 1] = False
            if order is None and t < K - 1:
                late[S.fid[n], 1] = False
        V_all = br_masked(S, 1, f, np.ones_like(S.fmask))
        V_nc, V_late = br_masked(S, 1, f, nc), br_masked(S, 1, f, late)
        rec = {"K": K, "q": q, "lam": lam, "alloc": alloc, "nashconv": float(sum(gaps.values())), "u1": u[1],
               "V_all": V_all, "V_nc": V_nc, "V_late": V_late, "pv_cancel": V_all - V_nc,
               "value_of_early_submission_without_cancel": V_nc - V_late,
               "p_cancel1": cs[1]["p_cancel"], "submit_by_t1": cs[1]["submit_by_t"], "seconds": time.time() - t0}
        out.append(rec)
        print(f"K={K} q={q} lam={lam} {alloc:7s}: V_all-V_nc={V_all - V_nc:.2e} V_nc-V_late={V_nc - V_late:.2e} "
              f"p_cancel={cs[1]['p_cancel']:.3f} submit_by_t={ {k: round(v, 3) for k, v in cs[1]['submit_by_t'].items()} } "
              f"nc={rec['nashconv']:.1e} {rec['seconds']:.1f}s", flush=True)
        json.dump(out, open("/home/claude/exp/hypotheses/out/h2_mech.json", "w"), indent=1)
print(f"total {time.time() - t00:.1f}s")
