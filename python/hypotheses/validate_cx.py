"""Check that TreeCX(slope=0.6, alloc='time') is the original TreeC, and time the solver."""
import sys, time, json
sys.path.insert(0, "/home/claude/exp/hypotheses"); sys.path.insert(0, "/home/claude/exp")
import numpy as np
from auction_c import TreeC
from cx import TreeCX, solve_and_measure

for K in (2, 3):
    t0 = time.time(); A = TreeC(K, q=0.3, lam=0.3); tA = time.time() - t0
    B = TreeCX(K, q=0.3, lam=0.3)
    same = (A.M == B.M and np.allclose(A.eprob, B.eprob, atol=1e-12) and np.array_equal(A.payoff, B.payoff)
            and np.array_equal(A.u1, B.u1) and all(a == b for a, b in zip(A.keys, B.keys)))
    P = TreeCX(K, q=0.3, lam=0.3, alloc="prorata"); F = TreeCX(K, q=0.3, lam=0.3, slope=0.0)
    print(f"K={K} nodes={A.M} build={tA:.2f}s identical_to_TreeC={same} prorata_nodes={P.M} "
          f"payoff_diff_prorata={float(np.abs(P.u1 - A.u1).max()):.3f} flat_eprob_diff={float(np.abs(F.eprob - A.eprob).max()):.3f}")
# reproduce cancel_dev_K2_grid.json, q=0.3 lam=0.3: private_value_at_c = 0.0269, p_cancel = 0.15
r = solve_and_measure(2, 0.3, 0.3, N=2000)
ref = json.load(open("/home/claude/exp/cancel_dev_K2_grid.json"))[0]
print("mine:", round(r["players"][1]["pv_u"], 6), round(r["players"][1]["p_cancel"], 6), "nashconv", r["nashconv"], f"{r['seconds']:.1f}s")
print("ref :", round(ref["private_value_at_c"], 6), round(ref["outcomes_cancel"]["p_cancel_p1"], 6), "nashconv", ref["nashconv_cancel"])
t0 = time.time(); r3 = solve_and_measure(3, 0.6, 0.6, N=200); print(f"K=3 N=200 {time.time()-t0:.1f}s nodes={r3['nodes']}")
