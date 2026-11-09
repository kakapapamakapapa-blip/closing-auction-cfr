"""Section 6 (instance C): the value of the right to cancel versus the value of remembering cancellations.

For each setting, run general-sum CFR (native payoffs) on
  (i)  the game with cancellation before the cutoff, full game and phi_H;
  (ii) the same game without cancellation (orders irrevocable once submitted), full game.
Report the compiled tables' payoffs, NashConv, the probability that the buyer cancels at least
once, fill statistics, the phi_H iterate gap, and table sizes.
"""
import json
import sys
import time
import numpy as np
from auction_c import TreeC
from general_sum import GSolver


def run(T, variant, N, track_iterates=False):
    S = GSolver(T, variant); S.setup()
    for _ in range(N):
        S.step()
    tab, ids = S.compiled()
    u, gaps, f = S.evaluate(tab, ids)
    stats = {
        "u1": u[1], "u2": u[2], "nashconv": sum(gaps.values()),
        "p_cancel": float(S.values_p(f, T.cycled)[0]),
        "p_fill": float(S.values_p(f, T.fill1.astype(float))[0]),
        "p_fill_at_100": float(S.values_p(f, ((T.fill1 > 0) & (T.price == 100)).astype(float))[0]),
        "p_fill_at_101": float(S.values_p(f, ((T.fill1 > 0) & (T.price == 101)).astype(float))[0]),
        "entries_p1": int(S.bmask[S.bowner == 1].sum()),
    }
    return S, stats


def iterate_gap(T, N):
    A, B = GSolver(T, "full"), GSolver(T, "H"); A.setup(); B.setup()
    mx = 0.0
    for n in range(1, N + 1):
        A.step(); B.step()
        if n % 50 == 0 or n == 1:
            sa, ia = A.current(); sb, ib = B.current(); d = T.dec
            mx = max(mx, float(np.abs(sa[ia[d]] - sb[ib[d]]).max()))
    return mx


if __name__ == "__main__":
    K, N = int(sys.argv[1]), int(sys.argv[2])
    grid = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [{"q": 0.4, "lam": 0.6}]
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    out = []
    for g in grid:
        t0 = time.time()
        Tc = TreeC(K, q=g["q"], lam=g["lam"], cancel=True)
        Tn = TreeC(K, q=g["q"], lam=g["lam"], cancel=False)
        _, full = run(Tc, "full", N)
        _, H = run(Tc, "H", N)
        _, nocan = run(Tn, "full", N)
        gap = iterate_gap(Tc, min(N, 500))
        rec = {"setting": g, "K": K, "N": N, "nodes": int(Tc.M), "cancel_full": full, "cancel_H": H,
               "no_cancel": nocan, "iterate_gap_H_vs_full": gap,
               "value_of_cancel_u1": full["u1"] - nocan["u1"], "seconds": time.time() - t0}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(out, open(f"cancel_value_K{K}{tag}.json", "w"), indent=1)
