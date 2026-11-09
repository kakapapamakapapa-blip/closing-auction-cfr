"""Section 6.9: freeze-before-disclose versus disclose-before-freeze.

For instance A (optionally with the continuous market of instance B) and the auxiliary zero-sum
game Gamma_0, compare the regime "hidden" (on-close interest is public only from the cutoff on,
when it can no longer be withdrawn) with the regime "disclose" (the net direction of resting
on-close interest is public before the cutoff, while it can still be cancelled). For each regime:

  * the game value of Gamma_0 with and without the right to cancel before the cutoff, as an
    interval [v_lo, v_hi] certified by exact best responses to the CFR strategies;
  * whether the exact layer phi_H is still exact: NashConv of phi_H against the full game and the
    largest difference between their CFR iterates;
  * the expected number of pre-cutoff cancellations of each player under the CFR strategies.
"""
import json
import sys
import numpy as np
from market import Market, Params
from cfr import Solver


def value_interval(S):
    tab, ids = S.compiled()
    f = S.edge_factors(tab, ids)
    v_hi = S.br(1, f)
    v_lo = -S.br(2, f)
    return float(v_lo), float(v_hi), f


def cancellations(S, f):
    T = S.T
    pi = S.reaches(f)
    out = {}
    for i in (1, 2):
        e = S.pe[i]
        e = e[T.elabel[e] == "cancel"]
        h = T.parent[e]
        out[str(i)] = float((pi[1][h] * pi[2][h] * pi[0][h] * f[e]).sum())
    return out


def iterate_gap(T, n_iter):
    A, B = Solver(T, "full"), Solver(T, "H")
    mx = 0.0
    for n in range(1, n_iter + 1):
        A.step(); B.step()
        if n % 50 == 0:
            sa, ia = A.current(); sb, ib = B.current(); d = T.dec
            mx = max(mx, float(np.abs(sa[ia[d]] - sb[ib[d]]).max()))
    return mx


if __name__ == "__main__":
    K, N = int(sys.argv[1]), int(sys.argv[2])
    cont = len(sys.argv) > 3 and sys.argv[3] == "cont"
    out = {"K": K, "N": N, "cont": cont, "runs": []}
    for disclose in (False, True):
        for cancel in (True, False):
            prm = Params(K=K, cont=cont, disclose=disclose, cancel=cancel)
            T = Market(prm)
            rec = {"regime": "disclose" if disclose else "hidden", "cancel": cancel, "nodes": int(T.M)}
            for v in ("full", "H"):
                S = Solver(T, v)
                for _ in range(N):
                    S.step()
                v_lo, v_hi, f = value_interval(S)
                rec[v] = {"v_lo": v_lo, "v_hi": v_hi, "nashconv": v_hi - v_lo,
                          "entries_p1": int(S.bmask[S.bowner == 1].sum()),
                          "cancellations": cancellations(S, f) if cancel else {"1": 0.0, "2": 0.0}}
            rec["iterate_gap_H_vs_full"] = iterate_gap(T, min(N, 1000)) if cancel else None
            out["runs"].append(rec)
            print(json.dumps(rec), flush=True)
    tag = "_cont" if cont else ""
    json.dump(out, open(f"regimes_K{K}{tag}.json", "w"), indent=1)
