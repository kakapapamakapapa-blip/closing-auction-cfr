"""Section 10.5: the model-game reset (Theorem 8.6) on the toy auction.

phi_H + drop s0 ("H_s0"). In the true game G the law of the exogenous
interest X at the cutoff depends on s0, so the merge drifts after the cutoff. In the model game
G~ the law of X uses the stationary mean 1/2 in place of s0, so from period 1 on the forgotten signal
is irrelevant and the merge is exact (Proposition 8.7); at period 0 the merge still loses the current
state of the signal chain. Theorem 8.6 bounds what is lost by training in G~ and
playing in G: |u_i^G(s) - u_i^G~(s)| <= osc(u_i) * delta for every profile s, where delta is the
largest total-variation distance between the laws of X in G and G~.
"""
import json
import sys
import numpy as np
from auction import Tree
from certificate import CertSolver


def train(T, v, N):
    S1 = CertSolver(T, v); S1.setup()
    for _ in range(N):
        S1.step()
    piJ = np.zeros(S1.nb); np.add.at(piJ, S1.b_of_f, S1.cum_reach)
    rho = np.where(piJ[S1.b_of_f] > 0, S1.cum_reach / np.where(piJ[S1.b_of_f] > 0, piJ[S1.b_of_f], 1), 0.0)
    S = CertSolver(T, v); S.setup(rho)
    for _ in range(N):
        S.step()
    return S


if __name__ == "__main__":
    K, N = int(sys.argv[1]), int(sys.argv[2])
    V = sys.argv[3] if len(sys.argv) > 3 else "H_s0"
    G, Gm = Tree(K), Tree(K, model=True)
    delta = 0.0
    # delta: max over cutoff chance nodes of TV between the laws of X in G and G~
    par = G.parent[G.cedges]
    for node in np.unique(par):
        kids = G.cedges[par == node]
        tv = 0.5 * np.abs(G.eprob[kids] - Gm.eprob[kids]).sum()
        delta = max(delta, float(tv))
    osc = float(np.ptp(G.payoff[G.term]))
    out = {"K": K, "N": N, "delta": delta, "osc_U1": osc, "runs": {}}
    for name, T in (("trained_in_G", G), ("trained_in_model", Gm)):
        S = train(T, V, N)
        tab, ids = S.compiled()
        ev_G = CertSolver(G, V); ev_M = CertSolver(Gm, V)
        nc_G = float(ev_G.nashconv(tab, ids)[0])
        nc_M = float(ev_M.nashconv(tab, ids)[0])
        t_of_b = np.zeros(S.nb, dtype=np.int64); t_of_b[S.b_of_f] = S.t_of_f
        struct = {}
        for i in (1, 2):
            own = S.bowner == i
            struct[str(i)] = {"t0": float(2 * S.E[own & (t_of_b == 0)].sum() / N),
                              "pre_t_ge_1": float(2 * S.E[own & (t_of_b >= 1) & (t_of_b < K)].sum() / N),
                              "post": float(2 * S.E[own & (t_of_b >= K)].sum() / N)}
        out["runs"][name] = {"nashconv_in_G": nc_G, "nashconv_in_model": nc_M, "structural_in_training_game": struct}
        print(name, json.dumps(out["runs"][name]), flush=True)
    out["bound_nashconv_G_from_model"] = out["runs"]["trained_in_model"]["nashconv_in_model"] + 4 * osc * delta
    print("delta", delta, "osc", osc, "bound", out["bound_nashconv_G_from_model"])
    out["variant"] = V
    json.dump(out, open(f"model_check_K{K}_{V}.json", "w"), indent=1)
