"""Worst-case CFR term of Theorem 3.7: G_J <= osc(U_1) * sum over histories h in J of the chance reach of h
(opponent reach <= 1), for player 1 in instance A, game Gamma_0, K = 3, N = 10^4."""
import json, numpy as np
from auction import Tree
from cfr import Solver
K, N = 3, 10000
T = Tree(K)
osc = float(np.ptp(T.payoff[T.term]))
pc = np.ones(T.M)
for d in range(1, T.maxd + 1):
    idx = T.levels[d]
    pc[idx] = pc[T.parent[idx]] * np.where(T.ekind[idx] == 0, T.eprob[idx], 1.0)
cert = json.load(open("certificate_K3.json"))["variants"]
out = {}
for v in ["full", "H", "H_s0", "H_chi", "A"]:
    S = Solver(T, v)
    G = np.zeros(S.nb)
    h1 = T.dec[T.player[T.dec] == 1]
    np.add.at(G, S.bid[h1], osc * pc[h1])
    nA = S.bmask.sum(1)
    own = S.bowner == 1
    wc = float((G[own] * np.sqrt(nA[own])).sum() / np.sqrt(N))
    c = cert[v]["1"]
    out[v] = {"actual": c["actual"], "lemma33": c["sum_pos_base"], "certificate": c["certificate"],
              "thm37_measured_G": c["apriori"], "cfr_term_worstcase": wc,
              "thm37_worstcase_G": wc + c["structural"], "structural": c["structural"]}
    print(v, {k: round(x, 4) for k, x in out[v].items()})
json.dump(out, open("apriori_K3.json", "w"), indent=1)
