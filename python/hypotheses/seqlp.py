"""Exact value of a two-player zero-sum game tree via the sequence-form LP (Koller, Megiddo and
von Stengel 1996), for the trees of auction.py / market.py / auction_c.py.

Player 1 maximises sum_z chance(z) * payoff(z). Information sets are the perfect-recall 'full' keys.
  max_{x, q} q_0   s.t.   F^T q - A^T x <= 0,   E x = e_0,   x >= 0,   q free,
where x is player 1's realisation plan, E/F the sequence constraints of players 1/2 and A the
sequence-form payoff matrix. ban1/ban2 in game_value: predicates (node, action) -> True forbid that
action for that player (its sequence is fixed at 0); used for mechanism checks.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack


def sequence_form(T, payoff=None):
    fid, nf, fmask, fowner = T.ids("full")
    pay = T.payoff if payoff is None else payoff
    M = T.M
    s = {1: np.zeros(M, dtype=np.int64), 2: np.zeros(M, dtype=np.int64)}
    ch = np.ones(M)
    seqid = {1: {}, 2: {}}
    for d in range(1, T.maxd + 1):
        idx = T.levels[d]
        par = T.parent[idx]
        s[1][idx] = s[1][par]; s[2][idx] = s[2][par]; ch[idx] = ch[par]
        for e, p in zip(idx.tolist(), par.tolist()):
            k = int(T.ekind[e])
            if k == 0:
                ch[e] *= T.eprob[e]
            else:
                key = (int(fid[p]), int(T.eact[e]))
                if key not in seqid[k]:
                    seqid[k][key] = len(seqid[k]) + 1
                s[k][e] = seqid[k][key]
    # parent sequence of every information set (perfect recall: unique)
    parent_seq = {1: {}, 2: {}}
    for n in T.dec.tolist():
        i, I = int(T.player[n]), int(fid[n])
        ps = int(s[i][n])
        assert parent_seq[i].setdefault(I, ps) == ps, "imperfect recall in 'full' keys"
    cons = {}
    for i in (1, 2):
        rows, cols, vals = [0], [0], [1.0]
        for r, (I, ps) in enumerate(sorted(parent_seq[i].items()), start=1):
            for a in range(int(fmask[I].sum())):
                rows.append(r); cols.append(seqid[i][(I, a)]); vals.append(1.0)
            rows.append(r); cols.append(ps); vals.append(-1.0)
        cons[i] = coo_matrix((vals, (rows, cols)), shape=(1 + len(parent_seq[i]), 1 + len(seqid[i]))).tocsr()
    z = T.term
    A = coo_matrix((ch[z] * pay[z], (s[1][z], s[2][z])), shape=(1 + len(seqid[1]), 1 + len(seqid[2]))).tocsr()
    return {"E": cons[1], "F": cons[2], "A": A, "seqid": seqid, "fid": fid, "s": s, "ch": ch}


def lp_value(E, F, A, banned=()):
    """max_x min_y x^T A y over realisation plans (E x = e0, F y = e0); sequences in `banned`
    (of the maximiser) are fixed at 0."""
    n1, m2 = E.shape[1], F.shape[0]
    c = np.zeros(n1 + m2); c[n1] = -1.0
    A_ub = hstack([-A.T, F.T]).tocsr()
    A_eq = hstack([E, coo_matrix((E.shape[0], m2))]).tocsr()
    b_eq = np.zeros(E.shape[0]); b_eq[0] = 1.0
    bounds = [(0, None)] * n1 + [(None, None)] * m2
    for j in banned:
        bounds[j] = (0, 0)
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs", options={"primal_feasibility_tolerance": 1e-10,
                                           "dual_feasibility_tolerance": 1e-10})
    assert res.status == 0, res.message
    return float(-res.fun), res.x[:n1]


def banned_seqs(T, sf, i, pred):
    """Sequence ids of player i for (node, action) pairs with pred(node, action) True."""
    out = set()
    for n in T.dec.tolist():
        if int(T.player[n]) == i:
            for a in range(int(T.nact[n])):
                if pred(n, a):
                    out.add(sf["seqid"][i][(int(sf["fid"][n]), a)])
    return sorted(out)


def game_value(T, payoff=None, ban1=None, ban2=None, sf=None):
    """Value for player 1. ban1 / ban2: predicates (node, action) -> True forbid that action for
    player 1 / player 2 (the player-2 case solves the role-swapped LP)."""
    sf = sf or sequence_form(T, payoff)
    E, F, A = sf["E"], sf["F"], sf["A"]
    if ban2 is not None:
        assert ban1 is None
        v, _ = lp_value(F, E, (-A.T).tocsr(), banned_seqs(T, sf, 2, ban2))
        return -v, None
    banned = banned_seqs(T, sf, 1, ban1) if ban1 is not None else ()
    return lp_value(E, F, A, banned)


def value_both_sides(T, payoff=None):
    """Value from player 1's LP and from player 2's LP; they must agree."""
    sf = sequence_form(T, payoff)
    v1, _ = lp_value(sf["E"], sf["F"], sf["A"])
    v2, _ = lp_value(sf["F"], sf["E"], (-sf["A"].T).tocsr())
    return v1, -v2
