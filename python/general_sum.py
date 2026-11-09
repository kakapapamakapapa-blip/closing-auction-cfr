"""Section 10.6: the toy auction as a general-sum game (u_1, u_2) instead of the auxiliary
zero-sum game U_1 = u_1 - u_2.

Player 1 (execution agent): u_1 = -(p - 100) q_1 - LAM (1 - q_1)^2.
Player 2 (liquidity provider): u_2 = (p - 100) q_2; it gains nothing from player 1's shortfall.
Reported per abstraction:
  * NashConv of the compiled tables in the general-sum game: sum_i [BR_i - u_i];
  * eps-CCE gap of the CFR iterates: max_i R^N_i / N, computed exactly;
  * the welfare and fill rate of the compiled tables.
"""
import json
import sys
import numpy as np
from auction import Tree, VARIANTS
from cfr import Solver, rm


class GSolver(Solver):
    def setup(self):
        self.U = {1: self.T.u1.astype(float), 2: self.T.u2.astype(float)}
        self.util_sum = {1: 0.0, 2: 0.0}

    def values_p(self, f, pay):
        T = self.T
        v = np.where(T.kind == 2, pay, 0.0)
        for d in range(T.maxd, 0, -1):
            idx = T.levels[d]
            np.add.at(v, T.parent[idx], f[idx] * v[idx])
        return v

    def step(self):
        T = self.T
        sigma = rm(self.R, self.bmask)
        f = self.edge_factors(sigma, self.bid)
        pi = self.reaches(f)
        for i in (1, 2):
            o = 2 if i == 1 else 1
            v = self.values_p(f, self.U[i])
            e, h = self.pe[i], self.pp[i]
            cfw = pi[o][h] * pi[0][h]
            np.add.at(self.R, (self.bid[h], T.eact[e]), cfw * (v[e] - v[h]))
            np.add.at(self.Sb, (self.bid[h], T.eact[e]), pi[i][h] * pi[0][h] * f[e])
            np.add.at(self.Sf, (self.fid[h], T.eact[e]), pi[i][h] * f[e])
            self.util_sum[i] += v[0]
        self.it += 1

    def br_p(self, i, f):
        T = self.T
        q = np.ones(T.M)
        for d in range(1, T.maxd + 1):
            idx = T.levels[d]
            q[idx] = q[T.parent[idx]] * np.where(T.ekind[idx] == i, 1.0, f[idx])
        val = np.where(T.kind == 2, self.U[i], 0.0)
        for d in range(T.maxd - 1, -1, -1):
            idx = T.levels[d]
            idx = idx[T.kind[idx] != 2]
            mine = idx[(T.kind[idx] == 1) & (T.player[idx] == i)]
            other = idx[~((T.kind[idx] == 1) & (T.player[idx] == i))]
            kids = T.levels[d + 1]
            kp = T.parent[kids]
            sel = np.isin(kp, other)
            np.add.at(val, kp[sel], f[kids[sel]] * val[kids[sel]])
            if len(mine):
                Q = np.zeros((self.nf, self.fmask.shape[1]))
                selm = np.isin(kp, mine)
                km = kids[selm]
                np.add.at(Q, (self.fid[kp[selm]], T.eact[km]), q[kp[selm]] * val[km])
                best = np.where(self.fmask, Q, -np.inf).argmax(1)
                val[mine] = val[T.child_of[mine, best[self.fid[mine]]]]
        return val[0]

    def evaluate(self, tab, ids):
        f = self.edge_factors(tab, ids)
        u = {i: float(self.values_p(f, self.U[i])[0]) for i in (1, 2)}
        gaps = {i: float(self.br_p(i, f) - u[i]) for i in (1, 2)}
        return u, gaps, f


def fill_rate(S, f):
    """Probability that player 1 is filled under edge probabilities f."""
    T = S.T
    q1 = np.zeros(T.M)
    q1[T.term] = (T.u1[T.term] > -1.4)  # unfilled => u_1 = -LAM = -1.5; filled => u_1 >= -1
    return float(S.values_p(f, q1.astype(float))[0])


if __name__ == "__main__":
    K, N = int(sys.argv[1]), int(sys.argv[2])
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else VARIANTS
    T = Tree(K)
    out = {"K": K, "N": N, "variants": {}}
    for v in variants:
        S = GSolver(T, v); S.setup()
        for _ in range(N):
            S.step()
        tab, ids = S.compiled()
        u, gaps, f = S.evaluate(tab, ids)
        tabm, idsm = S.mixture()
        fm = S.edge_factors(tabm, idsm)
        cce = {i: float(S.br_p(i, fm) - S.util_sum[i] / N) for i in (1, 2)}
        rec = {"u_table": u, "br_gap_table": gaps, "nashconv_table": sum(gaps.values()),
               "cce_gap": max(cce.values()), "regret_per_player": cce, "fill_rate_p1": fill_rate(S, f)}
        out["variants"][v] = rec
        print(v, json.dumps(rec), flush=True)
    json.dump(out, open(f"general_sum_K{K}.json", "w"), indent=1)
