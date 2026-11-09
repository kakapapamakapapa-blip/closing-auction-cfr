"""Section 10.2: measured alignment defect against Lemma 4.1, Lemma 4.4, Theorem 4.5 and Corollary 5.3.

Two passes of deterministic CFR per abstraction. Pass 1 accumulates the reach of every
base information set; pass 2 replays the same iterates and evaluates the certificate
r^n_I = l_I rho^n_J + eta^n_I with l_I = time-averaged reach share and rho^n_J = r^n_J.
The resulting E_cert is an upper bound on the alignment defect E_J (an infimum).
"""
import json
import sys
import numpy as np
from auction import Tree, VARIANTS
from cfr import Solver


class CertSolver(Solver):
    def setup(self, rho_bar=None):
        T = self.T
        self.b_of_f = -np.ones(self.nf, dtype=np.int64)
        for n in T.dec:
            self.b_of_f[self.fid[n]] = self.bid[n]
        self.cum_reach = np.zeros(self.nf)
        self.rho_bar = rho_bar
        self.Rf = np.zeros((self.nf, self.fmask.shape[1]))          # cumulative base regrets
        self.E = np.zeros(self.nb)                # certificate defect per bucket
        self.G = np.zeros(self.nb)                # max |r^n_J(a)|
        self.cor53 = np.zeros(self.nb)            # 2 sum_n pi^n(J) eps^n_J
        self.util_sum = {1: 0.0, 2: 0.0}

    def step(self):
        T = self.T
        from cfr import rm
        sigma = rm(self.R, self.bmask)
        f = self.edge_factors(sigma, self.bid)
        pi = self.reaches(f)
        v = self.values(f)
        base_r = np.zeros((self.nf, self.fmask.shape[1]))
        Q = np.zeros((self.nf, self.fmask.shape[1]))
        rr = np.zeros(self.nf)
        for i in (1, 2):
            sg = 1.0 if i == 1 else -1.0
            o = 2 if i == 1 else 1
            e, h = self.pe[i], self.pp[i]
            cfw = pi[o][h] * pi[0][h]
            r = cfw * sg * (v[e] - v[h])
            np.add.at(self.R, (self.bid[h], T.eact[e]), r)
            np.add.at(base_r, (self.fid[h], T.eact[e]), r)
            np.add.at(Q, (self.fid[h], T.eact[e]), cfw * sg * v[e])
            np.add.at(self.Sb, (self.bid[h], T.eact[e]), pi[i][h] * pi[0][h] * f[e])
            np.add.at(self.Sf, (self.fid[h], T.eact[e]), pi[i][h] * f[e])
            hs = T.dec[T.player[T.dec] == i]
            np.add.at(rr, self.fid[hs], pi[o][hs] * pi[0][hs])
            self.util_sum[i] += sg * v[0]
        # rr counted each decision node once; Q and base_r are sums over histories, as in r^n_I
        self.cum_reach += rr
        self.Rf += base_r
        rJ = np.zeros((self.nb, self.bmask.shape[1]))
        np.add.at(rJ, self.b_of_f, base_r)
        self.G = np.maximum(self.G, np.abs(rJ).max(1))
        if self.rho_bar is not None:
            eta = base_r - self.rho_bar[:, None] * rJ[self.b_of_f]
            np.add.at(self.E, self.b_of_f, np.abs(eta).max(1))
            # Corollary 5.3: eps^n_J = max_a spread over members of vbar(I, a) = Q_I(a) / pi^n(I)
            live = rr > 1e-300
            vbar = np.where(live[:, None], Q / np.where(live, rr, 1.0)[:, None], np.nan)
            vbar = np.where(self.fmask, vbar, np.nan)
            hi = np.full((self.nb, self.bmask.shape[1]), -np.inf); lo = np.full((self.nb, self.bmask.shape[1]), np.inf)
            np.maximum.at(hi, self.b_of_f, np.where(np.isnan(vbar), -np.inf, vbar))
            np.minimum.at(lo, self.b_of_f, np.where(np.isnan(vbar), np.inf, vbar))
            spread = np.where(np.isfinite(hi) & np.isfinite(lo), hi - lo, 0.0).max(1)
            piJ = np.zeros(self.nb); np.add.at(piJ, self.b_of_f, rr)
            self.cor53 += 2 * piJ * spread
        self.it += 1


def run(T, v, N):
    S1 = CertSolver(T, v); S1.setup()
    for _ in range(N):
        S1.step()
    piJ = np.zeros(S1.nb); np.add.at(piJ, S1.b_of_f, S1.cum_reach)
    rho_bar = np.where(piJ[S1.b_of_f] > 0, S1.cum_reach / np.where(piJ[S1.b_of_f] > 0, piJ[S1.b_of_f], 1), 0.0)
    S = CertSolver(T, v); S.setup(rho_bar)
    for _ in range(N):
        S.step()
    assert np.allclose(S.R, S1.R), "replay diverged"
    tabm, idsm = S.mixture()
    fm = S.edge_factors(tabm, idsm)
    out = {}
    nA = S.bmask.sum(1)
    for i in (1, 2):
        own_b = S.bowner == i
        own_f = S.fowner == i
        actual = S.br(i, fm) - S.util_sum[i] / N          # R^N_i / N, exact
        sum_RI = np.maximum(S.Rf[own_f].max(1), 0).sum() / N
        RJ = np.where(S.bmask, S.R, -np.inf).max(1)
        cert = (np.maximum(RJ[own_b], 0).sum() + 2 * S.E[own_b].sum()) / N
        apriori = (S.G[own_b] * np.sqrt(nA[own_b] * N)).sum() / N + 2 * S.E[own_b].sum() / N
        pre = own_b & np.isin(np.arange(S.nb), np.unique(S.b_of_f[(S.t_of_f < T.K)]))
        post = own_b & ~pre
        out[str(i)] = {
            "actual": float(actual), "sum_pos_base": float(sum_RI),
            "certificate": float(cert), "apriori": float(apriori),
            "cfr_term": float(np.maximum(RJ[own_b], 0).sum() / N),
            "structural": float(2 * S.E[own_b].sum() / N),
            "E_pre": float(S.E[pre].sum()), "cor53_pre": float(S.cor53[pre].sum()),
            "E_post": float(S.E[post].sum()), "cor53_post": float(S.cor53[post].sum()),
            "cor53_violations_pre": int((S.E[pre] > S.cor53[pre] + 1e-9).sum()),
            "cor53_violations_post": int((S.E[post] > S.cor53[post] + 1e-9).sum()),
        }
    return out


if __name__ == "__main__":
    K, N = int(sys.argv[1]), int(sys.argv[2])
    T = Tree(K)
    res = {"K": K, "N": N, "variants": {}}
    for v in VARIANTS:
        res["variants"][v] = run(T, v, N)
        print(v, json.dumps(res["variants"][v]["1"]), flush=True)
    json.dump(res, open(f"certificate_K{K}.json", "w"), indent=1)
