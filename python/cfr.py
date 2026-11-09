"""Vectorized vanilla CFR (simultaneous updates, regret matching) on bucket abstractions,
evaluated by exact best response in the perfect-recall game."""
import numpy as np


def rm(R, mask):
    P = np.where(mask, np.maximum(R, 0.0), 0.0)
    s = P.sum(1, keepdims=True)
    U = mask / mask.sum(1, keepdims=True)
    return np.where(s > 0, P / np.where(s > 0, s, 1), U)


def normalize(S, mask):
    s = S.sum(1, keepdims=True)
    U = mask / mask.sum(1, keepdims=True)
    return np.where(s > 0, S / np.where(s > 0, s, 1), U)


class Solver:
    def __init__(self, tree, variant, track_every=1):
        self.T = T = tree
        self.bid, self.nb, self.bmask, self.bowner = T.ids(variant)
        self.fid, self.nf, self.fmask, self.fowner = T.ids("full")
        self.R = np.zeros((self.nb, self.bmask.shape[1]))
        self.Sb = np.zeros((self.nb, self.bmask.shape[1]))     # compiled table: chance-weighted realization sums
        self.Sf = np.zeros((self.nf, self.fmask.shape[1]))     # Kuhn average per base infoset (= the mixture)
        self.track_every = track_every
        self.reach_log = []                  # per tracked iteration: pi_{-i}(I) per base infoset
        self.last_base_regret = None
        # decision edges per player, their parents
        self.pe = {i: T.pedges[i] for i in (1, 2)}
        self.pp = {i: T.parent[T.pedges[i]] for i in (1, 2)}
        self.it = 0
        self.t_of_f = np.zeros(self.nf, dtype=np.int64)
        for n in T.dec:
            self.t_of_f[self.fid[n]] = T.keys[n]["full"][1]

    def current(self):
        return rm(self.R, self.bmask), self.bid

    def value_pair(self, tab1, ids1, tab2, ids2):
        T = self.T
        f = np.ones(T.M)
        f[T.cedges] = T.eprob[T.cedges]
        for i, (tab, ids) in ((1, (tab1, ids1)), (2, (tab2, ids2))):
            e, h = self.pe[i], self.pp[i]
            f[e] = tab[ids[h], T.eact[e]]
        return self.values(f)[0]

    def edge_factors(self, tab, ids):
        """f[e] for all edges, decision edges from table rows ids[parent]."""
        T = self.T
        f = np.ones(T.M)
        f[T.cedges] = T.eprob[T.cedges]
        for i in (1, 2):
            e, h = self.pe[i], self.pp[i]
            f[e] = tab[ids[h], T.eact[e]]
        return f

    def reaches(self, f):
        T = self.T
        pi = {1: np.ones(T.M), 2: np.ones(T.M), 0: np.ones(T.M)}
        for d in range(1, T.maxd + 1):
            idx = T.levels[d]
            par = T.parent[idx]
            ek = T.ekind[idx]
            for k in (0, 1, 2):
                pi[k][idx] = pi[k][par] * np.where(ek == k, f[idx], 1.0)
        return pi

    def values(self, f):
        T = self.T
        v = np.where(T.kind == 2, T.payoff, 0.0).astype(float)
        for d in range(T.maxd, 0, -1):
            idx = T.levels[d]
            np.add.at(v, T.parent[idx], f[idx] * v[idx])
        return v

    def step(self):
        T = self.T
        sigma = rm(self.R, self.bmask)
        f = self.edge_factors(sigma, self.bid)
        pi = self.reaches(f)
        v = self.values(f)
        track = (self.it % self.track_every == 0)
        base_r = np.zeros((self.nf, self.fmask.shape[1]))
        for i in (1, 2):
            sg = 1.0 if i == 1 else -1.0
            o = 2 if i == 1 else 1
            e, h = self.pe[i], self.pp[i]
            cfw = pi[o][h] * pi[0][h]
            r = cfw * sg * (v[e] - v[h])
            np.add.at(self.R, (self.bid[h], T.eact[e]), r)
            np.add.at(base_r, (self.fid[h], T.eact[e]), r)
            np.add.at(self.Sb, (self.bid[h], T.eact[e]), pi[i][h] * pi[0][h] * f[e])
            np.add.at(self.Sf, (self.fid[h], T.eact[e]), pi[i][h] * f[e])
        if track:
            rr = np.zeros(self.nf)
            for i in (1, 2):
                o = 2 if i == 1 else 1
                hs = T.dec[T.player[T.dec] == i]
                np.add.at(rr, self.fid[hs], pi[o][hs] * pi[0][hs])
            self.reach_log.append(rr)
        self.last_base_regret = base_r
        self.it += 1

    # ---------- evaluation ----------
    def br(self, i, f):
        """Value of player i's perfect-recall best response against the fixed edge probs f."""
        T = self.T
        sg = 1.0 if i == 1 else -1.0
        q = np.ones(T.M)
        for d in range(1, T.maxd + 1):
            idx = T.levels[d]
            par = T.parent[idx]
            q[idx] = q[par] * np.where(T.ekind[idx] == i, 1.0, f[idx])
        val = np.where(T.kind == 2, sg * T.payoff, 0.0).astype(float)
        for d in range(T.maxd - 1, -1, -1):
            idx = T.levels[d]
            idx = idx[T.kind[idx] != 2]
            mine = idx[(T.kind[idx] == 1) & (T.player[idx] == i)]
            other = idx[~((T.kind[idx] == 1) & (T.player[idx] == i))]
            # others: expectation
            ch = T.child_of  # not used for chance; use parent arrays
            kids = T.levels[d + 1]
            kp = T.parent[kids]
            sel = np.isin(kp, other)
            np.add.at(val, kp[sel], f[kids[sel]] * val[kids[sel]])
            if len(mine):
                Q = np.zeros((self.nf, self.fmask.shape[1]))
                selm = np.isin(kp, mine)
                km = kids[selm]
                np.add.at(Q, (self.fid[kp[selm]], T.eact[km]), q[kp[selm]] * val[km])
                Qm = np.where(self.fmask, Q, -np.inf)
                best = Qm.argmax(1)
                val[mine] = val[T.child_of[mine, best[self.fid[mine]]]]
        return val[0]

    def nashconv(self, tab, ids):
        f = self.edge_factors(tab, ids)
        return self.br(1, f) + self.br(2, f), f

    def compiled(self):
        return normalize(self.Sb, self.bmask), self.bid

    def mixture(self):
        return normalize(self.Sf, self.fmask), self.fid

    def value(self, f):
        return self.values(f)[0]

    # ---------- diagnostics ----------
    def drift(self, phase=None):
        """Reach-weighted mean and max over buckets of the drift coefficient nu_J,
        computed from the tracked iterations; per player. phase in {None, 'pre', 'post'}."""
        T = self.T
        L = np.array(self.reach_log)  # n_it x nf
        # bucket of each base infoset
        b_of_f = -np.ones(self.nf, dtype=np.int64)
        for n in T.dec:
            b_of_f[self.fid[n]] = self.bid[n]
        out = {}
        for i in (1, 2):
            fs = np.where(self.fowner == i)[0]
            if phase is not None:
                tf = self.t_of_f[fs]
                fs = fs[(tf < T.K) if phase == "pre" else (tf >= T.K)]
            bs = np.unique(b_of_f[fs])
            num_tot, den_tot, mx = 0.0, 0.0, 0.0
            for b in bs:
                mem = fs[b_of_f[fs] == b]
                if len(mem) < 2:
                    continue
                X = L[:, mem]
                piJ = X.sum(1)
                if piJ.sum() <= 1e-12:
                    continue
                rho_bar = X.sum(0) / piJ.sum()
                rho = np.where(piJ[:, None] > 0, X / np.where(piJ[:, None] > 0, piJ[:, None], 1), rho_bar)
                nu = (piJ * np.abs(rho - rho_bar).sum(1)).sum() / piJ.sum()
                num_tot += nu * piJ.sum(); den_tot += piJ.sum()
                mx = max(mx, nu)
            out[i] = (num_tot / den_tot if den_tot > 0 else 0.0, mx)
        return out

    def member_spread(self):
        """max over buckets of the spread of members' instantaneous regret vectors (last iteration)."""
        T = self.T
        b_of_f = -np.ones(self.nf, dtype=np.int64)
        for n in T.dec:
            b_of_f[self.fid[n]] = self.bid[n]
        r = self.last_base_regret
        mx = 0.0
        for b in np.unique(b_of_f):
            mem = np.where(b_of_f == b)[0]
            if len(mem) < 2:
                continue
            sub = r[mem][:, self.bmask[b]]
            mx = max(mx, float((sub.max(0) - sub.min(0)).max()))
        return mx
