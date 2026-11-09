"""Staged endgame re-solving on instance A in the zero-sum game Gamma_0.

A coarse blueprint is trained on the whole game. Closer to the close, the remaining game is re-solved in
stages with increasingly exact solvers; every stage freezes all decisions before its start period at
the strategy built so far and is used only until the next stage starts:

    periods [0, t1)      blueprint: vanilla CFR on a coarse abstraction (default "A": drop s0 and chi)
    periods [t1, t2)     stage 1: vanilla CFR on phi_H, re-solving periods >= t1
    periods [t2, t3)     stage 2: external-sampling MCCFR on the full game, re-solving periods >= t2
    periods [t3, end]    stage 3: vanilla CFR on the full game, re-solving periods >= t3

Freezing the prefix fixes the distribution over histories at the root of every public subgame, so each
stage solves all public subgames at once (re-solving without a safety gadget). Every strategy, composite
or not, is evaluated by exact NashConv in the full game.

Usage: python3 staged.py K N_blue N_stage MC_visits [blueprint_variant]  -> staged_K{K}.json
"""
import json
import random
import sys
import time
import numpy as np
from auction import Tree
from cfr import Solver, rm, normalize


def node_periods(T):
    t = -np.ones(T.M, dtype=np.int64)
    for n in T.dec:
        t[n] = T.keys[n]["full"][1]
    return t


class FrozenCFR(Solver):
    """Vanilla CFR in which decisions before period t0 follow the fixed edge factors f0."""

    def __init__(self, T, variant, f0, t0, tnode):
        super().__init__(T, variant)
        self.f0, self.t0 = f0, t0
        de = np.concatenate([T.pedges[1], T.pedges[2]])
        self.frozen_e = de[tnode[T.parent[de]] < t0]
        self.live = {i: tnode[self.pp[i]] >= t0 for i in (1, 2)}

    def step(self):
        T = self.T
        sigma = rm(self.R, self.bmask)
        f = self.edge_factors(sigma, self.bid)
        f[self.frozen_e] = self.f0[self.frozen_e]
        pi = self.reaches(f)
        v = self.values(f)
        for i in (1, 2):
            sg = 1.0 if i == 1 else -1.0
            o = 2 if i == 1 else 1
            m = self.live[i]
            e, h = self.pe[i][m], self.pp[i][m]
            np.add.at(self.R, (self.bid[h], T.eact[e]), pi[o][h] * pi[0][h] * sg * (v[e] - v[h]))
            np.add.at(self.Sb, (self.bid[h], T.eact[e]), pi[i][h] * pi[0][h] * f[e])
        self.it += 1

    def composite(self, t_until=None):
        """Edge factors: frozen prefix, this solver's compiled table from t0 on."""
        T = self.T
        tab = normalize(self.Sb, self.bmask)
        f = self.edge_factors(tab, self.bid)
        f[self.frozen_e] = self.f0[self.frozen_e]
        return f


class FrozenES:
    """External-sampling MCCFR on the full game; decisions before t0 follow f0 (not updated)."""

    def __init__(self, T, f0, t0, tnode, seed=0, variant="full"):
        self.T, self.f0, self.t0 = T, f0, t0
        S = Solver(T, variant)
        self.bid, self.nb, self.bmask = S.bid, S.nb, S.bmask
        self.ev_full = Solver(T, "full")
        kids = [[] for _ in range(T.M)]
        for e in range(1, T.M):
            kids[T.parent[e]].append(e)
        for n in range(T.M):
            if T.kind[n] == 1:
                kids[n].sort(key=lambda e: T.eact[e])
        self.kids, self.kind, self.player = kids, T.kind.tolist(), T.player.tolist()
        self.prob, self.pay, self.b = T.eprob.tolist(), T.payoff.tolist(), self.bid.tolist()
        self.tn = tnode.tolist()
        self.fz = f0.tolist()
        w = self.bmask.shape[1]
        self.R = [[0.0] * w for _ in range(self.nb)]
        self.S = [[0.0] * w for _ in range(self.nb)]
        self.rng = random.Random(seed)
        self.visits = 0

    def strat(self, b, n):
        pos = [x if x > 0 else 0.0 for x in self.R[b][:n]]
        s = sum(pos)
        return [p / s for p in pos] if s > 0 else [1.0 / n] * n

    def sample(self, ch, probs):
        u, acc = self.rng.random(), 0.0
        for c, p in zip(ch, probs):
            acc += p
            if u <= acc:
                return c
        return ch[-1]

    def walk(self, h, i):
        self.visits += 1
        k = self.kind[h]
        if k == 2:
            return self.pay[h] if i == 1 else -self.pay[h]
        ch = self.kids[h]
        if k == 0:
            return self.walk(ch[0], i) if len(ch) == 1 else self.walk(self.sample(ch, [self.prob[c] for c in ch]), i)
        if self.tn[h] < self.t0:                       # frozen decision
            sg = [self.fz[c] for c in ch]
            if self.player[h] == i:                    # explore own frozen actions, no update
                return sum(p * self.walk(c, i) for p, c in zip(sg, ch) if p > 0)
            return self.walk(self.sample(ch, sg), i)
        b, n = self.b[h], len(ch)
        sg = self.strat(b, n)
        if self.player[h] == i:
            vals = [self.walk(c, i) for c in ch]
            v = sum(p * x for p, x in zip(sg, vals))
            r = self.R[b]
            for a in range(n):
                r[a] += vals[a] - v
            return v
        s = self.S[b]
        for a in range(n):
            s[a] += sg[a]
        return self.walk(self.sample(ch, sg), i)

    def run(self, visits):
        while self.visits < visits:
            self.walk(0, 1)
            self.walk(0, 2)

    def composite(self, ev):
        S = np.array(self.S)
        tab = normalize(S, self.bmask)
        f = ev.edge_factors(tab, self.bid)
        T = self.T
        de = np.concatenate([T.pedges[1], T.pedges[2]])
        fr = de[np.array(self.tn)[T.parent[de]] < self.t0]
        f[fr] = self.f0[fr]
        return f


def nashconv(ev, f):
    return float(ev.br(1, f) + ev.br(2, f))


# ---------------------------------------------------------------------------------------------------
# Safe re-solving (Resolve gadget, Burch, Johanson and Bowling 2014): the re-solving player P faces an
# opponent O who, at each of its information sets at the subgame root, may terminate and take the value
# its best response earns against the current strategy. P's re-solved strategy then cannot give O more
# than before, so NashConv cannot increase (up to the accuracy of the re-solve).
# ---------------------------------------------------------------------------------------------------

def br_values(ev, i, f):
    """Node values (player i's perspective) when i best-responds with perfect recall to edge factors f."""
    T = ev.T
    sg = 1.0 if i == 1 else -1.0
    q = np.ones(T.M)
    for d in range(1, T.maxd + 1):
        idx = T.levels[d]
        q[idx] = q[T.parent[idx]] * np.where(T.ekind[idx] == i, 1.0, f[idx])
    val = np.where(T.kind == 2, sg * T.payoff, 0.0).astype(float)
    for d in range(T.maxd - 1, -1, -1):
        idx = T.levels[d]
        idx = idx[T.kind[idx] != 2]
        mine = idx[(T.kind[idx] == 1) & (T.player[idx] == i)]
        kids = T.levels[d + 1]
        kp = T.parent[kids]
        sel = ~np.isin(kp, mine)
        np.add.at(val, kp[sel], f[kids[sel]] * val[kids[sel]])
        if len(mine):
            Q = np.zeros((ev.nf, ev.fmask.shape[1]))
            selm = ~sel
            km = kids[selm]
            np.add.at(Q, (ev.fid[kp[selm]], T.eact[km]), q[kp[selm]] * val[km])
            best = np.where(ev.fmask, Q, -np.inf).argmax(1)
            val[mine] = val[T.child_of[mine, best[ev.fid[mine]]]]
    return val


class Gadget:
    """Root structure shared by the vanilla and sampled safe re-solvers."""

    def __init__(self, ev, fref, t0, tnode, P):
        T = ev.T
        self.T, self.ev, self.t0, self.P = T, ev, t0, P
        self.O = 2 if P == 1 else 1
        roots = T.dec[(tnode[T.dec] == t0) & (T.player[T.dec] == 1)]
        self.roots = roots
        Jraw = ev.fid[T.child_of[roots, 0]] if P == 1 else ev.fid[roots]
        uniq, self.J = np.unique(Jraw, return_inverse=True)
        self.nJ = len(uniq)
        self.tval = br_values(ev, self.O, fref)[roots]           # O's terminate value at each root history
        root_of = -np.ones(T.M, dtype=np.int64)
        root_of[roots] = np.arange(len(roots))
        for d in range(1, T.maxd + 1):
            idx = T.levels[d]
            m = root_of[idx] < 0
            root_of[idx[m]] = root_of[T.parent[idx[m]]]
        self.root_of = root_of


class SafeCFR:
    """Vanilla CFR for the re-solving player P (abstraction `variant`) against a full-game opponent with
    the Resolve gadget at the subgame root; P's and chance's prefix follow fref."""

    def __init__(self, T, variant, fref, t0, tnode, P):
        self.T, self.P = T, P
        self.ev = ev = Solver(T, "full")
        self.g = Gadget(ev, fref, t0, tnode, P)
        O = self.g.O
        idsP = Solver(T, variant)
        self.bidP, self.maskP = idsP.bid, idsP.bmask
        self.bidO, self.maskO = ev.fid, ev.fmask
        self.RP = np.zeros(self.maskP.shape); self.SP = np.zeros(self.maskP.shape)
        self.RO = np.zeros(self.maskO.shape)
        self.G = np.zeros((self.g.nJ, 2))                        # gadget regrets: enter, terminate
        de = {i: T.pedges[i] for i in (1, 2)}
        self.liveP = de[P][tnode[T.parent[de[P]]] >= t0]
        self.liveO = de[O][tnode[T.parent[de[O]]] >= t0]
        self.prefP = de[P][tnode[T.parent[de[P]]] < t0]
        self.prefO = de[O][tnode[T.parent[de[O]]] < t0]
        self.fref = fref

    def step(self):
        T, g, P, O, ev = self.T, self.g, self.P, self.g.O, self.ev
        f = np.ones(T.M); f[T.cedges] = T.eprob[T.cedges]
        sP, sO = rm(self.RP, self.maskP), rm(self.RO, self.maskO)
        hP, hO = T.parent[self.liveP], T.parent[self.liveO]
        f[self.liveP] = sP[self.bidP[hP], T.eact[self.liveP]]
        f[self.liveO] = sO[self.bidO[hO], T.eact[self.liveO]]
        f[self.prefP] = self.fref[self.prefP]                   # O's prefix edges stay 1
        pg = rm(self.G, np.ones_like(self.G, dtype=bool))[:, 0]  # enter probability per root infoset
        pi = ev.reaches(f)
        sub = g.root_of >= 0
        pi[O] = pi[O].copy(); pi[O][sub] *= pg[g.J[g.root_of[sub]]]
        v = ev.values(f)
        sgP = 1.0 if P == 1 else -1.0
        np.add.at(self.RP, (self.bidP[hP], T.eact[self.liveP]), pi[O][hP] * pi[0][hP] * sgP * (v[self.liveP] - v[hP]))
        np.add.at(self.SP, (self.bidP[hP], T.eact[self.liveP]), pi[P][hP] * pi[0][hP] * f[self.liveP])
        np.add.at(self.RO, (self.bidO[hO], T.eact[self.liveO]), pi[P][hO] * pi[0][hO] * (-sgP) * (v[self.liveO] - v[hO]))
        r = g.roots
        w = pi[P][r] * pi[0][r]
        enter = np.zeros(g.nJ); term = np.zeros(g.nJ)
        np.add.at(enter, g.J, w * (-sgP) * v[r]); np.add.at(term, g.J, w * g.tval)
        cur = pg * enter + (1 - pg) * term
        self.G[:, 0] += enter - cur; self.G[:, 1] += term - cur

    def table_edges(self):
        T = self.T
        tab = normalize(self.SP, self.maskP)
        return self.liveP, tab[self.bidP[T.parent[self.liveP]], T.eact[self.liveP]]


class SafeES(FrozenES):
    """External-sampling MCCFR version of SafeCFR (both players on the full game)."""

    def __init__(self, T, fref, t0, tnode, P, seed=0):
        super().__init__(T, fref, t0, tnode, seed)
        self.P = P
        self.g = Gadget(Solver(T, "full"), fref, t0, tnode, P)
        self.isroot = {int(h): k for k, h in enumerate(self.g.roots)}
        self.G = [[0.0, 0.0] for _ in range(self.g.nJ)]

    def walk(self, h, i, entered=False):
        self.visits += 1
        k = self.kind[h]
        if k == 2:
            return self.pay[h] if i == 1 else -self.pay[h]
        ch = self.kids[h]
        if not entered and h in self.isroot:                   # gadget for O at the subgame root
            kk = self.isroot[h]; J = self.g.J[kk]; tv = self.g.tval[kk]
            gr = self.G[J]; pos = [max(gr[0], 0.0), max(gr[1], 0.0)]; s = pos[0] + pos[1]
            pe = pos[0] / s if s > 0 else 0.5
            if i == self.g.O:
                ve = self.walk(h, i, True)
                cur = pe * ve + (1 - pe) * tv
                gr[0] += ve - cur; gr[1] += tv - cur
                return cur
            if self.rng.random() < pe:
                return self.walk(h, i, True)
            return 0.0                                          # O terminated: no information for P
        if k == 0:
            return self.walk(ch[0], i) if len(ch) == 1 else self.walk(self.sample(ch, [self.prob[c] for c in ch]), i)
        if self.tn[h] < self.t0:                                # frozen prefix
            if self.player[h] == self.g.O:                      # O's prefix carries weight 1 on every branch
                return sum(self.walk(c, i) for c in ch)
            sg = [self.fz[c] for c in ch]
            if self.player[h] == i:
                return sum(p * self.walk(c, i) for p, c in zip(sg, ch) if p > 0)
            return self.walk(self.sample(ch, sg), i)
        b, n = self.b[h], len(ch)
        sg = self.strat(b, n)
        if self.player[h] == i:
            vals = [self.walk(c, i) for c in ch]
            v = sum(p * x for p, x in zip(sg, vals))
            r = self.R[b]
            for a in range(n):
                r[a] += vals[a] - v
            return v
        s = self.S[b]
        for a in range(n):
            s[a] += sg[a]
        return self.walk(self.sample(ch, sg), i)

    def table_edges(self, ev):
        T = self.T
        tab = normalize(np.array(self.S), self.bmask)
        de = T.pedges[self.P]
        live = de[np.array(self.tn)[T.parent[de]] >= self.t0]
        return live, tab[self.bid[T.parent[live]], T.eact[live]]


def safe_stage(T, fref, t0, tnode, kind, budget, variant="full", seed=0):
    """Re-solve periods >= t0 safely for both players; returns the new composite edge factors."""
    f = fref.copy()
    ev = Solver(T, "full")
    for P in (1, 2):
        if kind == "cfr":
            S = SafeCFR(T, variant, fref, t0, tnode, P)
            for _ in range(budget):
                S.step()
            e, val = S.table_edges()
        else:
            S = SafeES(T, fref, t0, tnode, P, seed + P)
            while S.visits < budget:
                S.walk(0, 1); S.walk(0, 2)
            e, val = S.table_edges(ev)
        f[e] = val
    return f


def subtree_share(T, tnode, t0):
    """Share of tree nodes at or below the decision nodes of period t0."""
    inside = np.zeros(T.M, dtype=bool)
    for d in range(T.maxd + 1):
        idx = T.levels[d]
        inside[idx] = (tnode[idx] >= t0) | (inside[T.parent[idx]] if d > 0 else False)
    return float(inside.mean())


if __name__ == "__main__":
    K, NB, NS, MC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    blue = sys.argv[5] if len(sys.argv) > 5 else "A"
    T = Tree(K)
    tnode = node_periods(T)
    ev = Solver(T, "full")
    t1, t2, t3 = max(K - 2, 1), K - 1, K
    out = {"K": K, "nodes": int(T.M), "N_blueprint": NB, "N_stage": NS, "mc_visits": MC, "blueprint": blue,
           "stage_starts": [t1, t2, t3], "subtree_share": {t: subtree_share(T, tnode, t) for t in (t1, t2, t3)},
           "rows": []}
    print("subtree share", out["subtree_share"], flush=True)

    def record(name, f, secs):
        nc = nashconv(ev, f)
        out["rows"].append({"strategy": name, "nashconv": nc, "seconds": secs})
        print(f"{name:60s} NashConv {nc:.5f}   {secs:6.1f}s", flush=True)
        return f

    t0 = time.time(); F = Solver(T, "full")
    for _ in range(NB):
        F.step()
    tab, ids = F.compiled(); fF = record("reference: full-game CFR", F.edge_factors(tab, ids), time.time() - t0)

    t0 = time.time(); B = Solver(T, blue)
    for _ in range(NB):
        B.step()
    tab, ids = B.compiled(); fB = record(f"blueprint ({blue}) everywhere", B.edge_factors(tab, ids), time.time() - t0)

    # unsafe re-solving of the last period from the near-equilibrium full-game strategy
    t0 = time.time(); E = FrozenCFR(T, "full", fF, t3, tnode)
    for _ in range(NS):
        E.step()
    record(f"full-game CFR + unsafe exact re-solve from t={t3}", E.composite(), time.time() - t0)
    t0 = time.time()
    record(f"full-game CFR + safe exact re-solve from t={t3}", safe_stage(T, fF, t3, tnode, "cfr", NS), time.time() - t0)

    # staged scheme, unsafe
    t0 = time.time(); S1 = FrozenCFR(T, "H", fB, t1, tnode)
    for _ in range(NS):
        S1.step()
    f1 = record(f"unsafe stage 1: blueprint + phi_H CFR from t={t1}", S1.composite(), time.time() - t0)
    t0 = time.time(); S2 = FrozenES(T, f1, t2, tnode, seed=1); S2.run(MC)
    f2 = record(f"unsafe stage 2: + MCCFR from t={t2}", S2.composite(ev), time.time() - t0)
    t0 = time.time(); S3 = FrozenCFR(T, "full", f2, t3, tnode)
    for _ in range(NS):
        S3.step()
    record(f"unsafe stage 3: + exact CFR from t={t3}", S3.composite(), time.time() - t0)

    # staged scheme, safe
    t0 = time.time(); g1 = record(f"safe stage 1: blueprint + phi_H CFR from t={t1}",
                                  safe_stage(T, fB, t1, tnode, "cfr", NS, variant="H"), time.time() - t0)
    t0 = time.time(); g2 = record(f"safe stage 2: + MCCFR from t={t2}",
                                  safe_stage(T, g1, t2, tnode, "es", MC, seed=10), time.time() - t0)
    t0 = time.time(); record(f"safe stage 3: + exact CFR from t={t3}",
                             safe_stage(T, g2, t3, tnode, "cfr", NS), time.time() - t0)
    json.dump(out, open(f"staged_K{K}.json", "w"), indent=1)
