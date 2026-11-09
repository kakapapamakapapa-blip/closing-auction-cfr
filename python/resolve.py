"""Staged endgame re-solving with a Maxmargin gadget (Moravcik et al. 2016), on instance A in Gamma_0.

Chain (all on phi_H, which reproduces full-game CFR exactly, Corollary 5.11):
    blueprint on a coarse abstraction for the whole game;
    stage 1: external-sampling MCCFR on phi_H, re-solving periods >= c-2 (largest subtree, sampled);
    stage 2: vanilla CFR on phi_H, re-solving periods >= c-1;
    stage 3: vanilla CFR on phi_H, re-solving period c.
Each stage freezes earlier periods at the strategy built so far. Safe stages use the Maxmargin gadget:
the opponent O picks the root information set J at which the re-solving player P's margin
(cbv_J - value_J) / W_J is smallest, and P maximizes that minimum. cbv_J is O's best-response value at J
against the strategy before the stage, so a nonnegative margin means O gains nothing anywhere.
The opponent inside the gadget plays on the full game.

Usage: python3 resolve.py K N_blue MC_visits N_cfr2 N_cfr3  -> resolve_K{K}.json
"""
import json
import random
import sys
import time
import numpy as np
from auction import Tree
from cfr import Solver, rm, normalize
from staged import node_periods, FrozenCFR, FrozenES, nashconv, Gadget


class MMCFR:
    """Vanilla CFR with the Maxmargin gadget. P plays on `variant`, O on the full game."""

    def __init__(self, T, variant, fref, t0, tnode, P):
        self.T, self.P = T, P
        self.ev = ev = Solver(T, "full")
        self.g = g = Gadget(ev, fref, t0, tnode, P)
        O = g.O
        sp = Solver(T, variant)
        self.bidP, self.maskP = sp.bid, sp.bmask
        self.bidO, self.maskO = ev.fid, ev.fmask
        self.RP = np.zeros(self.maskP.shape); self.SP = np.zeros(self.maskP.shape)
        self.RO = np.zeros(self.maskO.shape)
        de = {i: T.pedges[i] for i in (1, 2)}
        self.liveP = de[P][tnode[T.parent[de[P]]] >= t0]
        self.liveO = de[O][tnode[T.parent[de[O]]] >= t0]
        self.prefP = de[P][tnode[T.parent[de[P]]] < t0]
        self.fref = fref
        f = np.ones(T.M); f[T.cedges] = T.eprob[T.cedges]; f[self.prefP] = fref[self.prefP]
        pi = ev.reaches(f)
        w = pi[P][g.roots] * pi[0][g.roots]
        self.W = np.zeros(g.nJ); np.add.at(self.W, g.J, w)
        self.w = w
        self.alive = self.W > 1e-300
        self.GQ = np.zeros(g.nJ)

    def step(self):
        T, g, P, O, ev = self.T, self.g, self.P, self.g.O, self.ev
        f = np.ones(T.M); f[T.cedges] = T.eprob[T.cedges]
        sP, sO = rm(self.RP, self.maskP), rm(self.RO, self.maskO)
        hP, hO = T.parent[self.liveP], T.parent[self.liveO]
        f[self.liveP] = sP[self.bidP[hP], T.eact[self.liveP]]
        f[self.liveO] = sO[self.bidO[hO], T.eact[self.liveO]]
        f[self.prefP] = self.fref[self.prefP]
        pos = np.where(self.alive, np.maximum(self.GQ, 0), 0)
        q = pos / pos.sum() if pos.sum() > 0 else self.alive / self.alive.sum()
        scale = np.where(self.alive, q / np.where(self.alive, self.W, 1), 0)
        pi = ev.reaches(f)
        sub = g.root_of >= 0
        pi[O] = pi[O].copy(); pi[O][sub] = pi[O][sub] * scale[g.J[g.root_of[sub]]]
        v = ev.values(f)
        sgP = 1.0 if P == 1 else -1.0
        np.add.at(self.RP, (self.bidP[hP], T.eact[self.liveP]), pi[O][hP] * pi[0][hP] * sgP * (v[self.liveP] - v[hP]))
        np.add.at(self.SP, (self.bidP[hP], T.eact[self.liveP]), pi[P][hP] * pi[0][hP] * f[self.liveP])
        np.add.at(self.RO, (self.bidO[hO], T.eact[self.liveO]), pi[P][hO] * pi[0][hO] * (-sgP) * (v[self.liveO] - v[hO]))
        r = g.roots
        enter = np.zeros(g.nJ); term = np.zeros(g.nJ)
        np.add.at(enter, g.J, self.w * (-sgP) * v[r]); np.add.at(term, g.J, self.w * g.tval)
        m = np.where(self.alive, (enter - term) / np.where(self.alive, self.W, 1), 0)   # O's gain at J
        self.GQ += np.where(self.alive, m - (q * m).sum(), 0)

    def table_edges(self):
        T = self.T
        tab = normalize(self.SP, self.maskP)
        return self.liveP, tab[self.bidP[T.parent[self.liveP]], T.eact[self.liveP]]


class MMES:
    """External-sampling MCCFR with the Maxmargin gadget; P on `variant`, O on the full game."""

    def __init__(self, T, variant, fref, t0, tnode, P, seed=0):
        self.T, self.P, self.t0 = T, P, t0
        ev = Solver(T, "full")
        self.g = g = Gadget(ev, fref, t0, tnode, P)
        self.O = g.O
        sp = Solver(T, variant)
        nbP = sp.nb
        b = np.where(T.player == P, sp.bid, nbP + ev.fid)
        self.b = b.tolist()
        self.maskP = sp.bmask
        self.nbP = nbP
        kids = [[] for _ in range(T.M)]
        for e in range(1, T.M):
            kids[T.parent[e]].append(e)
        for n in range(T.M):
            if T.kind[n] == 1:
                kids[n].sort(key=lambda e: T.eact[e])
        self.kids, self.kind, self.player = kids, T.kind.tolist(), T.player.tolist()
        self.prob, self.pay = T.eprob.tolist(), T.payoff.tolist()
        nb = nbP + ev.nf
        self.R = [[0.0] * 3 for _ in range(nb)]
        self.S = [[0.0] * 3 for _ in range(nb)]
        self.rng = random.Random(seed)
        self.visits = 0
        f = np.ones(T.M); f[T.cedges] = T.eprob[T.cedges]
        de = T.pedges[P]; pref = de[tnode[T.parent[de]] < t0]; f[pref] = fref[pref]
        pi = ev.reaches(f)
        roots = g.roots
        w = pi[P][roots] * pi[0][roots]; c = pi[0][roots]
        self.W = np.zeros(g.nJ); np.add.at(self.W, g.J, w)
        self.C = np.zeros(g.nJ); np.add.at(self.C, g.J, c)
        self.alive = np.where(self.W > 1e-300)[0].tolist()
        self.byJ = {J: np.where(g.J == J)[0] for J in self.alive}
        self.wroot, self.croot = w, c
        self.GQ = [0.0] * g.nJ

    def strat(self, b, n):
        pos = [x if x > 0 else 0.0 for x in self.R[b][:n]]
        s = sum(pos)
        return [p / s for p in pos] if s > 0 else [1.0 / n] * n

    def pick(self, idx, weights):
        u, acc, tot = self.rng.random(), 0.0, float(weights.sum())
        for k, wt in zip(idx, weights):
            acc += wt / tot
            if u <= acc:
                return int(k)
        return int(idx[-1])

    def walk(self, h, i, wt):
        self.visits += 1
        k = self.kind[h]
        if k == 2:
            return self.pay[h] if i == 1 else -self.pay[h]
        ch = self.kids[h]
        if k == 0:
            if len(ch) == 1:
                return self.walk(ch[0], i, wt)
            u, acc = self.rng.random(), 0.0
            for c in ch:
                acc += self.prob[c]
                if u <= acc:
                    return self.walk(c, i, wt)
            return self.walk(ch[-1], i, wt)
        b, n = self.b[h], len(ch)
        sg = self.strat(b, n)
        if self.player[h] == i:
            vals = [self.walk(c, i, wt) for c in ch]
            v = sum(p * x for p, x in zip(sg, vals))
            r = self.R[b]
            for a in range(n):
                r[a] += wt * (vals[a] - v)
            return v
        s = self.S[b]
        for a in range(n):
            s[a] += wt * sg[a]
        u, acc = self.rng.random(), 0.0
        for a in range(n):
            acc += sg[a]
            if u <= acc:
                return self.walk(ch[a], i, wt)
        return self.walk(ch[-1], i, wt)

    def iterate(self):
        g, roots = self.g, self.g.roots
        pos = [max(self.GQ[J], 0.0) for J in self.alive]
        s = sum(pos)
        q = [p / s for p in pos] if s > 0 else [1.0 / len(self.alive)] * len(self.alive)
        # traverser P: sample J by q, a root history by chance reach, weight C_J / W_J
        u, acc, J = self.rng.random(), 0.0, self.alive[-1]
        for Jc, qc in zip(self.alive, q):
            acc += qc
            if u <= acc:
                J = Jc; break
        idx = self.byJ[J]
        k = self.pick(idx, self.croot[idx])
        self.walk(int(roots[k]), self.P, float(self.C[J] / self.W[J]))
        # traverser O: every J, a root history by P-and-chance reach, weight W_J; margins update GQ
        m = []
        for J in self.alive:
            idx = self.byJ[J]
            k = self.pick(idx, self.wroot[idx])
            v = self.walk(int(roots[k]), self.O, float(self.W[J]))
            m.append(v - float(g.tval[k]))
        mbar = sum(qc * mc for qc, mc in zip(q, m))
        for J, mc in zip(self.alive, m):
            self.GQ[J] += mc - mbar

    def table_edges(self):
        T = self.T
        tab = normalize(np.array(self.S[:self.nbP]), self.maskP)
        de = T.pedges[self.P]
        tn = self.tnode
        live = de[tn[T.parent[de]] >= self.t0]
        bid = np.array(self.b)
        return live, tab[bid[T.parent[live]], T.eact[live]]


def mm_stage(T, fref, t0, tnode, kind, budget, variant="H", seed=0):
    f = fref.copy()
    for P in (1, 2):
        if kind == "cfr":
            S = MMCFR(T, variant, fref, t0, tnode, P)
            for _ in range(budget):
                S.step()
        else:
            S = MMES(T, variant, fref, t0, tnode, P, seed + P)
            S.tnode = tnode
            while S.visits < budget:
                S.iterate()
        e, val = S.table_edges()
        f[e] = val
    return f


def unsafe_stage(T, fref, t0, tnode, kind, budget, variant="H", seed=0):
    if kind == "cfr":
        S = FrozenCFR(T, variant, fref, t0, tnode)
        for _ in range(budget):
            S.step()
        return S.composite()
    S = FrozenES(T, fref, t0, tnode, seed, variant)
    S.run(budget)
    return S.composite(Solver(T, "full"))


if __name__ == "__main__":
    K, NB, MC, N2, N3 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    T = Tree(K)
    tn = node_periods(T)
    ev = Solver(T, "full")
    t1, t2, t3 = K - 2, K - 1, K
    out = {"K": K, "nodes": int(T.M), "N_blueprint": NB, "mc_visits": MC, "N_stage2": N2, "N_stage3": N3,
           "stage_starts": [t1, t2, t3], "rows": []}

    def rec(name, f, secs):
        nc = nashconv(ev, f)
        out["rows"].append({"strategy": name, "nashconv": nc, "seconds": secs})
        print(f"{name:62s} NashConv {nc:.5f}  {secs:7.1f}s", flush=True)
        return f

    t = time.time(); F = Solver(T, "H")
    for _ in range(NB):
        F.step()
    tab, ids = F.compiled(); fF = rec("reference: CFR on phi_H (= full game), whole game", F.edge_factors(tab, ids), time.time() - t)
    t = time.time(); B = Solver(T, "A")
    for _ in range(NB):
        B.step()
    tab, ids = B.compiled(); fB = rec("blueprint: CFR on drop s0 and chi", B.edge_factors(tab, ids), time.time() - t)

    # safety check on a near-equilibrium strategy: re-solve period c
    t = time.time(); rec("reference + unsafe re-solve of period c", unsafe_stage(T, fF, t3, tn, "cfr", N3), time.time() - t)
    t = time.time(); rec("reference + Maxmargin re-solve of period c", mm_stage(T, fF, t3, tn, "cfr", N3), time.time() - t)

    # staged chains from the blueprint
    for label, stage in (("unsafe", unsafe_stage), ("Maxmargin", mm_stage)):
        t = time.time(); f1 = rec(f"{label} 1: blueprint + MCCFR on phi_H from c-2", stage(T, fB, t1, tn, "es", MC, seed=3), time.time() - t)
        t = time.time(); f2 = rec(f"{label} 2: + CFR on phi_H from c-1", stage(T, f1, t2, tn, "cfr", N2), time.time() - t)
        t = time.time(); rec(f"{label} 3: + CFR on phi_H for period c", stage(T, f2, t3, tn, "cfr", N3), time.time() - t)
    # single safe re-solve from c-2 with vanilla CFR (no staging)
    t = time.time(); rec("Maxmargin single: blueprint + CFR on phi_H from c-2", mm_stage(T, fB, t1, tn, "cfr", N2 + N3), time.time() - t)
    json.dump(out, open(f"resolve_K{K}.json", "w"), indent=1)
