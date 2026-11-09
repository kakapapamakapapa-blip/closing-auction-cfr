"""Section 10.4: external-sampling MCCFR (Lanctot et al. 2009) on the same tree and abstractions.

Regrets and average strategies live on buckets. One iteration = one traversal for each player.
The average strategy uses stochastically weighted averaging (the sampled opponent node adds its
current strategy), which in expectation gives the chance-weighted realization weights of the
compiled table of Theorem 7.3. Cost is counted in node visits; vanilla CFR visits all M nodes
per iteration.
"""
import json
import math
import random
import sys
import time
import numpy as np
from auction import Tree
from cfr import Solver


class ES:
    def __init__(self, T, variant, seed):
        self.T = T
        S = Solver(T, variant)
        self.ev = S                                   # exact evaluator
        self.bid, self.nb, self.bmask = S.bid, S.nb, S.bmask
        M = T.M
        kids = [[] for _ in range(M)]
        for e in range(1, M):
            kids[T.parent[e]].append(e)
        for n in range(M):
            if T.kind[n] == 1:
                kids[n].sort(key=lambda e: T.eact[e])
        self.kids = kids
        self.kind = T.kind.tolist()
        self.player = T.player.tolist()
        self.prob = T.eprob.tolist()
        self.pay = T.payoff.tolist()
        self.b = self.bid.tolist()
        w = self.bmask.shape[1]
        self.R = [[0.0] * w for _ in range(self.nb)]
        self.S = [[0.0] * w for _ in range(self.nb)]
        self.rng = random.Random(seed)
        self.visits = 0

    def strat(self, b, n):
        r = self.R[b]
        pos = [x if x > 0 else 0.0 for x in r[:n]]
        s = sum(pos)
        return [p / s for p in pos] if s > 0 else [1.0 / n] * n

    def walk(self, h, i):
        self.visits += 1
        k = self.kind[h]
        if k == 2:
            return self.pay[h] if i == 1 else -self.pay[h]
        ch = self.kids[h]
        if k == 0:
            if len(ch) == 1:
                return self.walk(ch[0], i)
            u, acc = self.rng.random(), 0.0
            for c in ch:
                acc += self.prob[c]
                if u <= acc:
                    return self.walk(c, i)
            return self.walk(ch[-1], i)
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
        u, acc = self.rng.random(), 0.0
        for a in range(n):
            acc += sg[a]
            if u <= acc:
                return self.walk(ch[a], i)
        return self.walk(ch[-1], i)

    def iterate(self):
        self.walk(0, 1)
        self.walk(0, 2)

    def exploitability(self):
        S = np.array(self.S)
        s = S.sum(1, keepdims=True)
        U = self.bmask / self.bmask.sum(1, keepdims=True)
        tab = np.where(s > 0, S / np.where(s > 0, s, 1), U)
        nc, _ = self.ev.nashconv(tab, self.bid)
        return float(nc)


if __name__ == "__main__":
    K, budget, seeds = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    variants = sys.argv[4].split(",") if len(sys.argv) > 4 else ["full", "H", "A"]
    T = Tree(K)
    marks = [int(round(10 ** x)) for x in np.arange(4.0, math.log10(budget) + 1e-9, 0.25)]
    out = {"K": K, "M": int(T.M), "budget_visits": budget, "runs": []}
    for v in variants:
        for seed in range(seeds):
            es = ES(T, v, seed)
            curve, train_s, j = [], 0.0, 0
            while j < len(marks):
                t1 = time.perf_counter()
                es.iterate()
                train_s += time.perf_counter() - t1      # training time only; evaluation excluded
                if es.visits >= marks[j]:
                    curve.append({"visits": es.visits, "seconds": train_s, "nc": es.exploitability()})
                    j += 1
            out["runs"].append({"variant": v, "seed": seed, "entries": int(es.bmask.sum()), "curve": curve})
            print(v, seed, [(c["visits"], round(c["nc"], 4)) for c in curve[-3:]], round(train_s, 1), flush=True)
    tag = sys.argv[5] if len(sys.argv) > 5 else ""
    json.dump(out, open(f"mccfr_K{K}{tag}.json", "w"), indent=1)
