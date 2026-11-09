"""Second toy instance (Section 10.6): the closing auction with a continuous market, as a
general-sum game.

Same as auction.py, plus: the buyer (player 1) may at any period, before or after the cutoff,
take one unit from the continuous book at the current ask, provided it is not yet filled and has
no resting on-close order (no-overfill). The take is printed on the public tape. The ask is
100.5 after a D signal and 101.5 after a U signal before the cutoff, and 100.5 + 0.5 (X + 1)
after it. Assumption 2.2 does not hold: the buyer's inventory can change after the cutoff.
Payoffs are the native ones: u_1 = -(price - 100) q_1 - LAM (1 - q_1)^2, u_2 = (p - 100) q_2.
"""
import numpy as np
from auction import sig_probs, x_probs, cross, chi_of, LIMITS, LAM

VARIANTS = ["full", "H", "H_s0m", "H_s0", "H_chi", "A"]


def ask(phase, sig, X):
    if phase == "pre":
        return 101.5 if sig[-1] == "U" else 100.5
    return 100.5 + 0.5 * (X + 1)


def keys_for(i, t, sig, own_acts, order, chi, tau):
    return {"full": (i, t, sig, own_acts, chi, tau),
            "H": (i, t, sig, order, chi, tau),
            "H_s0m": (i, t, sig if t == 0 else sig[1:], order, chi, tau),
            "H_s0": (i, t, sig[1:], order, chi, tau),
            "H_last": (i, t, sig[-1:], order, chi, tau),
            "H_nosig": (i, t, order, chi, tau),
            "H_chi": (i, t, sig, order, tau),
            "A": (i, t, sig[1:], order, tau)}


class TreeC:
    def __init__(self, K):
        self.K = K
        self._xstate = {}
        self.parent, self.depth, self.kind, self.player = [], [], [], []
        self.ekind, self.eprob, self.eact, self.payoff = [], [], [], []
        self.u1, self.u2, self.keys, self.nact = [], [], [], []
        root = self._new(-1, 0, 0, 0, -1, 1.0, -1)
        self._pre_chance(root, 0, (), (), (), None, None, None, None)
        self._finalize()

    def _new(self, parent, depth, kind, player, ekind, eprob, eact):
        n = len(self.parent)
        for lst, val in ((self.parent, parent), (self.depth, depth), (self.kind, kind), (self.player, player),
                         (self.ekind, ekind), (self.eprob, eprob), (self.eact, eact), (self.payoff, 0.0),
                         (self.u1, 0.0), (self.u2, 0.0), (self.keys, None), (self.nact, 0)):
            lst.append(val)
        return n

    def _acts(self, i, phase, order, tau):
        if i == 1 and tau is not None:
            return ["noop"]
        if order is None:
            base = ["noop", "lo", "hi"]
            return base + (["take"] if i == 1 else [])
        return ["noop", "cancel"] if phase == "pre" else ["noop"]

    def _pre_chance(self, node, t, sig, a1, a2, o1, o2, tau, price):
        if t == self.K:
            return self._cut_chance(node, sig, a1, a2, o1, o2, tau, price)
        prev = sig[-1] if sig else None
        for s, pr in sig_probs(prev).items():
            child = self._new(node, self.depth[node] + 1, 1, 1, 0, pr, -1)
            self._p1(child, t, sig + (s,), a1, a2, o1, o2, "pre", None, tau, price, None)

    def _apply(self, i, act, order, t):
        if act in ("noop", "take"):
            return order
        if act == "cancel":
            return None
        return (LIMITS[i][0 if act == "lo" else 1], t)

    def _p1(self, node, t, sig, a1, a2, o1, o2, phase, chi, tau, price, X):
        acts = self._acts(1, phase, o1, tau)
        self.kind[node] = 1; self.player[node] = 1; self.nact[node] = len(acts)
        self.keys[node] = keys_for(1, t, sig, a1, o1, chi, tau)
        for k, act in enumerate(acts):
            child = self._new(node, self.depth[node] + 1, 1, 2, 1, 1.0, k)
            ntau, nprice = (t, ask(phase, sig, X)) if act == "take" else (tau, price)
            self._p2(child, t, sig, a1 + (act,), a2, o1, o2, phase, chi, act, tau, ntau, nprice, X)

    def _p2(self, node, t, sig, a1, a2, o1, o2, phase, chi, act1, tau, ntau, nprice, X):
        # player 2 moves without seeing player 1's current action: its key uses the old tape tau
        acts = self._acts(2, phase, o2, None)
        self.kind[node] = 1; self.player[node] = 2; self.nact[node] = len(acts)
        self.keys[node] = keys_for(2, t, sig, a2, o2, chi, tau)
        n1 = self._apply(1, act1, o1, t)
        for k, act in enumerate(acts):
            n2 = self._apply(2, act, o2, t)
            if phase == "pre":
                child = self._new(node, self.depth[node] + 1, 0, 0, 2, 1.0, k)
                self._pre_chance(child, t + 1, sig, a1, a2 + (act,), n1, n2, ntau, nprice)
            else:
                child = self._new(node, self.depth[node] + 1, 2, 0, 2, 1.0, k)
                self._terminal(child, n1, n2, X, ntau, nprice)

    def _cut_chance(self, node, sig, a1, a2, o1, o2, tau, price):
        n_up = sum(s == "U" for s in sig)
        for X, pr in x_probs(n_up, self.K).items():
            chi = chi_of(o1, o2, X)
            child = self._new(node, self.depth[node] + 1, 1, 1, 0, pr, -1)
            self._p1(child, self.K, sig, a1, a2, o1, o2, "post", chi, tau, price, X)

    def _terminal(self, node, o1, o2, X, tau, price):
        p, q1, q2 = cross(o1, o2, X, self.K)
        if tau is not None:
            u1 = -(price - 100)
        else:
            u1 = -(p - 100) * q1 - LAM * (1 - q1) ** 2
        u2 = (p - 100) * q2
        self.u1[node], self.u2[node], self.payoff[node] = u1, u2, u1 - u2

    def _finalize(self):
        for name in ["parent", "depth", "kind", "player", "ekind", "eprob", "eact", "payoff", "nact", "u1", "u2"]:
            setattr(self, name, np.array(getattr(self, name)))
        self.M = len(self.parent)
        self.dec = np.where(self.kind == 1)[0]
        self.term = np.where(self.kind == 2)[0]
        self.maxd = int(self.depth.max())
        self.levels = [np.where(self.depth == d)[0] for d in range(self.maxd + 1)]
        self.child_of = -np.ones((self.M, 4), dtype=np.int64)
        e = np.where(self.ekind >= 1)[0]
        self.child_of[self.parent[e], self.eact[e]] = e
        self.pedges = {i: np.where(self.ekind == i)[0] for i in (1, 2)}
        self.cedges = np.where(self.ekind == 0)[0]

    def ids(self, variant):
        table, ids, legal = {}, -np.ones(self.M, dtype=np.int64), []
        for n in self.dec:
            k = self.keys[n][variant]
            if k not in table:
                table[k] = len(table); legal.append(self.nact[n])
            else:
                assert legal[table[k]] == self.nact[n], "legal sets differ inside a bucket"
            ids[n] = table[k]
        mask = np.zeros((len(table), 4), dtype=bool)
        for b, L in enumerate(legal):
            mask[b, :L] = True
        owner = np.zeros(len(table), dtype=np.int64)
        for n in self.dec:
            owner[ids[n]] = self.player[n]
        return ids, len(table), mask, owner
