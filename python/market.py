"""Parametrized closing-auction instances for Sections 6.9-6.10.

One builder covers instance A (no continuous market), instance B (continuous market open before
and after the cutoff), two disclosure regimes and an optional ban on pre-cutoff cancellations:

  * regime "hidden"   : nothing about on-close interest is public before the cutoff (Nasdaq-like:
                        dissemination starts when cancellation stops);
  * regime "disclose" : from period 1 to c - 1 both players see the net direction of resting
                        on-close interest, chi_pre = clip(#buy orders - #sell orders), while
                        cancellation is still allowed (a pre-opening-style regime).

Payoffs: u_1 = -(price - 100) q_1 - lam (1 - q_1)^2 for the buyer, u_2 = (p - 100) q_2 for the
seller; payoff = u_1 - u_2 is the auxiliary zero-sum game Gamma_0.
"""
from dataclasses import dataclass, asdict
import numpy as np
from auction import cross

LIMITS = {1: (100, 101), 2: (100, 101)}
VARIANTS = ["full", "H", "H_s0m", "H_s0", "H_chi", "A"]


@dataclass
class Params:
    K: int = 2
    lam: float = 1.5
    stay: float = 0.7
    pp0: float = 0.15   # P(X = +1) = pp0 + pp1 * f, f = share of U signals
    pp1: float = 0.45
    pm0: float = 0.60   # P(X = -1) = pm0 - pm1 * f
    pm1: float = 0.45
    cont: bool = False  # continuous market for the buyer
    askD: float = 100.5
    askU: float = 101.5
    ask0: float = 100.5  # post-cutoff ask = ask0 + aslope * (X + 1)
    aslope: float = 0.5
    disclose: bool = False
    cancel: bool = True
    model: bool = False  # model game: s0 replaced by its mean 1/2 in the law of X

    def dict(self):
        return asdict(self)


def chi_post(o1, o2, X):
    B = (o1 is not None) + (X == 1)
    S = (o2 is not None) + (X == -1)
    return int(np.clip(B - S, -1, 1))


def chi_pre(o1, o2):
    return int(np.clip((o1 is not None) - (o2 is not None), -1, 1))


class Market:
    def __init__(self, prm: Params):
        self.p = prm
        self.K = prm.K
        self.parent, self.depth, self.kind, self.player = [], [], [], []
        self.ekind, self.eprob, self.eact, self.payoff = [], [], [], []
        self.u1, self.u2, self.keys, self.nact, self.acts = [], [], [], [], []
        root = self._new(-1, 0, 0, 0, -1, 1.0, -1)
        self._pre_chance(root, 0, (), (), (), (), None, None, None, None)
        self._finalize()

    # ---------------- construction ----------------
    def _new(self, parent, depth, kind, player, ekind, eprob, eact):
        n = len(self.parent)
        for lst, val in ((self.parent, parent), (self.depth, depth), (self.kind, kind), (self.player, player),
                         (self.ekind, ekind), (self.eprob, eprob), (self.eact, eact), (self.payoff, 0.0),
                         (self.u1, 0.0), (self.u2, 0.0), (self.keys, None), (self.nact, 0), (self.acts, None)):
            lst.append(val)
        return n

    def sig_probs(self, prev):
        if prev is None:
            return {"U": 0.5, "D": 0.5}
        s = self.p.stay
        return {prev: s, ("D" if prev == "U" else "U"): 1 - s}

    def x_probs(self, sig):
        p = self.p
        n_up = sum(s == "U" for s in sig)
        if p.model:
            n_up = sum(s == "U" for s in sig[1:]) + 0.5
        f = n_up / p.K
        pp = p.pp0 + p.pp1 * f
        pm = p.pm0 - p.pm1 * f
        return {+1: pp, 0: 1 - pp - pm, -1: pm}

    def ask(self, phase, sig, X):
        p = self.p
        if phase == "pre":
            return p.askU if sig[-1] == "U" else p.askD
        return p.ask0 + p.aslope * (X + 1)

    def _acts(self, i, phase, order, tau):
        if i == 1 and tau is not None:
            return ["noop"]
        if order is None:
            return ["noop", "lo", "hi"] + (["take"] if (i == 1 and self.p.cont) else [])
        if phase == "pre" and self.p.cancel:
            return ["noop", "cancel"]
        return ["noop"]

    def _keys(self, i, t, sig, ch, own, order, chi, tau):
        s0m = sig if t == 0 else sig[1:]
        return {"full": (i, t, sig, ch, own, chi, tau),
                "H": (i, t, sig, ch, order, chi, tau),
                "H_s0m": (i, t, s0m, ch, order, chi, tau),
                "H_s0": (i, t, sig[1:], ch, order, chi, tau),
                "H_chi": (i, t, sig, order, tau),
                "A": (i, t, sig[1:], order, tau)}

    def _pre_chance(self, node, t, sig, ch, a1, a2, o1, o2, tau, price):
        if t == self.K:
            return self._cut_chance(node, sig, ch, a1, a2, o1, o2, tau, price)
        if self.p.disclose and t >= 1:
            ch = ch + (chi_pre(o1, o2),)
        prev = sig[-1] if sig else None
        for s, pr in self.sig_probs(prev).items():
            child = self._new(node, self.depth[node] + 1, 1, 1, 0, pr, -1)
            self._p1(child, t, sig + (s,), ch, a1, a2, o1, o2, "pre", None, tau, price, None)

    def _apply(self, i, act, order, t):
        if act in ("noop", "take"):
            return order
        if act == "cancel":
            return None
        return (LIMITS[i][0 if act == "lo" else 1], t)

    def _p1(self, node, t, sig, ch, a1, a2, o1, o2, phase, chi, tau, price, X):
        acts = self._acts(1, phase, o1, tau)
        self.kind[node] = 1; self.player[node] = 1; self.nact[node] = len(acts); self.acts[node] = acts
        self.keys[node] = self._keys(1, t, sig, ch, a1, o1, chi, tau)
        for k, act in enumerate(acts):
            child = self._new(node, self.depth[node] + 1, 1, 2, 1, 1.0, k)
            ntau, nprice = (t, self.ask(phase, sig, X)) if act == "take" else (tau, price)
            self._p2(child, t, sig, ch, a1 + (act,), a2, o1, o2, phase, chi, act, tau, ntau, nprice, X)

    def _p2(self, node, t, sig, ch, a1, a2, o1, o2, phase, chi, act1, tau, ntau, nprice, X):
        acts = self._acts(2, phase, o2, None)
        self.kind[node] = 1; self.player[node] = 2; self.nact[node] = len(acts); self.acts[node] = acts
        self.keys[node] = self._keys(2, t, sig, ch, a2, o2, chi, tau)
        n1 = self._apply(1, act1, o1, t)
        for k, act in enumerate(acts):
            n2 = self._apply(2, act, o2, t)
            if phase == "pre":
                child = self._new(node, self.depth[node] + 1, 0, 0, 2, 1.0, k)
                self._pre_chance(child, t + 1, sig, ch, a1, a2 + (act,), n1, n2, ntau, nprice)
            else:
                child = self._new(node, self.depth[node] + 1, 2, 0, 2, 1.0, k)
                self._terminal(child, n1, n2, X, ntau, nprice)

    def _cut_chance(self, node, sig, ch, a1, a2, o1, o2, tau, price):
        for X, pr in self.x_probs(sig).items():
            if pr <= 0:
                continue
            child = self._new(node, self.depth[node] + 1, 1, 1, 0, pr, -1)
            self._p1(child, self.K, sig, ch, a1, a2, o1, o2, "post", chi_post(o1, o2, X), tau, price, X)

    def _terminal(self, node, o1, o2, X, tau, price):
        p, q1, q2 = cross(o1, o2, X, self.K)
        lam = self.p.lam
        u1 = -(price - 100) if tau is not None else -(p - 100) * q1 - lam * (1 - q1) ** 2
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
        # label of every decision edge
        self.elabel = np.array([""] * self.M, dtype=object)
        for i in (1, 2):
            for e in self.pedges[i]:
                self.elabel[e] = self.acts[self.parent[e]][self.eact[e]]

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
