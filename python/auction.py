"""Toy discrete double auction with a commitment cutoff (an instance of the model of Section 2).

  * Player 1 is an execution agent who must buy 1 unit (arrival price 100, shortfall
    penalty LAM). Player 2 is a liquidity provider who may sell 1 unit.
  * K pre-cutoff periods t = 0..K-1: chance draws a public signal s_t in {U, D}
    (persistent Markov chain; the number of U's plays the role of the excitation),
    then both players act simultaneously. Orders are on-close LOC orders at the
    low (100) or high (101) limit; before the cutoff they are hidden and cancellable.
  * At the cutoff c = K, chance draws exogenous on-close interest X in {-1, 0, +1}
    (law depends on the excitation): X = +1 adds a competing LOC buy at 101 with
    time stamp c - 1/2 (after pre-cutoff orders, before post-cutoff ones); X = -1
    adds an LOC sell at 100. A public imbalance indicator chi in {-1, 0, +1} is
    disseminated: it reflects X and both players' resting orders.
  * One post-cutoff period: a player without a resting order may submit; no cancels.
  * Cross: uniform price maximizing volume, then minimal imbalance, then closeness
    to 100; price-time priority; residual sell liquidity of 1 unit rests at 101
    behind all other orders.
Payoffs: u_1 = -(p-100) q_1 - LAM (1-q_1)^2, u_2 = (p-100) q_2; the toy game is the
zero-sum relative-performance game U_1 = u_1 - u_2, U_2 = -U_1.
"""
import numpy as np

PRICES = (99, 100, 101)
LIMITS = {1: (100, 101), 2: (100, 101)}
LAM = 1.5
EXO_STAMP = None   # set from K


def sig_probs(prev):
    if prev is None:
        return {"U": 0.5, "D": 0.5}
    stay = 0.7
    return {prev: stay, ("D" if prev == "U" else "U"): 1 - stay}


def x_probs(n_up, K):
    f = n_up / K
    pp = 0.15 + 0.45 * f
    pm = 0.60 - 0.45 * f
    return {+1: pp, 0: 1 - pp - pm, -1: pm}


def cross(o1, o2, X, K):
    """o_i = None or (limit, stamp). Player 1 buys, player 2 sells. Returns price, q1, q2."""
    buys, sells = [], [(101, 1e9, "bg", 1.0)]
    if X == +1:
        buys.append((101, K - 0.5, "ex", 1.0))
    if X == -1:
        sells.append((100, -1, "ex", 1.0))
    if o1 is not None:
        buys.append((o1[0], o1[1], 1, 1.0))
    if o2 is not None:
        sells.append((o2[0], o2[1], 2, 1.0))
    best = None
    for p in PRICES:
        D = sum(b[3] for b in buys if b[0] >= p)
        S = sum(x[3] for x in sells if x[0] <= p)
        V = min(D, S)
        k = (-V, abs(D - S), abs(p - 100), p)
        if best is None or k < best[0]:
            best = (k, p, V)
    _, p, V = best
    if V <= 0:
        return p, 0.0, 0.0
    f1 = _alloc([b for b in buys if b[0] >= p], V, lambda b: (-b[0], b[1]), 1)
    f2 = _alloc([x for x in sells if x[0] <= p], V, lambda x: (x[0], x[1]), 2)
    return p, f1, f2


def _alloc(elig, V, keyf, who):
    elig = sorted(elig, key=keyf)
    rem, got, i = V, 0.0, 0
    while i < len(elig) and rem > 1e-12:
        j = i
        while j < len(elig) and keyf(elig[j]) == keyf(elig[i]):
            j += 1
        grp = elig[i:j]
        tot = sum(b[3] for b in grp)
        take = min(rem, tot)
        for b in grp:
            if b[2] == who:
                got += take * b[3] / tot
        rem -= take
        i = j
    return got


def payoff(p, q1, q2):
    u1 = -(p - 100) * q1 - LAM * (1 - q1) ** 2
    u2 = (p - 100) * q2
    return u1 - u2


def actions(phase, order):
    if order is None:
        return ["noop", "lo", "hi"]
    return ["noop", "cancel"] if phase == "pre" else ["noop"]


def chi_of(o1, o2, X):
    B = (o1 is not None) + (X == 1)
    S = (o2 is not None) + (X == -1)
    return int(np.clip(B - S, -1, 1))


VARIANTS = ["full", "H", "H_s0", "H_chi", "A"]


def keys_for(i, t, sig, own_acts, order, chi):
    full = (i, t, sig, own_acts, chi)
    H = (i, t, sig, order, chi)
    H_s0 = (i, t, sig[1:], order, chi)
    H_chi = (i, t, sig, order)
    A = (i, t, sig[1:], order)
    H_s0m = (i, t, sig if t == 0 else sig[1:], order, chi)   # forget s0 once s1 is observed
    return {"full": full, "H": H, "H_s0": H_s0, "H_chi": H_chi, "A": A, "H_s0m": H_s0m}


class Tree:
    def __init__(self, K, model=False):
        """model=True builds the model game: the law of X uses the stationary mean 1/2 in place of s0."""
        self.K = K
        self.model = model
        self._xstate = {}
        self.parent, self.depth, self.kind, self.player = [], [], [], []
        self.ekind, self.eprob, self.eact, self.payoff = [], [], [], []
        self.u1, self.u2 = [], []
        self.keys = []      # dict variant -> key, for decision nodes
        self.nact = []
        self._build()
        self._finalize()

    def _new(self, parent, depth, kind, player, ekind, eprob, eact):
        n = len(self.parent)
        self.parent.append(parent); self.depth.append(depth)
        self.kind.append(kind); self.player.append(player)
        self.ekind.append(ekind); self.eprob.append(eprob); self.eact.append(eact)
        self.payoff.append(0.0); self.keys.append(None); self.nact.append(0)
        self.u1.append(0.0); self.u2.append(0.0)
        return n

    def _build(self):
        K = self.K
        root = self._new(-1, 0, 0, 0, -1, 1.0, -1)
        # state: t, sig, acts1, acts2, o1, o2
        self._pre_chance(root, 0, (), (), (), None, None)

    # pre-cutoff period t: chance signal -> P1 -> P2
    def _pre_chance(self, node, t, sig, a1, a2, o1, o2):
        K = self.K
        if t == K:
            return self._cut_chance(node, sig, a1, a2, o1, o2)
        prev = sig[-1] if sig else None
        for s, pr in sig_probs(prev).items():
            child = self._new(node, self.depth[node] + 1, 1, 1, 0, pr, -1)
            self._p1(child, t, sig + (s,), a1, a2, o1, o2, "pre", None)

    def _apply(self, i, act, order, t):
        if act == "noop":
            return order
        if act == "cancel":
            return None
        return (LIMITS[i][0 if act == "lo" else 1], t)

    def _p1(self, node, t, sig, a1, a2, o1, o2, phase, chi):
        acts = actions(phase, o1)
        self.kind[node] = 1; self.player[node] = 1; self.nact[node] = len(acts)
        self.keys[node] = keys_for(1, t, sig, a1, o1, chi)
        for k, act in enumerate(acts):
            child = self._new(node, self.depth[node] + 1, 1, 2, 1, 1.0, k)
            self._p2(child, t, sig, a1 + (act,), a2, o1, o2, phase, chi, act)

    def _p2(self, node, t, sig, a1, a2, o1, o2, phase, chi, act1):
        acts = actions(phase, o2)
        self.kind[node] = 1; self.player[node] = 2; self.nact[node] = len(acts)
        self.keys[node] = keys_for(2, t, sig, a2, o2, chi)
        n1 = self._apply(1, act1, o1, t)
        for k, act in enumerate(acts):
            n2 = self._apply(2, act, o2, t)
            if phase == "pre":
                child = self._new(node, self.depth[node] + 1, 0, 0, 2, 1.0, k)
                self.kind[child] = 0
                self._pre_chance(child, t + 1, sig, a1, a2 + (act,), n1, n2)
            else:
                child = self._new(node, self.depth[node] + 1, 2, 0, 2, 1.0, k)
                self._terminal(child, n1, n2, chi)

    def _cut_chance(self, node, sig, a1, a2, o1, o2):
        K = self.K
        n_up = sum(s == "U" for s in sig)
        if self.model:
            n_up = sum(s == "U" for s in sig[1:]) + 0.5
        for X, pr in x_probs(n_up, K).items():
            if pr <= 0:
                continue
            chi = chi_of(o1, o2, X)
            child = self._new(node, self.depth[node] + 1, 1, 1, 0, pr, -1)
            self._xstate[child] = X
            self._p1(child, K, sig, a1, a2, o1, o2, "post", chi)

    _xstate = {}

    def _terminal(self, node, o1, o2, chi):
        # recover X: walk up to the cut chance edge
        n = node
        while n not in self._xstate:
            n = self.parent[n]
        X = self._xstate[n]
        p, q1, q2 = cross(o1, o2, X, self.K)
        self.payoff[node] = payoff(p, q1, q2)
        self.u1[node] = -(p - 100) * q1 - LAM * (1 - q1) ** 2
        self.u2[node] = (p - 100) * q2

    def _finalize(self):
        for name in ["parent", "depth", "kind", "player", "ekind", "eprob", "eact", "payoff", "nact", "u1", "u2"]:
            setattr(self, name, np.array(getattr(self, name)))
        self.M = len(self.parent)
        self.dec = np.where(self.kind == 1)[0]
        self.term = np.where(self.kind == 2)[0]
        self.maxd = int(self.depth.max())
        self.levels = [np.where(self.depth == d)[0] for d in range(self.maxd + 1)]
        # child_of[node, slot]
        self.child_of = -np.ones((self.M, 3), dtype=np.int64)
        e = np.where(self.ekind >= 1)[0]
        self.child_of[self.parent[e], self.eact[e]] = e
        self.pedges = {i: np.where(self.ekind == i)[0] for i in (1, 2)}
        self.cedges = np.where(self.ekind == 0)[0]

    def ids(self, variant):
        """Map decision nodes to bucket ids under a variant; returns (id array, n, legal mask)."""
        table, ids = {}, -np.ones(self.M, dtype=np.int64)
        legal = []
        for n in self.dec:
            k = self.keys[n][variant]
            if k not in table:
                table[k] = len(table)
                legal.append(self.nact[n])
            else:
                assert legal[table[k]] == self.nact[n], "legal sets differ inside a bucket"
            ids[n] = table[k]
        mask = np.zeros((len(table), 3), dtype=bool)
        for b, L in enumerate(legal):
            mask[b, :L] = True
        owner = np.zeros(len(table), dtype=np.int64)
        for n in self.dec:
            owner[ids[n]] = self.player[n]
        return ids, len(table), mask, owner
