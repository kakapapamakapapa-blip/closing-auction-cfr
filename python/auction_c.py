"""Instance C: a closing auction with a priority race inside the dark window.

Two strategic buyers (each wants one unit, arrival price 100, shortfall penalty lam) race for
one unit of sell interest with market-on-close (MOC) orders, which carry no price protection.
Same clock and signals as instance A. In each pre-cutoff period t, after both players move, a
competing exogenous MOC buy arrives with probability q and stamp t + 1/2; on-close interest is
hidden before the cutoff, so these arrivals are unobserved. At the cutoff chance draws Z:
'good' brings one unit of sell interest at 100, 'bad' one unit at 101, 'none' nothing; more U
signals make 'bad' more likely. MOC orders fill by time priority, so a buyer who wants the unit
in the good state must submit early, which exposes it to a fill at 101 in the bad state.
Cancelling before the cutoff hedges that exposure; after the cutoff new MOC orders are still
accepted but cannot be cancelled.

Payoffs are native: u_i = -(p - 100) q_i - lam (1 - q_i)^2 for both buyers. With lam < 1 a buyer
prefers no fill to a fill at 101.
The imbalance indicator chi (from the cutoff on) counts all on-close interest, arrivals included.
cancel=False removes the right to cancel resting orders before the cutoff.
"""
import numpy as np
from auction import _alloc, LIMITS

PRICES = (99, 100, 101)
MOC = 102   # market-on-close: a limit above every price on the grid


def z_probs(n_up, K):
    """Closing state Z at the cutoff: 'good' adds a sell at 100, 'bad' a sell at 101, 'none' nothing.
    More U signals make the bad state more likely."""
    f = n_up / K
    return {"good": 0.75 - 0.6 * f, "none": 0.10, "bad": 0.15 + 0.6 * f}


def cross_c(o1, o2, Z, exo, K):
    buys = [(MOC, s, "exo", 1.0) for s in exo]
    sells = []
    if Z == "good":
        sells.append((100, -1, "ex", 1.0))
    if Z == "bad":
        sells.append((101, -1, "ex", 1.0))
    if o1 is not None:
        buys.append((o1[0], o1[1], 1, 1.0))
    if o2 is not None:
        buys.append((o2[0], o2[1], 2, 1.0))
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
    elig = [b for b in buys if b[0] >= p]
    f1 = _alloc(elig, V, lambda b: (-b[0], b[1]), 1)
    f2 = _alloc(elig, V, lambda b: (-b[0], b[1]), 2)
    return p, f1, f2


def chi_c(o1, o2, Z, exo):
    B = (o1 is not None) + (o2 is not None) + len(exo)
    S = (Z != "none")
    return int(np.clip(B - S, -1, 1))


VARIANTS = ["full", "H", "H_s0", "H_chi", "A"]


def keys_for(i, t, sig, own_acts, order, chi):
    return {"full": (i, t, sig, own_acts, chi),
            "H": (i, t, sig, order, chi),
            "H_s0": (i, t, sig[1:], order, chi),
            "H_last": (i, t, sig[-1:], order, chi),
            "H_nosig": (i, t, order, chi),
            "H_chi": (i, t, sig, order),
            "A": (i, t, sig[1:], order)}


class TreeC:
    def __init__(self, K, q=0.4, lam=0.6, cancel=True, persist=0.7):
        self.K, self.q, self.lam, self.cancel, self.persist = K, q, lam, cancel, persist
        cols = ["parent", "depth", "kind", "player", "ekind", "eprob", "eact", "payoff", "u1", "u2", "nact", "keys"]
        for c in cols:
            setattr(self, c, [])
        self.hist = []          # per node: player 1's own action history (for cycle statistics)
        self.price, self.fill1 = [], []
        root = self._new(-1, 0, 0, 0, -1, 1.0, -1)
        self._chance(root, 0, (), (), (), None, None, ())
        self._finalize()

    def _new(self, parent, depth, kind, player, ekind, eprob, eact):
        n = len(self.parent)
        for lst, val in ((self.parent, parent), (self.depth, depth), (self.kind, kind), (self.player, player),
                         (self.ekind, ekind), (self.eprob, eprob), (self.eact, eact), (self.payoff, 0.0),
                         (self.u1, 0.0), (self.u2, 0.0), (self.nact, 0), (self.keys, None), (self.hist, None),
                         (self.price, 0.0), (self.fill1, 0.0)):
            lst.append(val)
        return n

    def _sig(self, prev):
        if prev is None:
            return {"U": 0.5, "D": 0.5}
        return {prev: self.persist, ("D" if prev == "U" else "U"): 1 - self.persist}

    def _acts(self, phase, order):
        if order is None:
            return ["noop", "moc"]
        if phase == "pre" and self.cancel:
            return ["noop", "cancel"]
        return ["noop"]

    def _chance(self, node, t, sig, a1, a2, o1, o2, exo):
        """Chance node at the start of period t: arrival during period t-1 (if t>0), then
        the period-t signal (t < K) or the cutoff draw X (t = K)."""
        K, q = self.K, self.q
        arrivals = [(False, 1.0)] if t == 0 else [(False, 1 - q), (True, q)]
        for arr, pa in arrivals:
            if pa <= 0:
                continue
            exo2 = exo + ((t - 1) + 0.5,) if arr else exo
            if t < K:
                prev = sig[-1] if sig else None
                for s, ps in self._sig(prev).items():
                    child = self._new(node, self.depth[node] + 1, 1, 1, 0, pa * ps, -1)
                    self._p1(child, t, sig + (s,), a1, a2, o1, o2, "pre", None, exo2, None)
            else:
                n_up = sum(s == "U" for s in sig)
                for X, px in z_probs(n_up, K).items():
                    if px <= 0:
                        continue
                    chi = chi_c(o1, o2, X, exo2)
                    child = self._new(node, self.depth[node] + 1, 1, 1, 0, pa * px, -1)
                    self._p1(child, K, sig, a1, a2, o1, o2, "post", chi, exo2, X)

    @staticmethod
    def _apply(i, act, order, t):
        if act == "noop":
            return order
        if act == "cancel":
            return None
        return (MOC, t)

    def _p1(self, node, t, sig, a1, a2, o1, o2, phase, chi, exo, X):
        acts = self._acts(phase, o1)
        self.kind[node] = 1; self.player[node] = 1; self.nact[node] = len(acts)
        self.keys[node] = keys_for(1, t, sig, a1, o1, chi)
        self.hist[node] = a1
        for k, act in enumerate(acts):
            child = self._new(node, self.depth[node] + 1, 1, 2, 1, 1.0, k)
            self._p2(child, t, sig, a1 + (act,), a2, o1, o2, phase, chi, act, exo, X)

    def _p2(self, node, t, sig, a1, a2, o1, o2, phase, chi, act1, exo, X):
        acts = self._acts(phase, o2)
        self.kind[node] = 1; self.player[node] = 2; self.nact[node] = len(acts)
        self.keys[node] = keys_for(2, t, sig, a2, o2, chi)
        n1 = self._apply(1, act1, o1, t)
        for k, act in enumerate(acts):
            n2 = self._apply(2, act, o2, t)
            if phase == "pre":
                child = self._new(node, self.depth[node] + 1, 0, 0, 2, 1.0, k)
                self._chance(child, t + 1, sig, a1, a2 + (act,), n1, n2, exo)
            else:
                child = self._new(node, self.depth[node] + 1, 2, 0, 2, 1.0, k)
                p, q1, q2 = cross_c(n1, n2, X, exo, self.K)
                u1 = -(p - 100) * q1 - self.lam * (1 - q1) ** 2
                u2 = -(p - 100) * q2 - self.lam * (1 - q2) ** 2
                self.u1[child], self.u2[child], self.payoff[child] = u1, u2, u1 - u2
                self.hist[child] = a1
                self.price[child], self.fill1[child] = p, q1

    def _finalize(self):
        for name in ["parent", "depth", "kind", "player", "ekind", "eprob", "eact", "payoff", "nact", "u1", "u2",
                     "price", "fill1"]:
            setattr(self, name, np.array(getattr(self, name)))
        self.M = len(self.parent)
        self.dec = np.where(self.kind == 1)[0]
        self.term = np.where(self.kind == 2)[0]
        self.maxd = int(self.depth.max())
        self.levels = [np.where(self.depth == d)[0] for d in range(self.maxd + 1)]
        self.child_of = -np.ones((self.M, 3), dtype=np.int64)
        e = np.where(self.ekind >= 1)[0]
        self.child_of[self.parent[e], self.eact[e]] = e
        self.pedges = {i: np.where(self.ekind == i)[0] for i in (1, 2)}
        self.cedges = np.where(self.ekind == 0)[0]
        # cycle indicator on terminals: player 1 cancelled at least once
        self.cycled = np.zeros(self.M)
        for n in self.term:
            self.cycled[n] = 1.0 if "cancel" in self.hist[n] else 0.0

    def ids(self, variant):
        table, ids, legal = {}, -np.ones(self.M, dtype=np.int64), []
        for n in self.dec:
            k = self.keys[n][variant]
            if k not in table:
                table[k] = len(table); legal.append(self.nact[n])
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
