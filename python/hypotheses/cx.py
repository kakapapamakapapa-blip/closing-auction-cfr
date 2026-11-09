"""Instance C variants for hypothesis screening (auction_c.py itself is not modified).

TreeCX(K, q, lam, cancel, persist, slope=0.6, alloc="time")
  slope : sensitivity of the closing-state law to the share f of U signals, centred so that the
          marginal law of Z is the same for every slope and persist (E f = 1/2 for the symmetric
          chain started from the uniform law):
              P(good) = 0.45 - slope (f - 1/2),  P(none) = 0.10,  P(bad) = 0.45 + slope (f - 1/2).
          slope = 0.6 reproduces auction_c.z_probs; slope = 0 makes every signal uninformative.
  alloc : "time"    -> price-time priority (auction_c.cross_c, unchanged);
          "prorata" -> all eligible MOC buy orders share the executed volume in proportion to size,
                       whatever their time stamps (exogenous arrivals included).

Helpers: training with GSolver, exact private value of the right to cancel, cancel / submission
statistics. All best responses are perfect-recall best responses in the full tree.
"""
import sys
import time
sys.path.insert(0, "/home/claude/exp")
import numpy as np
from auction import _alloc
from auction_c import TreeC, PRICES, MOC, chi_c, cross_c, keys_for, z_probs
from general_sum import GSolver
from cancel_dev import br_masked


def z_law(n_up, K, slope):
    if slope == 0.6:                    # bit-identical to the original instance C
        return z_probs(n_up, K)
    f = n_up / K
    return {"good": 0.45 - slope * (f - 0.5), "none": 0.10, "bad": 0.45 + slope * (f - 0.5)}


def cross_prorata(o1, o2, Z, exo):
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
    key = lambda b: (-b[0],)            # no time component: one pro-rata group per limit
    return p, _alloc(elig, V, key, 1), _alloc(elig, V, key, 2)


class TreeCX(TreeC):
    def __init__(self, K, q=0.4, lam=0.6, cancel=True, persist=0.7, slope=0.6, alloc="time"):
        assert alloc in ("time", "prorata") and 0.0 <= slope <= 0.9
        self.slope, self.alloc = slope, alloc
        super().__init__(K, q=q, lam=lam, cancel=cancel, persist=persist)

    def _chance(self, node, t, sig, a1, a2, o1, o2, exo):
        K, q = self.K, self.q
        arrivals = [(False, 1.0)] if t == 0 else [(False, 1 - q), (True, q)]
        for arr, pa in arrivals:
            if pa <= 0:
                continue
            exo2 = exo + ((t - 1) + 0.5,) if arr else exo
            if t < K:
                prev = sig[-1] if sig else None
                for s, ps in self._sig(prev).items():
                    if ps <= 0:
                        continue
                    child = self._new(node, self.depth[node] + 1, 1, 1, 0, pa * ps, -1)
                    self._p1(child, t, sig + (s,), a1, a2, o1, o2, "pre", None, exo2, None)
            else:
                n_up = sum(s == "U" for s in sig)
                for X, px in z_law(n_up, K, self.slope).items():
                    if px <= 0:
                        continue
                    chi = chi_c(o1, o2, X, exo2)
                    child = self._new(node, self.depth[node] + 1, 1, 1, 0, pa * px, -1)
                    self._p1(child, K, sig, a1, a2, o1, o2, "post", chi, exo2, X)

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
                if self.alloc == "time":
                    p, q1, q2 = cross_c(n1, n2, X, exo, self.K)
                else:
                    p, q1, q2 = cross_prorata(n1, n2, X, exo)
                u1 = -(p - 100) * q1 - self.lam * (1 - q1) ** 2
                u2 = -(p - 100) * q2 - self.lam * (1 - q2) ** 2
                self.u1[child], self.u2[child], self.payoff[child] = u1, u2, u1 - u2
                self.hist[child] = a1
                self.price[child], self.fill1[child] = p, q1


def train(T, variant, N):
    S = GSolver(T, variant); S.setup()
    for _ in range(N):
        S.step()
    return S


def cancel_stats(T, S, f):
    """Per player: exact private value of the right to cancel against the opponent's fixed strategy
    (perfect-recall BR with all actions minus BR that never cancels), u_i, the expected number of
    pre-cutoff cancellations and of pre-/post-cutoff MOC submissions (for K <= 3 an order can be
    cancelled at most once, so the expected count is the probability of cancelling)."""
    pi = S.reaches(f)
    reach = pi[0] * pi[1] * pi[2]
    out = {}
    for i in (1, 2):
        mine = [n for n in T.dec if T.player[n] == i]
        can = [n for n in mine if T.keys[n]["H"][3] is not None and T.keys[n]["H"][1] < T.K and T.nact[n] == 2]
        sub_pre = [n for n in mine if T.keys[n]["H"][3] is None and T.keys[n]["H"][1] < T.K]
        sub_post = [n for n in mine if T.keys[n]["H"][3] is None and T.keys[n]["H"][1] == T.K]
        allow_all = np.ones_like(S.fmask)
        allow_nc = np.ones_like(S.fmask)
        for n in can:
            allow_nc[S.fid[n], 1] = False
        br_all = br_masked(S, i, f, allow_all)
        br_nc = br_masked(S, i, f, allow_nc)
        u = float(S.values_p(f, S.U[i])[0])
        ex = lambda nodes: float(sum(reach[n] * f[T.child_of[n, 1]] for n in nodes))
        by_t = {}
        for n in sub_pre + sub_post:
            t = int(T.keys[n]["H"][1])
            by_t[t] = by_t.get(t, 0.0) + float(reach[n] * f[T.child_of[n, 1]])
        out[i] = {"u": u, "br_all": br_all, "br_nocancel": br_nc, "submit_by_t": by_t,
                  "pv_exact": br_all - br_nc,          # >= 0 by construction, exact given opponent
                  "pv_u": u - br_nc,                   # cancel_dev.py's private_value_at_c definition
                  "p_cancel": ex(can), "n_submit_pre": ex(sub_pre), "n_submit_post": ex(sub_post)}
    return out


def solve_and_measure(K, q, lam, persist=0.7, slope=0.6, alloc="time", N=2000, nocancel=False):
    t0 = time.time()
    T = TreeCX(K, q=q, lam=lam, cancel=True, persist=persist, slope=slope, alloc=alloc)
    S = train(T, "full", N)
    tab, ids = S.compiled()
    u, gaps, f = S.evaluate(tab, ids)
    rec = {"K": K, "q": q, "lam": lam, "persist": persist, "slope": slope, "alloc": alloc, "N": N,
           "nodes": int(T.M), "nashconv": float(sum(gaps.values())), "u1": u[1], "u2": u[2],
           "players": cancel_stats(T, S, f)}
    if nocancel:
        Tn = TreeCX(K, q=q, lam=lam, cancel=False, persist=persist, slope=slope, alloc=alloc)
        Sn = train(Tn, "full", N)
        tn, inn = Sn.compiled()
        un, gn, _ = Sn.evaluate(tn, inn)
        rec.update({"u1_nocancel": un[1], "u2_nocancel": un[2], "nashconv_nocancel": float(sum(gn.values())),
                    "equilibrium_value_p1": u[1] - un[1]})
    rec["seconds"] = time.time() - t0
    return rec
