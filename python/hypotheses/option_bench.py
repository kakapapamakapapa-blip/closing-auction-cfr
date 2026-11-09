"""Single-agent option benchmark for instance C (no strategic rival; exogenous MOC arrivals w.p. q per period).
Payoff relative to the no-fill baseline -lam: holding an order stamped tau that is first in line gives M(f)=0.9*lam-P(bad|f);
P(first | stamp tau) = (1-q)^tau. Post-cutoff entry (stamp K) fills only if chi=-1 (no arrivals, supply exists)."""
import itertools, sys
import numpy as np

def paths(K, p):
    out = []
    for s in itertools.product((1, 0), repeat=K):  # 1=U
        pr = 0.5
        for a, b in zip(s, s[1:]):
            pr *= p if a == b else 1 - p
        out.append((s, pr))
    return out

def M(nU, K, lam):
    return 0.9 * lam - (0.15 + 0.6 * nU / K)

def bench(K, p, q, lam):
    P = paths(K, p)
    # with cancel right: submit at 0, decide at K-1 with all signals -> E[M^+]
    EMp = sum(pr * max(0.0, M(sum(s), K, lam)) for s, pr in P)
    # without cancel right: optimal stopping over submission period tau in {0..K-1} or post-cutoff (K)
    def V(t, prefix):
        cont = [(s, pr) for s, pr in P if s[:t + 1] == prefix]
        tot = sum(pr for _, pr in cont)
        EM = sum(pr * M(sum(s), K, lam) for s, pr in cont) / tot
        submit = (1 - q) ** t * EM
        if t == K - 1:
            wait = (1 - q) ** K * max(0.0, EM)  # f known at K-1: post-cutoff entry value
            return max(submit, wait)
        nxt = 0.0
        for b in (1, 0):
            sub = [(s, pr) for s, pr in cont if s[t + 1] == b]
            w = sum(pr for _, pr in sub)
            if w > 0:
                nxt += w / tot * V(t + 1, prefix + (b,))
        return max(submit, nxt)
    Vnc = sum(0.5 * V(0, (b,)) for b in (1, 0))
    # time-value bound (q->1): commit after s0 or never
    tv = 0.0
    for b in (1, 0):
        cont = [(s, pr) for s, pr in P if s[0] == b]
        tot = sum(pr for _, pr in cont)
        EM0 = sum(pr * M(sum(s), K, lam) for s, pr in cont) / tot
        EMp0 = sum(pr * max(0.0, M(sum(s), K, lam)) for s, pr in cont) / tot
        tv += 0.5 * (EMp0 - max(0.0, EM0))
    # residual sd of terminal P(bad) after s0
    var = 0.0
    for b in (1, 0):
        cont = [(s, pr) for s, pr in P if s[0] == b]
        tot = sum(pr for _, pr in cont)
        m1 = sum(pr * 0.6 * sum(s) / K for s, pr in cont) / tot
        m2 = sum(pr * (0.6 * sum(s) / K) ** 2 for s, pr in cont) / tot
        var += 0.5 * (m2 - m1 ** 2)
    return EMp, Vnc, EMp - Vnc, tv, var ** 0.5

print("Thesis grid, K=3, persistence 0.7 (benchmark value of right vs thesis private value)")
thesis = {(0.3,0.3):0.031,(0.3,0.45):0.016,(0.3,0.6):0.000,(0.3,0.75):0.029,(0.6,0.3):0.031,(0.6,0.45):0.029,(0.6,0.6):0.030,(0.6,0.75):0.020}
print(f"{'q':>4} {'lam':>5} {'f*':>6} {'E[M+]':>7} {'Vnc':>7} {'right':>7} {'TVbound':>8} {'thesis':>7}")
for q in (0.3, 0.6):
    for lam in (0.3, 0.45, 0.6, 0.75):
        EMp, Vnc, val, tv, sd = bench(3, 0.7, q, lam)
        fstar = (0.9 * lam - 0.15) / 0.6
        print(f"{q:4.1f} {lam:5.2f} {fstar:6.3f} {EMp:7.4f} {Vnc:7.4f} {val:7.4f} {tv:8.4f} {thesis[(q,lam)]:7.3f}")

print("\nK=2 grid, persistence 0.7")
for q in (0.3, 0.6):
    for lam in (0.3, 0.45, 0.6, 0.75):
        EMp, Vnc, val, tv, sd = bench(2, 0.7, q, lam)
        print(f"q={q} lam={lam}: right={val:.4f} TVbound={tv:.4f}")

print("\nResidual sd of terminal P(bad) after s0, and benchmark value of right (q=0.6, lam=0.45), by persistence and K")
for K in (2, 3, 4, 5):
    row = []
    for p in (0.5, 0.6, 0.7, 0.8, 0.9, 0.97):
        EMp, Vnc, val, tv, sd = bench(K, p, 0.6, 0.45)
        row.append(f"p={p}: sd={sd:.3f} v={val:.4f}")
    print(f"K={K}: " + " | ".join(row))
print("\nq sweep (K=3, p=0.7, lam=0.45): value of right")
print(" ".join(f"q={q}:{bench(3,0.7,q,0.45)[2]:.4f}" for q in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)))
