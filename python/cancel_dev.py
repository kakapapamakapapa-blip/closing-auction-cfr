"""Private versus equilibrium value of the right to cancel (instance C).

For each setting:
  sigma_c  = CFR solution of the game with cancellation (compiled tables, full game);
  sigma_nc = CFR solution of the game without cancellation.
Private value of cancelling at sigma_nc: player 1's best-response gain in the cancel game against
  sigma_nc (embedded, cancel probability 0) - is sigma_nc still an equilibrium once cancelling is allowed?
Private value at sigma_c: u_1(sigma_c) minus player 1's best response in the cancel game restricted
  to strategies that never cancel, against sigma_c.
Equilibrium value: u_1(sigma_c) - u_1(sigma_nc).
Also reports who receives the scarce unit at 100 and how often strategic buyers are filled at 101.
"""
import json
import sys
import numpy as np
from auction_c import TreeC
from general_sum import GSolver


def train(T, N):
    S = GSolver(T, "full"); S.setup()
    for _ in range(N):
        S.step()
    return S


def br_masked(S, i, f, allow):
    """Value of player i's perfect-recall best response against edge probs f, using only actions
    where allow[bucket-of-full-game, action] is True."""
    T = S.T
    q = np.ones(T.M)
    for d in range(1, T.maxd + 1):
        idx = T.levels[d]
        q[idx] = q[T.parent[idx]] * np.where(T.ekind[idx] == i, 1.0, f[idx])
    val = np.where(T.kind == 2, S.U[i], 0.0)
    for d in range(T.maxd - 1, -1, -1):
        idx = T.levels[d]
        idx = idx[T.kind[idx] != 2]
        mine = idx[(T.kind[idx] == 1) & (T.player[idx] == i)]
        other = idx[~((T.kind[idx] == 1) & (T.player[idx] == i))]
        kids = T.levels[d + 1]
        kp = T.parent[kids]
        sel = np.isin(kp, other)
        np.add.at(val, kp[sel], f[kids[sel]] * val[kids[sel]])
        if len(mine):
            Q = np.zeros((S.nf, S.fmask.shape[1]))
            selm = np.isin(kp, mine)
            km = kids[selm]
            np.add.at(Q, (S.fid[kp[selm]], T.eact[km]), q[kp[selm]] * val[km])
            best = np.where(S.fmask & allow, Q, -np.inf).argmax(1)
            val[mine] = val[T.child_of[mine, best[S.fid[mine]]]]
    return float(val[0])


def outcomes(S, f):
    T = S.T
    ev = lambda x: float(S.values_p(f, x.astype(float))[0])
    good_any = (T.price == 100)
    return {"p_unit_at_100_to_p1": ev(T.fill1 * good_any),
            "p_fill_at_101_p1": ev(T.fill1 * (T.price == 101)),
            "p_cancel_p1": ev(T.cycled)}


def analyse(K, q, lam, N):
    Tc = TreeC(K, q=q, lam=lam, cancel=True)
    Tn = TreeC(K, q=q, lam=lam, cancel=False)
    Sc, Sn = train(Tc, N), train(Tn, N)
    tc, ic = Sc.compiled(); uc, gc, fc = Sc.evaluate(tc, ic)
    tn, inn = Sn.compiled(); un, gn, fn = Sn.evaluate(tn, inn)
    # embed sigma_nc into the cancel game by key
    nkey = {}
    for n in Tn.dec:
        nkey[Tn.keys[n]["full"]] = (inn[n], Tn.nact[n])
    emb = np.zeros_like(tc)
    for n in Tc.dec:
        k = Tc.keys[n]["full"]
        b = ic[n]
        if k in nkey:
            bn, na = nkey[k]
            row = tn[bn]
            emb[b, :] = 0.0
            emb[b, 0] = row[0]
            if Tc.nact[n] == 2 and na == 2:             # empty book: noop / moc in both games
                emb[b, 1] = row[1]
        else:                                           # only reachable after a cancel
            emb[b, :Tc.nact[n]] = 1.0 / Tc.nact[n]
    fe = Sc.edge_factors(emb, ic)
    u_emb = float(Sc.values_p(fe, Sc.U[1])[0])
    allow_all = np.ones_like(Sc.fmask)
    br_emb = br_masked(Sc, 1, fe, allow_all)
    # restrict player 1 to never cancel: at full-game buckets whose node has a resting order pre-cutoff,
    # forbid action 1 (cancel)
    allow_nc = np.ones_like(Sc.fmask)
    # resting-order nodes before the cutoff are those with an order in the H key and two actions
    for n in Tc.dec:
        if Tc.player[n] == 1 and Tc.keys[n]["H"][3] is not None and Tc.nact[n] == 2:
            allow_nc[Sc.fid[n], 1] = False
    br_c_nocancel = br_masked(Sc, 1, fc, allow_nc)
    rec = {"K": K, "q": q, "lam": lam, "N": N,
           "u1_cancel": uc[1], "u1_nocancel": un[1], "equilibrium_value": uc[1] - un[1],
           "nashconv_cancel": sum(gc.values()), "nashconv_nocancel": sum(gn.values()),
           "u1_embedded_check": u_emb,
           "private_value_at_nc": br_emb - u_emb,
           "private_value_at_c": uc[1] - br_c_nocancel,
           "outcomes_cancel": outcomes(Sc, fc), "outcomes_nocancel": outcomes(Sn, fn)}
    return rec


if __name__ == "__main__":
    K, N = int(sys.argv[1]), int(sys.argv[2])
    grid = json.loads(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    out = []
    for g in grid:
        rec = analyse(K, g["q"], g["lam"], N)
        out.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(out, open(f"cancel_dev_K{K}{tag}.json", "w"), indent=1)
