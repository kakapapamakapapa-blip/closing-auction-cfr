"""Tables for H1 and H2 from out/cancel_*.json (player 1; CFR equilibria found are symmetric, and
player 2's numbers are reported where they differ)."""
import glob
import json
from collections import defaultdict

R = []
for p in sorted(glob.glob("/home/claude/exp/hypotheses/out/cancel_*.json")):
    R += json.load(open(p))
for r in R:
    r["players"] = {int(k): v for k, v in r["players"].items()}


def key(r):
    return (r["K"], r["q"], r["lam"], r["persist"], r["slope"], r["alloc"])


D = {}
for r in R:
    D[key(r)] = r
asym = [(key(r), r["players"][1]["pv_exact"], r["players"][2]["pv_exact"]) for r in R
        if abs(r["players"][1]["pv_exact"] - r["players"][2]["pv_exact"]) > 1e-3]
print(f"records {len(R)}; max NashConv {max(r['nashconv'] for r in R):.2e}; "
      f"median {sorted(r['nashconv'] for r in R)[len(R)//2]:.2e}; asymmetric pv (>1e-3): {len(asym)}")
for a in asym:
    print("  asym", a)
print(f"total CFR seconds {sum(r['seconds'] for r in R):.0f}")

print("\nH1: pv (exact private value of the right to cancel, P1) / p_cancel, time priority")
print("K  q    lam  | persist 0.5 .. 0.9 (slope 0.6)                                     | flat p.5  flat p.9 | slope .3   slope .9 (p .7) | monotone in persist?")
settings = sorted({(r["K"], r["q"], r["lam"]) for r in R if r["tag"] == "H1_persist"})
mono_up = mono_down = other = 0
for K, q, lam in settings:
    row, pvs = [], []
    for p in (0.5, 0.6, 0.7, 0.8, 0.9):
        r = D.get((K, q, lam, p, 0.6, "time"))
        if r:
            pvs.append(r["players"][1]["pv_exact"])
            row.append(f"{r['players'][1]['pv_exact']:.4f}/{r['players'][1]['p_cancel']:.2f}")
        else:
            row.append("    -     ")
    fl = [D.get((K, q, lam, p, 0.0, "time")) for p in (0.5, 0.9)]
    sl = [D.get((K, q, lam, 0.7, s, "time")) for s in (0.3, 0.9)]
    fmt = lambda r: f"{r['players'][1]['pv_exact']:.4f}/{r['players'][1]['p_cancel']:.2f}" if r else "    -    "
    tol = 2e-4
    up = all(b >= a - tol for a, b in zip(pvs, pvs[1:]))
    down = all(b <= a + tol for a, b in zip(pvs, pvs[1:]))
    tag = "flat" if (up and down) else ("up" if up else ("down" if down else "non-monotone"))
    mono_up += tag == "up"; mono_down += tag == "down"; other += tag not in ("up", "down")
    print(f"{K}  {q:.1f}  {lam:.2f} | {' '.join(row)} | {fmt(fl[0])} {fmt(fl[1])} | {fmt(sl[0])} {fmt(sl[1])} | {tag}")
print(f"monotone up {mono_up}, down {mono_down}, other {other}")

print("\nH1 by K at equal settings (persist 0.7, slope 0.6): pv K=2 -> K=3")
for (K, q, lam) in settings:
    if K == 3:
        a, b = D.get((2, q, lam, 0.7, 0.6, "time")), D.get((3, q, lam, 0.7, 0.6, "time"))
        if a and b:
            print(f"  q={q} lam={lam}: {a['players'][1]['pv_exact']:.4f} -> {b['players'][1]['pv_exact']:.4f}")

print("\nH2: time priority vs pro rata (persist 0.7 unless stated): pv / p_cancel / E#pre-cutoff submissions / u1 / NashConv")
for K in (2, 3):
    for q in (0.3, 0.6):
        for lam in (0.3, 0.45, 0.6, 0.75):
            for p in (0.5, 0.7, 0.9):
                a, b = D.get((K, q, lam, p, 0.6, "time")), D.get((K, q, lam, p, 0.6, "prorata"))
                if not (a and b):
                    continue
                f = lambda r: (f"{r['players'][1]['pv_exact']:.4f} / {r['players'][1]['p_cancel']:.3f} / "
                               f"{r['players'][1]['n_submit_pre']:.2f} / {r['u1']:.4f} / {r['nashconv']:.1e}")
                st = lambda r: {k: round(v, 3) for k, v in r['players'][1].get('submit_by_t', {}).items()}
                eq = ""
                if "equilibrium_value_p1" in b:
                    eq = f" | eq.value of cancel right: time {a.get('equilibrium_value_p1', float('nan')):+.4f} prorata {b['equilibrium_value_p1']:+.4f}"
                print(f"K={K} q={q} lam={lam} p={p}: TIME {f(a)} {st(a)} || PRORATA {f(b)} {st(b)}{eq}")
