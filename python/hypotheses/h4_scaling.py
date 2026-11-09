"""H4: relate phi_H's MCCFR node-visit speedup to its table compression ratio, using only the stored
results (mccfr_stats.json and the per-seed curves mccfr_K*_seeds10.json; no new MCCFR runs).

  c_K = entries(full) / entries(H)            (table compression)
  s_K = visits_full(eps_K) / visits_H(eps_K)  (speedup to the full game's final median accuracy)
Candidate laws: s = 1, s = c, s = c^2 (worst-case-bound ratio), s = c - 1, s = a c, s = c^alpha.
For each law: is it inside all three bootstrap 95% CIs? For one-parameter laws: the set of
parameter values consistent with all three CIs, and a seed-bootstrap distribution of alpha.
Also: the speedup at several accuracy targets along the median curves (stability check).
"""
import json
import math
import sys
sys.path.insert(0, "/home/claude/exp")
import numpy as np
from mccfr_stats import crossing as _crossing


def crossing(xs, ys, eps):
    try:
        return _crossing(xs, ys, eps)
    except OverflowError:                 # near-flat last segment: target effectively never reached
        return float("inf")

st = json.load(open("/home/claude/exp/mccfr_stats.json"))
rows = []
for r in st:
    c = r["entries"]["full"] / r["entries"]["H"]
    rows.append({"K": r["K"], "c": c, "s": r["ratio"], "lo": r["ratio_ci"][0], "hi": r["ratio_ci"][1],
                 "M": r["M"], "target": r["target"]})
print("K  compression c  speedup s  95% CI          s/c    log s / log c")
for x in rows:
    print(f"{x['K']}  {x['c']:.3f}         {x['s']:.3f}      [{x['lo']:.2f}, {x['hi']:.2f}]   {x['s']/x['c']:.3f}  "
          f"{math.log(x['s'])/math.log(x['c']):+.3f}")

laws = {"s = 1": lambda c: 1.0, "s = c": lambda c: c, "s = c^2": lambda c: c * c, "s = c - 1": lambda c: c - 1,
        "s = sqrt(c)": lambda c: math.sqrt(c)}
print("\nzero-parameter laws inside all three CIs?")
for name, g in laws.items():
    ins = [x["lo"] <= g(x["c"]) <= x["hi"] for x in rows]
    print(f"  {name:12s} predictions {[round(g(x['c']), 2) for x in rows]}  inside {ins}  -> {'consistent' if all(ins) else 'rejected'}")


def feasible(fun):
    lo, hi = -np.inf, np.inf
    for x in rows:
        a, b = fun(x)
        lo, hi = max(lo, a), min(hi, b)
    return lo, hi


lc = lambda x: math.log(x["c"])
alpha = feasible(lambda x: ((math.log(x["lo"]) / lc(x)) if x["lo"] > 0 else -np.inf, math.log(x["hi"]) / lc(x)))
a_lin = feasible(lambda x: (x["lo"] / x["c"], x["hi"] / x["c"]))
print(f"\none-parameter laws: s = c^alpha consistent for alpha in [{alpha[0]:.3f}, {alpha[1]:.3f}];  "
      f"s = a c consistent for a in [{a_lin[0]:.3f}, {a_lin[1]:.3f}]")
X = np.array([math.log(x["c"]) for x in rows]); Y = np.array([math.log(x["s"]) for x in rows])
print(f"least squares through the origin on log-log: all three points alpha = {float(X @ Y / (X @ X)):.3f}; "
      f"K = 3, 4 only alpha = {float(X[1:] @ Y[1:] / (X[1:] @ X[1:])):.3f}")
b, a0 = np.polyfit(X, Y, 1)
print(f"free-intercept log-log fit (2 parameters, 1 residual dof): log s = {a0:.3f} + {b:.3f} log c")

# seed bootstrap of alpha from the stored per-seed curves (same crossing rule as mccfr_stats.py)
rng = np.random.default_rng(1)
cur = {}
for x in rows:
    d = json.load(open(f"/home/claude/exp/mccfr_K{x['K']}_seeds10.json"))
    V = {}
    for v in ("full", "H"):
        runs = [r for r in d["runs"] if r["variant"] == v]
        V[v] = (np.array([c["visits"] for c in runs[0]["curve"]], float),
                np.array([[c["nc"] for c in r["curve"]] for r in runs]))
    cur[x["K"]] = V
B = 4000
alphas, alphas34, dropped = [], [], 0
for _ in range(B):
    ls = []
    for x in rows:
        V = cur[x["K"]]
        mb = {v: np.median(V[v][1][rng.integers(0, V[v][1].shape[0], V[v][1].shape[0])], axis=0) for v in V}
        eps = float(np.median(V["full"][1], axis=0)[-1])
        cf, ch = crossing(V["full"][0], mb["full"], eps), crossing(V["H"][0], mb["H"], eps)
        s = cf / ch if (np.isfinite(cf) and np.isfinite(ch) and ch > 0) else (0.0 if np.isinf(ch) else np.nan)
        ls.append(s)
    if all(np.isfinite(v) and v > 0 for v in ls[1:]):
        y = np.log(ls[1:]); alphas34.append(float(X[1:] @ y / (X[1:] @ X[1:])))
    if all(np.isfinite(v) and v > 0 for v in ls):
        y = np.log(ls); alphas.append(float(X @ y / (X @ X)))
    else:
        dropped += 1
q = lambda a: (float(np.percentile(a, 2.5)), float(np.median(a)), float(np.percentile(a, 97.5)))
print(f"\nseed bootstrap ({B} reps): alpha (K = 3, 4) 2.5/50/97.5% = {tuple(round(v, 3) for v in q(alphas34))}; "
      f"alpha (K = 2, 3, 4) = {tuple(round(v, 3) for v in q(alphas))} with {dropped} reps dropped (K = 2 ratio 0 or undefined)")

print("\nspeedup along the median curves at other accuracy targets (visits_full / visits_H; interpolated, no extrapolation)")
for r in st:
    xf, yf = np.array(r["curve"]["full"]["visits"]), np.array(r["curve"]["full"]["median"])
    xh, yh = np.array(r["curve"]["H"]["visits"]), np.array(r["curve"]["H"]["median"])
    out = []
    for eps in np.geomspace(max(yf.min(), yh.min()) * 1.0001, min(yf.max(), yh.max()) * 0.9999, 6):
        cf, ch = crossing(xf, yf, eps), crossing(xh, yh, eps)
        if cf <= xf[-1] and ch <= xh[-1]:
            out.append((round(float(eps), 4), round(cf / ch, 2)))
    print(f"  K={r['K']} c={r['entries']['full'] / r['entries']['H']:.2f}: {out}")
