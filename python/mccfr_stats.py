"""Statistics for the 10-seed MCCFR runs.

For each K and variant, take the median exploitability across seeds at every checkpoint (node-visit
budget). Report (i) the median exploitability at the final budget with a bootstrap 95% CI, and
(ii) the node visits at which phi_H's median curve first reaches the final median accuracy of the full
game, with a bootstrap 95% CI for the ratio of visits (full / phi_H). Bootstrap resamples seeds within
each variant; crossings are interpolated on log-log scales and extrapolated along the last segment
when a resampled curve does not reach the target.
"""
import json, math, sys
import numpy as np

rng = np.random.default_rng(0)


def crossing(xs, ys, eps):
    lx, ly = np.log(xs), np.log(np.maximum(ys, 1e-300))
    for j in range(len(xs)):
        if ys[j] <= eps:
            if j == 0:
                return float(xs[0])
            w = (math.log(eps) - ly[j - 1]) / (ly[j] - ly[j - 1])
            return float(math.exp(lx[j - 1] + w * (lx[j] - lx[j - 1])))
    slope = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
    if slope >= 0:
        return float("inf")
    return float(math.exp(lx[-1] + (math.log(eps) - ly[-1]) / slope))


def analyse(path, B=4000):
    d = json.load(open(path))
    V = {}
    for v in ("full", "H"):
        runs = [r for r in d["runs"] if r["variant"] == v]
        V[v] = {"x": np.array([c["visits"] for c in runs[0]["curve"]], float),
                "s": np.array([[c["seconds"] for c in r["curve"]] for r in runs]),
                "y": np.array([[c["nc"] for c in r["curve"]] for r in runs]),
                "entries": runs[0].get("entries")}
    med = {v: np.median(V[v]["y"], axis=0) for v in V}
    eps = float(med["full"][-1])
    vis = {v: crossing(V[v]["x"], med[v], eps) for v in V}
    final = {v: float(med[v][-1]) for v in V}
    boot_ratio, boot_final = [], {v: [] for v in V}
    n = {v: V[v]["y"].shape[0] for v in V}
    for _ in range(B):
        mb = {v: np.median(V[v]["y"][rng.integers(0, n[v], n[v])], axis=0) for v in V}
        cf, ch = crossing(V["full"]["x"], mb["full"], eps), crossing(V["H"]["x"], mb["H"], eps)
        if math.isinf(cf) and math.isinf(ch):
            boot_ratio.append(float("nan"))
        elif math.isinf(ch):
            boot_ratio.append(0.0)
        else:
            boot_ratio.append(cf / ch)
        for v in V:
            boot_final[v].append(mb[v][-1])
    out = {"K": d["K"], "M": d["M"], "seeds": n, "entries": {v: V[v]["entries"] for v in V},
           "budget": float(V["full"]["x"][-1]), "target": eps,
           "visits_to_target": vis, "ratio": vis["full"] / vis["H"],
           "ratio_ci": [float(np.nanpercentile(boot_ratio, 2.5)), float(np.nanpercentile(boot_ratio, 97.5))],
           "final_median": final,
           "final_ci": {v: [float(np.percentile(boot_final[v], 2.5)), float(np.percentile(boot_final[v], 97.5))] for v in V},
           "sec_per_Mvisit_runs": {v: float(np.median(V[v]["s"][:, -1] / V[v]["x"][-1] * 1e6)) for v in V},
           "curve": {v: {"visits": V[v]["x"].tolist(), "median": med[v].tolist(),
                         "lo": np.percentile(V[v]["y"], 25, axis=0).tolist(),
                         "hi": np.percentile(V[v]["y"], 75, axis=0).tolist()} for v in V}}
    return out


if __name__ == "__main__":
    res = []
    for K in sys.argv[1:]:
        r = analyse(f"mccfr_K{K}_seeds10.json")
        res.append(r)
        print(f"K={r['K']} M={r['M']} entries={r['entries']} budget={r['budget']:.3g} target={r['target']:.4f}")
        print(f"  final median full {r['final_median']['full']:.4f} {r['final_ci']['full']}  H {r['final_median']['H']:.4f} {r['final_ci']['H']}")
        print(f"  visits to target: full {r['visits_to_target']['full']:.3g}  H {r['visits_to_target']['H']:.3g}  ratio {r['ratio']:.2f} CI {r['ratio_ci']}")
    json.dump(res, open("mccfr_stats.json", "w"), indent=1)
