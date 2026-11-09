"""Section 6.10: summary statistics of the random-instance sweep (sweep_40.json)."""
import json
import sys
import numpy as np

fn = sys.argv[1] if len(sys.argv) > 1 else "sweep_40.json"
D = json.load(open(fn))
n = len(D)
nc = lambda r, game, v, k="nc_table": r[game][v][k]
out = {"instances": n}

# (a) exactness of phi_H in both games
dH = [max(abs(nc(r, g, "H") - nc(r, g, "full")) for g in ("native", "gamma0")) for r in D]
out["max_abs_diff_H_vs_full_hidden"] = max(dH)

# relative exploitability of each merge: NashConv(variant) / NashConv(full) in the same game
def ratio(r, g, v):
    return nc(r, g, v) / nc(r, g, "full")

for g in ("native", "gamma0"):
    for v in ("H_s0m", "H_s0", "H_chi"):
        x = np.array([ratio(r, g, v) for r in D])
        out[f"ratio_{g}_{v}"] = {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.1)),
                                  "q90": float(np.quantile(x, 0.9)), "min": float(x.min()), "max": float(x.max()),
                                  "share_gt2": float((x > 2).mean()), "share_le1.1": float((x <= 1.1).mean())}

# (c) chi: free natively but costly in Gamma_0
chi_nat = np.array([ratio(r, "native", "H_chi") for r in D])
chi_g0 = np.array([ratio(r, "gamma0", "H_chi") for r in D])
out["chi_free_native_costly_gamma0"] = int(((chi_nat <= 1.1) & (chi_g0 > 2)).sum())
out["chi_costly_native"] = int((chi_nat > 2).sum())
out["chi_costly_gamma0"] = int((chi_g0 > 2).sum())
out["chi_more_costly_in_gamma0_than_native"] = int((chi_g0 > chi_nat).sum())

# (d) ranking of the two merges: is chi costlier than s0?
s0_nat = np.array([ratio(r, "native", "H_s0") for r in D])
s0_g0 = np.array([ratio(r, "gamma0", "H_s0") for r in D])
out["chi_costlier_than_s0_native"] = int((chi_nat > s0_nat * 1.1).sum())
out["chi_costlier_than_s0_gamma0"] = int((chi_g0 > s0_g0 * 1.1).sum())
out["s0_costly_native"] = int((s0_nat > 2).sum())
out["s0_costly_gamma0"] = int((s0_g0 > 2).sum())

# (e) deployment: table versus mixture
for g in ("native", "gamma0"):
    for v in ("full", "H", "H_s0", "H_chi", "H_s0m"):
        gaps = np.array([abs(nc(r, g, v) - nc(r, g, v, "nc_mixture")) for r in D])
        out[f"deploy_gap_{g}_{v}_max"] = float(gaps.max())
    rel = np.array([nc(r, g, "H_s0m") / nc(r, g, "H_s0m", "nc_mixture") for r in D])
    out[f"deploy_ratio_{g}_H_s0m"] = {"median": float(np.median(rel)), "max": float(rel.max()),
                                       "share_gt1.1": float((rel > 1.1).mean())}

# (b) disclosure regime: phi_H against the full game
dis = np.array([nc(r, "disclose_gamma0", "H") / nc(r, "disclose_gamma0", "full") for r in D])
out["disclose_ratio_H_full"] = {"median": float(np.median(dis)), "q10": float(np.quantile(dis, 0.1)),
                                "q90": float(np.quantile(dis, 0.9)), "min": float(dis.min()), "max": float(dis.max()),
                                "share_gt1.1": float((dis > 1.1).mean()), "share_gt2": float((dis > 2).mean()),
                                "share_gt5": float((dis > 5).mean())}

# (f) value of Gamma_0 in the two regimes (certified intervals of the full-game runs)
ov, gaps, widths = 0, [], []
for r in D:
    a, b = r["gamma0"]["full"], r["disclose_gamma0"]["full"]
    lo, hi = max(a["v_lo"], b["v_lo"]), min(a["v_hi"], b["v_hi"])
    ov += lo <= hi
    # largest possible difference between the two values
    gaps.append(max(abs(a["v_hi"] - b["v_lo"]), abs(b["v_hi"] - a["v_lo"])))
    widths.append(abs(0.5 * (a["v_lo"] + a["v_hi"])))
out["value_intervals_overlap"] = int(ov)
out["value_max_possible_diff"] = {"median": float(np.median(gaps)), "max": float(max(gaps))}
out["value_abs_median"] = float(np.median(widths))
out["nc_full_native_median"] = float(np.median([nc(r, "native", "full") for r in D]))
out["nc_full_gamma0_median"] = float(np.median([nc(r, "gamma0", "full") for r in D]))
out["nc_full_disclose_median"] = float(np.median([nc(r, "disclose_gamma0", "full") for r in D]))
print(json.dumps(out, indent=1))
json.dump(out, open("sweep_summary.json", "w"), indent=1)
