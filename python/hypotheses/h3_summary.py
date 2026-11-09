"""Summary of H3 CFR runs (out/disclose_cfr.json) plus the stored sweep_40.json disclosure data."""
import json
import numpy as np
C = json.load(open("/home/claude/exp/hypotheses/out/disclose_cfr.json"))
L = {r["name"]: r for r in json.load(open("/home/claude/exp/hypotheses/out/disclose_lp.json"))}
print("name        | G0 hidden [v_lo, v_hi]      | G0 disclose [v_lo, v_hi]   | LP v*      | overlap | NC_H/NC_full hid, dis | native u1 hid -> dis (NC_full hid/dis; NC_H dis)")
rat_h, rat_d, du, ov = [], [], [], 0
for r in C:
    h, d = r["g0_hid_full"], r["g0_dis_full"]
    o = max(h["v_lo"], d["v_lo"]) <= min(h["v_hi"], d["v_hi"]); ov += o
    rh = r["g0_hid_H"]["nashconv"] / h["nashconv"]; rd = r["g0_dis_H"]["nashconv"] / d["nashconv"]
    rat_h.append(rh); rat_d.append(rd)
    nh, nd = r["native_hid_full"], r["native_dis_full"]
    du.append(max(abs(nd["u1"] - nh["u1"]), abs(nd["u2"] - nh["u2"])))
    print(f"{r['name']:11s} | [{h['v_lo']:.6f}, {h['v_hi']:.6f}] | [{d['v_lo']:.6f}, {d['v_hi']:.6f}] | {L[r['name']]['v_hid_c']:.6f} | {o} | "
          f"{rh:.2f}, {rd:.2f} | {nh['u1']:.6f} -> {nd['u1']:.6f} ({nh['nashconv']:.1e}/{nd['nashconv']:.1e}; {r['native_dis_H']['nashconv']:.1e})")
print(f"overlap {ov}/{len(C)}; G0 NC_H/NC_full hidden: max {max(rat_h):.4f}; disclose: median {np.median(rat_d):.2f} max {max(rat_d):.2f}, "
      f">1.1 in {sum(x > 1.1 for x in rat_d)}/{len(C)}; max native payoff change {max(du):.2e}")
nat_rd = [r["native_dis_H"]["nashconv"] / r["native_dis_full"]["nashconv"] for r in C]
print(f"native NC_H/NC_full under disclosure: median {np.median(nat_rd):.2f} max {max(nat_rd):.2f}")
S = json.load(open("/home/claude/exp/sweep_40.json"))
rd = np.array([s["disclose_gamma0"]["H"]["nc_table"] / s["disclose_gamma0"]["full"]["nc_table"] for s in S])
rh = np.array([s["gamma0"]["H"]["nc_table"] / s["gamma0"]["full"]["nc_table"] for s in S])
print(f"sweep_40 (instance B): G0 NC_H/NC_full hidden max {rh.max():.6f}; disclose median {np.median(rd):.2f}, >1.1 {int((rd > 1.1).sum())}/40, "
      f">2 {int((rd > 2).sum())}/40, >5 {int((rd > 5).sum())}/40, max {rd.max():.2f}")
w = [(s["disclose_gamma0"]["H"]["v_hi"] - s["disclose_gamma0"]["H"]["v_lo"]) for s in S]
inside = sum(s["disclose_gamma0"]["H"]["v_lo"] - 1e-9 <= L[f"B_sweep{s['seed']}"]["v_dis_c"] <= s["disclose_gamma0"]["H"]["v_hi"] + 1e-9 for s in S)
print(f"sweep_40: phi_H (disclose) certified intervals contain the exact value in {inside}/40 (median width {np.median(w):.2e})")
