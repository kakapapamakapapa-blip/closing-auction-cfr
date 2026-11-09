"""Disclosure in instance C at K = 3 (native payoffs), on the three settings where it changed CFR payoffs
at K = 2: CFR in both regimes (full game, N = 2000) and the hidden equilibrium embedded in the
disclosure game. Usage: python3 h3c_k3.py -> out/h3c_k3.json"""
import json, sys, time
sys.path.insert(0, "/home/claude/exp/hypotheses"); sys.path.insert(0, "/home/claude/exp")
from cx import train, cancel_stats
from cx_disclose import TreeCD
from general_sum import GSolver
from h3_embed_lib import embed
out, t00 = [], time.time()
strip_c = lambda k: (k[0], k[1], tuple(s for s in k[2] if s in ("U", "D")), k[3], k[4])
for q, lam in ((0.3, 0.75), (0.6, 0.45), (0.6, 0.75)):
    t0 = time.time()
    Th, Td = TreeCD(3, q=q, lam=lam, disclose=False), TreeCD(3, q=q, lam=lam, disclose=True)
    Sh = train(Th, "full", 2000); th, ih = Sh.compiled(); uh, gh, fh = Sh.evaluate(th, ih)
    Sd = train(Td, "full", 2000); td, idd = Sd.compiled(); ud, gd, fd = Sd.evaluate(td, idd)
    f = embed(Th, th, ih, Td, Sd, strip_c)
    ue = {i: float(Sd.values_p(f, Sd.U[i])[0]) for i in (1, 2)}
    ge = {i: float(Sd.br_p(i, f) - ue[i]) for i in (1, 2)}
    ch, cd = cancel_stats(Th, Sh, fh), cancel_stats(Td, Sd, fd)
    rec = {"q": q, "lam": lam, "nodes_dis": int(Td.M), "u1_hid": uh[1], "u1_dis": ud[1], "u2_dis": ud[2],
           "nc_hid": float(sum(gh.values())), "nc_dis": float(sum(gd.values())), "nc_hidden_eq_in_dis_game": float(sum(ge.values())),
           "pv1_hid": ch[1]["pv_exact"], "pv1_dis": cd[1]["pv_exact"], "pc1_hid": ch[1]["p_cancel"], "pc1_dis": cd[1]["p_cancel"],
           "seconds": time.time() - t0}
    out.append(rec)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump(out, open("/home/claude/exp/hypotheses/out/h3c_k3.json", "w"), indent=1)
print(f"total {time.time() - t00:.1f}s")
