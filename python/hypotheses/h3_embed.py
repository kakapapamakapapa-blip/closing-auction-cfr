"""Does the hidden-regime CFR equilibrium survive disclosure?

Train CFR (full game, N = 2000) in the hidden regime, embed the compiled strategy into the disclosure
regime (players ignore the disclosed information), and compute its exact NashConv in the disclosure
game. Small NashConv: the old equilibrium is still an (approximate) equilibrium after disclosure, so
any payoff change found by CFR under disclosure is equilibrium selection, not a forced change.
  C : instance C, native payoffs (general sum), K = 2, 8 settings (q x lam), persist 0.7.
  AB: market.py instances in Gamma_0 (zero sum), K = 2: default A, default B, 10 instance-A draws.
Usage: python3 h3_embed.py -> out/h3_embed.json
"""
import json
import sys
import time
sys.path.insert(0, "/home/claude/exp/hypotheses"); sys.path.insert(0, "/home/claude/exp")
import numpy as np
from cx import train
from cx_disclose import TreeCD
from general_sum import GSolver
from cfr import Solver
from market import Market, Params
from sweep import draw


def embed(Th, tab, ids, Td, Sd, strip):
    """Edge factors on the disclosure tree Td from the hidden-regime table (tab, ids on Th)."""
    row = {Th.keys[n]["full"]: ids[n] for n in Th.dec}
    f = np.ones(Td.M)
    f[Td.cedges] = Td.eprob[Td.cedges]
    for i in (1, 2):
        e = Sd.pe[i]
        h = Sd.pp[i]
        r = np.array([row[strip(Td.keys[n]["full"])] for n in h])
        f[e] = tab[r, Td.eact[e]]
    return f


out = {"C": [], "AB": []}
t00 = time.time()
strip_c = lambda k: (k[0], k[1], tuple(s for s in k[2] if s in ("U", "D")), k[3], k[4])
for q in (0.3, 0.6):
    for lam in (0.3, 0.45, 0.6, 0.75):
        Th, Td = TreeCD(2, q=q, lam=lam, disclose=False), TreeCD(2, q=q, lam=lam, disclose=True)
        Sh = train(Th, "full", 2000)
        tab, ids = Sh.compiled()
        uh, gh, _ = Sh.evaluate(tab, ids)
        Sd = GSolver(Td, "full"); Sd.setup()
        f = embed(Th, tab, ids, Td, Sd, strip_c)
        u = {i: float(Sd.values_p(f, Sd.U[i])[0]) for i in (1, 2)}
        gaps = {i: float(Sd.br_p(i, f) - u[i]) for i in (1, 2)}
        rec = {"q": q, "lam": lam, "nashconv_hidden_game": float(sum(gh.values())),
               "nashconv_in_disclosure_game": float(sum(gaps.values())), "gain_p1": gaps[1], "u1": u[1], "u1_hidden": uh[1]}
        out["C"].append(rec)
        print(f"C  q={q} lam={lam}: NashConv hidden {rec['nashconv_hidden_game']:.2e} -> embedded in disclosure game "
              f"{rec['nashconv_in_disclosure_game']:.2e} (u1 {u[1]:.4f} = {uh[1]:.4f})", flush=True)

strip_ab = lambda k: (k[0], k[1], k[2], (), k[4], k[5], k[6])
inst = [("A_default", {"cont": False}), ("B_default", {"cont": True})] + \
       [(f"A_draw{s}", dict(cont=False, **draw(np.random.default_rng(5000 + s)))) for s in range(10)]
for name, prm in inst:
    Th, Td = Market(Params(K=2, **prm)), Market(Params(K=2, disclose=True, **prm))
    Sh = Solver(Th, "full")
    for _ in range(2000):
        Sh.step()
    tab, ids = Sh.compiled()
    nch = float(Sh.nashconv(tab, ids)[0])
    Sd = Solver(Td, "full")
    f = embed(Th, tab, ids, Td, Sd, strip_ab)
    vhi, vlo = float(Sd.br(1, f)), float(-Sd.br(2, f))
    rec = {"name": name, "nashconv_hidden_game": nch, "nashconv_in_disclosure_game": vhi - vlo, "v_lo": vlo, "v_hi": vhi}
    out["AB"].append(rec)
    print(f"AB {name:10s}: NashConv hidden {nch:.2e} -> embedded in disclosure game {vhi - vlo:.2e}", flush=True)
json.dump(out, open("/home/claude/exp/hypotheses/out/h3_embed.json", "w"), indent=1)
print(f"total {time.time() - t00:.1f}s")
