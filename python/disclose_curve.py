"""Instance B (c = 2) in Gamma_0 with revocable interest disclosed before the cutoff: NashConv of the
compiled tables of the full game and of phi_H along CFR iterations, in both disclosure regimes.

Usage: python3 disclose_curve.py N  -> disclose_curve.json
"""
import json
import sys
from market import Market, Params
from cfr import Solver

N = int(sys.argv[1])
marks = [m for m in (10, 30, 100, 300, 1000, 3000, 10000, 30000) if m <= N]
out = {"N": N, "runs": []}
for disclose in (False, True):
    T = Market(Params(K=2, cont=True, disclose=disclose))
    for v in ("full", "H"):
        S = Solver(T, v)
        curve = []
        for n in range(1, N + 1):
            S.step()
            if n in marks:
                tab, ids = S.compiled()
                curve.append({"n": n, "nc": float(S.nashconv(tab, ids)[0])})
        out["runs"].append({"disclose": disclose, "variant": v, "nodes": int(T.M), "curve": curve})
        print(disclose, v, [(c["n"], round(c["nc"], 5)) for c in curve], flush=True)
json.dump(out, open("disclose_curve.json", "w"), indent=1)
