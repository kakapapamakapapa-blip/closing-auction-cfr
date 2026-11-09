"""Test: replace the signal history by (number of U signals, last signal) on top of phi_H (instance A, Gamma_0)."""
import sys, time, numpy as np
from auction import Tree
from cfr import Solver
K, N = int(sys.argv[1]), int(sys.argv[2])
T = Tree(K)
for n in T.dec:
    k = T.keys[n]
    i, t, sig, order, chi = k["H"]
    k["CNT"] = (i, t, (sig.count("U"), sig[-1:]), order, chi)
for v in sys.argv[3].split(","):
    t0 = time.time(); S = Solver(T, v)
    for _ in range(N): S.step()
    tab, ids = S.compiled()
    print(f"K={K} {v:5s} entries/player {int(S.bmask[S.bowner==1].sum()):6d}  NashConv {S.nashconv(tab, ids)[0]:.6f}  {time.time()-t0:.0f}s", flush=True)
