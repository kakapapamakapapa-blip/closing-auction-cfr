"""Section 6.7: cost per node visit of external-sampling MCCFR on the full game and on phi_H.

The two abstractions are run alternately on the same core (full, phi_H, full, phi_H, ...), each for
a fixed number of node visits from a fresh start, so that both see the same machine conditions.
Usage: python3 timing.py [visits_per_run] [repetitions]
"""
import json
import sys
import time
from auction import Tree
from mccfr import ES

if __name__ == "__main__":
    V = float(sys.argv[1]) if len(sys.argv) > 1 else 2e7
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out = {"visits_per_run": V, "runs": []}
    for K in (3, 4):
        T = Tree(K)
        for rep in range(reps):
            for v in ("full", "H"):
                es = ES(T, v, 1000 + rep)
                t0 = time.perf_counter()
                while es.visits < V:
                    es.iterate()
                dt = time.perf_counter() - t0
                rec = {"K": K, "variant": v, "rep": rep, "visits": es.visits, "seconds": dt,
                       "us_per_visit": dt / es.visits * 1e6}
                out["runs"].append(rec)
                print(json.dumps(rec), flush=True)
    json.dump(out, open("timing.json", "w"), indent=1)
