"""Section 10.3: Monte Carlo check of the thinning-coupling bound (Theorem 8.2).

Two d-variate exponential Hawkes processes with initial excitations e and e_hat are
driven by one Poisson random measure per coordinate (marks uniform under a dominating
level). An atom joins a path if its mark lies below that path's intensity. The paths
split at the first atom that lies between the two intensities. Theorem 8.2 says the
split happens before tau with probability exactly
    p_B = 1 - exp(-||e - e_hat||_1 (1 - exp(-beta tau)) / beta),
because before the split both paths receive the same jumps, so the band between the
intensities has deterministic height |e^k - e_hat^k| exp(-beta t).
The simulation does not use that fact: each path's intensity is recomputed from its own
event history.
"""
import json
import math
import sys
import numpy as np


def split_before(mu, alpha, beta, e, eh, tau, rng):
    d = len(mu)
    ex_a, ex_b = np.array(e, float), np.array(eh, float)
    t = 0.0
    while True:
        lam_a = mu + ex_a
        lam_b = mu + ex_b
        M = np.maximum(lam_a, lam_b)              # dominating level until the next jump (intensities decay)
        rate = M.sum()
        w = rng.exponential(1.0 / rate)
        if t + w >= tau:
            return False
        t += w
        decay = math.exp(-beta * w)
        ex_a *= decay; ex_b *= decay
        k = rng.choice(d, p=M / rate)
        u = rng.random() * M[k]                   # mark of the proposed atom
        in_a = u <= mu[k] + ex_a[k]
        in_b = u <= mu[k] + ex_b[k]
        if in_a != in_b:
            return True
        if in_a:                                  # common event: both paths jump by alpha[:, k]
            ex_a += alpha[:, k]; ex_b += alpha[:, k]


def p_B(e, eh, beta, tau):
    return 1 - math.exp(-np.abs(np.array(e) - np.array(eh)).sum() * (1 - math.exp(-beta * tau)) / beta)


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    mu = np.array([1.0, 0.8])
    alpha = np.array([[0.6, 0.3], [0.2, 0.5]])  # spectral radius / beta < 1
    cases = [
        {"beta": 1.5, "tau": 2.0, "e": [0.5, 0.2], "eh": [0.3, 0.2]},
        {"beta": 1.5, "tau": 2.0, "e": [1.0, 0.0], "eh": [0.0, 0.6]},
        {"beta": 0.8, "tau": 5.0, "e": [0.4, 0.4], "eh": [0.1, 0.5]},
        {"beta": 3.0, "tau": 1.0, "e": [2.0, 1.0], "eh": [0.5, 0.5]},
    ]
    out = []
    for c in cases:
        hits = sum(split_before(mu, alpha, c["beta"], c["e"], c["eh"], c["tau"], rng) for _ in range(R))
        p = hits / R
        se = math.sqrt(p * (1 - p) / R)
        row = dict(c, R=R, empirical=p, se=se, p_B=p_B(c["e"], c["eh"], c["beta"], c["tau"]))
        row["z"] = (p - row["p_B"]) / se
        out.append(row)
        print(json.dumps(row), flush=True)
    json.dump({"mu": mu.tolist(), "alpha": alpha.tolist(), "cases": out}, open("hawkes_check.json", "w"), indent=1)
