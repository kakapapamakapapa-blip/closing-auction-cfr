"""Exact table sizes of player 1 in instance A as functions of the window length K (no tree needed).
Pre-cutoff observations: 2^(t+1) signal histories. Post-cutoff: 2^K histories x 3 values of chi.
Own sequences: e_t ending empty and L*e_s ending with an order submitted at s (Proposition 4.5)."""
import json
L = 2
def e(n):
    a = [1, 1]
    while len(a) <= n:
        a.append(2 * a[-1] + (L - 1) * a[-2])
    return a[n]
def counts(K):
    full = H = 0
    for t in range(K):                      # pre-cutoff decisions: empty -> 3 actions, resting -> 2
        obs = 2 ** (t + 1)
        empty_seq, rest_seq = e(t), L * sum(e(s) for s in range(t))
        full += obs * (3 * empty_seq + 2 * rest_seq)
        H += obs * (3 * 1 + 2 * L * t)
    obs = 3 * 2 ** K                       # post-cutoff decision: empty -> 3, resting -> 1
    empty_seq, rest_seq = e(K), L * sum(e(s) for s in range(K))
    full += obs * (3 * empty_seq + 1 * rest_seq)
    H += obs * (3 * 1 + 1 * L * K)
    return full, H
rows = []
for K in range(1, 13):
    f, h = counts(K)
    rows.append({"K": K, "entries_full": f, "entries_H": h, "ratio": f / h})
    print(K, f, h, round(f / h, 2))
json.dump(rows, open("scaling_A.json", "w"), indent=1)
