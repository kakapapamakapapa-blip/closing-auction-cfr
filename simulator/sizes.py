"""Size of the closing-cross game at real time resolution (decision at every NOII publication).
Counts agent information sets / abstraction keys and memory, for the simulator's discretization:
C pre-cutoff periods (signal U/D; wait/submit MOC/cancel; up to Q lots), 330 NOII publications,
each observed as (imbalance bucket 7 values) x (mid offset, 5 values in the coarse keys),
post-cutoff own actions wait/MOC/LOC/IO (MOC until 15:55, LOC until 15:58, IO until 16:00)."""
import json, math
from functools import lru_cache

NE = 330
def tsec(e): return 10 * e if e < 30 else 300 + (e - 30)

def pre_sequences(C, Q):
    """number of own pre-cutoff action sequences of length C and number of distinct resting states"""
    # state: tuple of stamps (sorted), actions wait / submit (if len<Q) / cancel newest (if len>0)
    from collections import Counter
    cur = Counter({(): 1})
    for t in range(C):
        nxt = Counter()
        for st, n in cur.items():
            nxt[st] += n
            if len(st) < Q: nxt[st + (t,)] += n
            if len(st) > 0: nxt[st[:-1]] += n
        cur = nxt
    return sum(cur.values()), len(cur), cur

def post_own_states(Q, rest):
    """post commitments (moc, loc, io) with rest + total <= Q"""
    return [(a, b, c) for a in range(Q + 1) for b in range(Q + 1) for c in range(Q + 1) if rest + a + b + c <= Q]

def own_post_sequences(Q, rest, D_epochs):
    """number of own post-cutoff action sequences over decision epochs (exact DP)"""
    @lru_cache(None)
    def f(i, left):
        if i == len(D_epochs): return 1
        e = D_epochs[i]; ts = tsec(e)
        acts = 1 + (left > 0) * ((ts < 300) + (ts < 480) + 1)
        tot = f(i + 1, left)
        if left > 0: tot += ((ts < 300) + (ts < 480) + 1) * f(i + 1, left - 1)
        return tot
    return f(0, Q - rest)

def log10(x): return math.log10(x) if x > 0 else float('-inf')

def main():
    out = []
    for C in (6, 10, 50):
        Q = 2
        nseq, nrest, dist = pre_sequences(C, Q)
        epochs = list(range(NE))
        # perfect recall at real resolution, agent, at the last decision (epoch 329):
        #   signals 2^C * own pre sequences * observation sequences (>= 2^330 even with a binary alphabet)
        #   * own post sequences
        post_seq = sum(n * own_post_sequences(Q, len(st), tuple(epochs)) for st, n in dist.items())
        lg_obs_min = NE * math.log10(2)      # binary observation per publication
        lg_obs_sim = NE * math.log10(7 * 5)  # simulator alphabet
        lg_full = C * math.log10(2) + log10(post_seq) + lg_obs_sim
        lg_full_min = C * math.log10(2) + log10(post_seq) + lg_obs_min
        # phiH: own pre sequences replaced by resting states
        post_seq_H = sum(own_post_sequences(Q, len(st), tuple(epochs)) for st in dist)
        lg_H = C * math.log10(2) + log10(post_seq_H) + lg_obs_sim
        # coarse keys (variant 2 of the simulator): per decision epoch:
        #   resting state x post commitments x nU (C+1) x current imb (7) x previous imb (7) x mid (5)
        rest_states = list(dist.keys())
        own = sum(len(post_own_states(Q, len(st))) for st in rest_states)
        coarse_post = NE * own * (C + 1) * 7 * 7 * 5
        coarse_pre = sum(t_states for t_states in [C * nrest * (C + 1)])
        coarse = coarse_post + coarse_pre
        bytes_per = 40
        # with 10-second time blocks after 15:55 (30 + 30 blocks) instead of 330 epochs
        blocks = 30 + 30
        coarse_blocks = blocks * own * (C + 1) * 7 * 7 * 5 + coarse_pre
        out.append(dict(C=C, own_pre_sequences=nseq, resting_states=nrest,
                        log10_infosets_full_PR=round(lg_full, 1), log10_infosets_full_PR_binary_obs=round(lg_full_min, 1),
                        log10_infosets_phiH_PR=round(lg_H, 1),
                        coarse_keys_330=coarse, coarse_GB=round(coarse * bytes_per / 1e9, 2),
                        coarse_keys_blocks60=coarse_blocks, coarse_blocks_MB=round(coarse_blocks * bytes_per / 1e6, 1)))
    json.dump(out, open('sizes.json', 'w'), indent=1)
    for r in out: print(r)

if __name__ == '__main__':
    main()
