"""Extra test suggested by H3: disclosure of resting on-close interest in instance C (priority race).

TreeCD(K, q, lam, cancel, persist, disclose): as instance C, but if disclose is True, at the start of
every pre-cutoff period t >= 1 both buyers observe the number of resting MOC buy orders (both
strategic orders and exogenous arrivals so far), while cancellation is still possible - the analogue
of market.py's disclose-before-freeze regime. The disclosed count is appended to the public signal
tuple as a token 'b<count>', so all abstraction keys that keep signals also keep the disclosure;
the closing-state law counts only 'U' tokens and the signal chain uses the last 'U'/'D' token.
"""
import sys
sys.path.insert(0, "/home/claude/exp/hypotheses"); sys.path.insert(0, "/home/claude/exp")
from auction_c import TreeC, chi_c, z_probs


class TreeCD(TreeC):
    def __init__(self, K, q=0.4, lam=0.6, cancel=True, persist=0.7, disclose=False):
        self.disclose = disclose
        super().__init__(K, q=q, lam=lam, cancel=cancel, persist=persist)

    def _chance(self, node, t, sig, a1, a2, o1, o2, exo):
        K, q = self.K, self.q
        arrivals = [(False, 1.0)] if t == 0 else [(False, 1 - q), (True, q)]
        for arr, pa in arrivals:
            if pa <= 0:
                continue
            exo2 = exo + ((t - 1) + 0.5,) if arr else exo
            if t < K:
                prev = next((s for s in reversed(sig) if s in ("U", "D")), None)
                tok = ()
                if self.disclose and t >= 1:
                    tok = (f"b{(o1 is not None) + (o2 is not None) + len(exo2)}",)
                for s, ps in self._sig(prev).items():
                    child = self._new(node, self.depth[node] + 1, 1, 1, 0, pa * ps, -1)
                    self._p1(child, t, sig + tok + (s,), a1, a2, o1, o2, "pre", None, exo2, None)
            else:
                n_up = sum(s == "U" for s in sig)
                for X, px in z_probs(n_up, K).items():
                    if px <= 0:
                        continue
                    chi = chi_c(o1, o2, X, exo2)
                    child = self._new(node, self.depth[node] + 1, 1, 1, 0, pa * px, -1)
                    self._p1(child, K, sig, a1, a2, o1, o2, "post", chi, exo2, X)
