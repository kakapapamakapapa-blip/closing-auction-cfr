"""Vector figures for the thesis (written to ../thesis/figures).

Usage: python3 figures.py [scaling] [frontier] [mccfr] [sweep] [gamma0]   (no argument: all)
Colour identity is fixed across figures: full game orange, phi_H blue, coarse layers grey with
direct labels.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True, "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})
FULL, PHIH, AQUA, GREY = "#eb6834", "#2a78d6", "#1baf7a", "#898781"
INK, INK2 = "#0b0b0b", "#52514e"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "thesis", "figures")
PH = r"$\phi_{\mathrm{H}}$"
LABEL = {"full": "full game", "H": PH, "H_s0m": "forget $s_0$ after $s_1$", "H_s0": "drop $s_0$",
         "H_last": "last signal only", "H_chi": r"drop $\chi$", "A": r"drop $s_0$ and $\chi$",
         "H_nosig": "no signals"}


def finish(ax):
    ax.grid(True, which="major")
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def fig_scaling():
    d = json.load(open("scaling_A.json"))
    K = np.array([r["K"] for r in d]); F = np.array([r["entries_full"] for r in d]); H = np.array([r["entries_H"] for r in d])
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.axvspan(0.6, 4.5, color="#f3f2ee", lw=0)
    ax.text(2.55, 3.2e9, "trees built and\nsolved exactly", ha="center", va="top", color=INK2, fontsize=7.5)
    ax.text(8.5, 3.2e9, "counted without building the tree", ha="center", va="top", color=INK2, fontsize=7.5)
    ax.plot(K, F, "-o", color=FULL, lw=2, ms=4.5, mfc=FULL, mec="white", mew=0.8, label="full game")
    ax.plot(K, H, "-o", color=PHIH, lw=2, ms=4.5, mfc=PHIH, mec="white", mew=0.8, label=PH)
    ax.annotate(f"{F[-1]:.2g}".replace("e+09", r"$\times10^{9}$"), (K[-1], F[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.annotate(f"{H[-1] / 1e5:.1f}" + r"$\times10^{5}$", (K[-1], H[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.text(K[-1] - 0.15, np.sqrt(F[-1] * H[-1]), f"{F[-1] / H[-1]:,.0f}" + r"$\times$", ha="right", va="center",
            color=INK, fontsize=8)
    ax.annotate("", xy=(K[-1] + 0.02, H[-1] * 1.4), xytext=(K[-1] + 0.02, F[-1] / 1.4),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
    ax.set_yscale("log"); ax.set_xlim(0.6, 13.6); ax.set_ylim(10, 4e9)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("pre-cutoff periods $c$"); ax.set_ylabel("table entries per player")
    finish(ax)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55), handlelength=1.8)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_scaling.pdf")


def _frontier(ax, fn, title, place):
    d = json.load(open(fn))["variants"]
    pts = [(v, r["entries"], r["nashconv"]) for v, r in d.items()]
    ref = d["full"]["nashconv"]
    ax.axhline(ref, color="#c3c2b7", lw=1.0, ls=":", zorder=1)
    for v, x, y in pts:
        col = FULL if v == "full" else PHIH if v == "H" else GREY
        ax.plot([x], [y], "o", ms=7 if v in ("full", "H") else 6, color=col, mec="white", mew=1.0, zorder=3)
        dx, dy, ha, va = place.get(v, (6, 0, "left", "center"))
        ax.annotate(LABEL[v], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=7.5, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title, color=INK)
    ax.set_xlabel("table entries, both players")
    finish(ax)


def fig_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    up, dn = (0, 7, "center", "bottom"), (0, -7, "center", "top")
    _frontier(axes[0], "frontier_B_K3.json", "instance B, $c=3$",
              {"full": up, "H": up, "H_chi": dn, "H_s0": (5, 5, "left", "bottom"),
               "H_s0m": (5, -5, "left", "top"), "A": (-5, 5, "right", "bottom"),
               "H_last": (-5, -5, "right", "top"), "H_nosig": (0, 7, "center", "bottom")})
    _frontier(axes[1], "frontier_C_K3.json", "instance C, $c=3$",
              {"full": up, "H": up, "H_chi": dn, "H_s0": up, "A": (-4, 6, "right", "bottom"),
               "H_nosig": dn, "H_last": (6, 0, "left", "center")})
    axes[0].set_ylabel("NashConv, native payoffs")
    for ax in axes:
        lo, hi = ax.get_xlim(); ax.set_xlim(lo / 2.2, hi * 2.2)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo / 1.6, hi * 2.0)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "fig_frontier.pdf")


def fig_mccfr():
    st = {r["K"]: r for r in json.load(open("mccfr_stats.json"))}
    Ks = [k for k in (3, 4) if k in st]
    fig, axes = plt.subplots(1, len(Ks), figsize=(6.3, 2.75), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, K in zip(axes, Ks):
        r = st[K]
        if K == 4 and os.path.exists("vanilla_K4.json"):
            van = [x for x in json.load(open("vanilla_K4.json"))["runs"] if x["variant"] == "full"][0]["curve"]
            ax.plot([c["visits"] for c in van], [c["nc"] for c in van], "--", color=GREY, lw=1.5)
            pt = min(van, key=lambda c: abs(c["visits"] - 1.6e7))
            ax.annotate("vanilla CFR", (pt["visits"], pt["nc"]), xytext=(7, 3),
                        textcoords="offset points", ha="left", fontsize=7.5, color=INK2)
        if K == 3:
            res = json.load(open("results_K3.json"))
            M = res["nodes"]
            cur = res["variants"]["full"]["curve"]
            ax.plot([c["n"] * M for c in cur], [c["nc_table"] for c in cur], "--", color=GREY, lw=1.5)
            pt = min(cur, key=lambda c: abs(c["n"] * M - 1.4e6))
            ax.annotate("vanilla CFR", (pt["n"] * M, pt["nc_table"]), xytext=(7, 3),
                        textcoords="offset points", ha="left", fontsize=7.5, color=INK2)
        for v, col, lab in (("full", FULL, "MCCFR, full game"), ("H", PHIH, "MCCFR, " + PH)):
            c = r["curve"][v]
            x = np.array(c["visits"])
            ax.fill_between(x, c["lo"], c["hi"], color=col, alpha=0.16, lw=0)
            ax.plot(x, c["median"], color=col, lw=2, label=lab)
        ax.axhline(r["target"], color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e6, r["budget"] * 1.05)
        ymax = max(max(r["curve"]["full"]["hi"][i] for i in range(len(r["curve"]["full"]["hi"]))
                       if r["curve"]["full"]["visits"][i] >= 1e6), 0.05)
        ax.set_ylim(r["target"] / 3, ymax * 1.2)
        ax.set_title(f"instance A, $c={K}$, game $\\Gamma_0$", color=INK)
        ax.set_xlabel("node visits")
        finish(ax)
    axes[0].set_ylabel("NashConv")
    axes[0].legend(loc="lower left", handlelength=1.8)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "fig_mccfr.pdf")


def fig_sweep():
    D = json.load(open("sweep_40.json"))
    nc = lambda r, g, v: r[g][v]["nc_table"]
    rows = [("H", "native", PH + r", game $\Gamma$", PHIH),
            ("H", "gamma0", PH + r", game $\Gamma_0$", PHIH),
            ("H", "disclose_gamma0", PH + r", $\Gamma_0$, disclosure before $c$", "hollow"),
            ("H_chi", "native", r"drop $\chi$, game $\Gamma$", GREY),
            ("H_chi", "gamma0", r"drop $\chi$, game $\Gamma_0$", GREY),
            ("H_s0", "native", r"drop $s_0$, game $\Gamma$", GREY),
            ("H_s0", "gamma0", r"drop $s_0$, game $\Gamma_0$", GREY)]
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for k, (v, g, lab, col) in enumerate(rows):
        y = len(rows) - 1 - k
        x = np.array([nc(r, g, v) / nc(r, g, "full") for r in D])
        x = np.maximum(x, 0.3)
        jit = y + rng.uniform(-0.18, 0.18, len(x))
        if col == "hollow":
            ax.plot(x, jit, "o", ms=4.0, mfc="white", mec=PHIH, mew=1.0)
        else:
            ax.plot(x, jit, "o", ms=3.6, color=col, alpha=0.75, mec="white", mew=0.4)
        ax.plot([np.median(x)] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows][::-1])
    ax.set_xscale("log"); ax.axvline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("NashConv of the merge / NashConv of the full game (40 random instances)")
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_sweep.pdf")


def fig_gamma0():
    d = json.load(open("results_K3.json"))["variants"]
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    for key, col, lab in (("H_s0", GREY, LABEL["H_s0"]), ("H_chi", GREY, LABEL["H_chi"]), ("A", GREY, LABEL["A"])):
        cur = d[key]["curve"]
        ax.plot([c["n"] for c in cur], [c["nc_table"] for c in cur], "-", color=col, lw=1.4)
        dy = {"H_chi": -4, "A": 4}.get(key, 0)
        ax.annotate(lab, (cur[-1]["n"], cur[-1]["nc_table"]), xytext=(4, dy), textcoords="offset points",
                    va="top" if dy < 0 else "bottom" if dy > 0 else "center", fontsize=7.5, color=INK)
    cur = d["full"]["curve"]
    ax.plot([c["n"] for c in cur], [c["nc_table"] for c in cur], "-", color=FULL, lw=2.2, label="full game")
    cur = d["H"]["curve"]
    ax.plot([c["n"] for c in cur], [c["nc_table"] for c in cur], "o", ms=5, mfc="white", mec=PHIH, mew=1.4,
            label=PH)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8, 4e4)
    ax.set_xlabel("CFR iterations"); ax.set_ylabel("NashConv")
    finish(ax)
    ax.legend(loc="lower left", handlelength=1.8)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_gamma0.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["scaling", "frontier", "mccfr", "sweep", "gamma0"]
    for w in which:
        globals()["fig_" + w]()
        print("wrote", w)
