"""Figure 4: how many of the 4 routes the final model finishes as each disturbance grows."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figs_phase2"
os.makedirs(OUT, exist_ok=True)
BLUE, VERM = "#0072B2", "#D55E00"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def finished(tag):
    """Number of routes (out of 4) the run finished."""
    with open(f"results/{tag}_summary.csv") as f:
        return sum(int(float(r["success"])) for r in csv.DictReader(f))


noise = [0, 1, 2, 4, 8, 16, 32, 64, 128]
bright = [10, 20, 40, 60, 80, 100, 120, 140, 180, 220, 300, 450]
n_vals = [finished("cnn_mac9f" if n == 0 else f"cnn_mac9f_noise{n}") for n in noise]
b_vals = [finished("cnn_mac9f" if b == 100 else f"cnn_mac9f_brightness{b}") for b in bright]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True, gridspec_kw={"width_ratios": [9, 12]})
for ax, xs, vals, title, xlabel in (
        (axes[0], noise, n_vals, "Random pixel noise", "Noise size (out of 255)"),
        (axes[1], bright, b_vals, "Brightness", "Brightness (% of normal)")):
    ax.bar([str(x) for x in xs], vals, color=[VERM if v <= 1 else BLUE for v in vals], edgecolor="white")
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.set_ylim(0, 4.9)
    ax.tick_params(axis="x", labelsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, str(v), ha="center", fontsize=8)
axes[0].set_ylabel("Routes finished (of 4)"); axes[0].set_yticks(range(5))
axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
axes[0].text(0.6, 4.85, "formal proof\nfails from here", fontsize=7, va="top")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_thresholds.png", dpi=200); plt.close(fig)
print("saved fig4")
