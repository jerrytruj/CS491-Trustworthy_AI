"""Makes the three Phase 2 report figures from the results CSVs."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "results/figs_phase2"
os.makedirs(OUT, exist_ok=True)
ROUTES = ["A", "B", "C", "D"]
BLUE, VERM, GRAY, SKY, GREEN = "#0072B2", "#D55E00", "#8C8C8C", "#56B4E9", "#009E73"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def drive_scores(tag):
    """Route completion and success flag for routes A to D of one run."""
    d = {r["route"].replace("route_", ""): r for r in read(f"results/{tag}_summary.csv")}
    return ([float(d[x]["route_completion"]) for x in ROUTES], [int(float(d[x]["success"])) for x in ROUTES])


# Figure 1: route completion by training round
models = [("Autopilot", "autopilot_clean", GRAY), ("Imitation only", "cnn_mac9c", VERM),
          ("DAgger round 1", "cnn_mac9d", SKY), ("DAgger round 2", "cnn_mac9e", BLUE),
          ("DAgger round 3", "cnn_mac9f", GREEN)]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
w = 0.16
for k, (label, tag, color) in enumerate(models):
    comp, ok = drive_scores(tag)
    for i in range(4):
        ax.bar(i + (k - 2) * w, comp[i], w, color=color, hatch="" if ok[i] else "///",
               edgecolor="white" if ok[i] else "black", linewidth=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels([f"Route {r}" for r in ROUTES])
ax.set_ylabel("Route completion"); ax.set_ylim(0, 1.05)
ax.legend(handles=[Patch(color=c, label=l) for l, _, c in models], frameon=False, ncol=5,
          loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
ax.text(1.0, -0.2, "Hatched bars = route not finished", transform=ax.transAxes, ha="right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_dagger_rounds.png", dpi=200); plt.close(fig)

# Figure 2: verification outcome by noise size
ver = [r for r in read("results/verify_summary.csv") if r["disturbance"] == "noise"]
labels = [f"±{r['level']}" for r in ver]
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
for ax, prefix, title in zip(axes, ("loose", "strict"), ("Within one steering slot", "Same steering slot")):
    bottom = np.zeros(len(ver))
    for key, color, name in (("verified", BLUE, "Verified"), ("falsified", VERM, "Flip found"), ("unknown", GRAY, "Unknown")):
        vals = np.array([int(r[f"{prefix}_{key}"]) for r in ver], dtype=float)
        ax.bar(labels, vals, bottom=bottom, color=color, label=name, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_title(title, fontsize=10); ax.set_xlabel("Pixel noise (out of 255)")
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_ylabel("Frames (of 60)")
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, frameon=False, fontsize=8, ncol=3, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(f"{OUT}/fig2_verification.png", dpi=200); plt.close(fig)

# Figure 3: RTAMT lane margins for the final model (cnn_mac9f)
rt = {(r["tag"], r["route"]): r for r in read("results/rtamt_summary.csv")}
rows = [("Clean", "cnn_mac9f")] + [(f"Brightness {p}%", f"cnn_mac9f_brightness{p}") for p in (60, 80, 120, 140)] \
     + [(f"Noise ±{n}/255", f"cnn_mac9f_noise{n}") for n in (1, 2, 4, 8)]
M = np.array([[float(rt[(tag, r)]["margin_lane"]) for r in ROUTES] for _, tag in rows])
fig, ax = plt.subplots(figsize=(5.2, 4.2))
im = ax.imshow(np.clip(M, -5, 5), cmap="RdBu", vmin=-5, vmax=5, aspect="auto")
for i, (_, tag) in enumerate(rows):
    for j, r in enumerate(ROUTES):
        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if abs(M[i, j]) > 3 else "black")
        if int(rt[(tag, r)]["failed"]):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2))
ax.set_xticks(range(4)); ax.set_xticklabels([f"Route {r}" for r in ROUTES])
ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for n, _ in rows])
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax); cb.set_label("Lane margin (m), colors clipped at ±5")
ax.set_title("Positive = within 2 m of lane center. Black box = crash or left road", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_rtamt_margins.png", dpi=200); plt.close(fig)
print("saved 3 figures in", OUT)
