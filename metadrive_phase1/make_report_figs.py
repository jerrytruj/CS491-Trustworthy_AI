"""Step 5: turn the evaluation outputs into a table + figures for the Phase 1 report.

    python make_report_figs.py
Writes results/figs/baseline_table.md, lane_offset.png, trajectories.png
"""
import os, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import FIXED_ROUTES

COLORS = {"autopilot": "#2a78d6", "cnn": "#eb6834"}
LABELS = {"autopilot": "Autopilot (teacher)", "cnn": "CNN policy"}
os.makedirs("results/figs", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})

# ---- table ----
frames = []
for pol in COLORS:
    p = f"results/{pol}_clean_summary.csv"
    if os.path.exists(p):
        d = pd.read_csv(p); d.insert(0, "policy", LABELS[pol]); frames.append(d)
tab = pd.concat(frames)
cols = ["policy", "route", "route_completion", "success", "collision", "out_of_road",
        "mean_lane_dev_m", "max_lane_dev_m", "pct_time_dev_over_1m", "mean_speed_kmh"]
tab = tab[cols].copy()
tab["route_completion"] = (tab["route_completion"] * 100).round(1)
avg = tab.groupby("policy", sort=False).mean(numeric_only=True).round(2).reset_index()
avg.insert(1, "route", "AVERAGE")
out = pd.concat([tab, avg]).rename(columns={"route_completion": "route_completion_%"})
with open("results/figs/baseline_table.md", "w") as f:
    f.write(out.to_markdown(index=False))
out.to_csv("results/figs/baseline_table.csv", index=False)
print(out.to_string(index=False))

# ---- lane offset over the route, one panel per route ----
routes = list(FIXED_ROUTES)
fig, axes = plt.subplots(1, len(routes), figsize=(11, 2.6), sharey=True)
for ax, route in zip(axes, routes):
    for pol in COLORS:
        p = f"results/traces/{pol}_clean_{route}.csv"
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        ax.plot(t["route_completion"] * 100, t["lat_offset"], color=COLORS[pol], lw=1.6, label=LABELS[pol])
    ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
    ax.set_title(route.replace("_", " "), fontsize=9)
    ax.set_xlabel("Route completion (%)")
axes[0].set_ylabel("Offset from lane center (m)")
axes[0].legend(frameon=False, fontsize=8, loc="best")
fig.tight_layout(); fig.savefig("results/figs/lane_offset.png", dpi=200); plt.close(fig)

# ---- top-down trajectories ----
fig, axes = plt.subplots(1, len(routes), figsize=(11, 3.2))
for ax, route in zip(axes, routes):
    for pol, lw, ls in [("autopilot", 3.0, "-"), ("cnn", 1.4, "--")]:
        p = f"results/traces/{pol}_clean_{route}.csv"
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        ax.plot(t["x"], t["y"], color=COLORS[pol], lw=lw, ls=ls, label=LABELS[pol])
        if pol == "cnn" and (t["out_of_road"].iloc[-1] or t["crash"].iloc[-1]):
            ax.plot(t["x"].iloc[-1], t["y"].iloc[-1], "x", color="#0b0b0b", ms=8, mew=2)
    ax.set_aspect("equal", "datalim"); ax.set_title(route.replace("_", " "), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(frameon=False, fontsize=8, loc="best")
fig.tight_layout(); fig.savefig("results/figs/trajectories.png", dpi=200); plt.close(fig)
print("figures saved to results/figs/")
