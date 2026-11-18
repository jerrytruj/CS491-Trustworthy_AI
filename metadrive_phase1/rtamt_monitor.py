"""Runs STL lane-keeping checks (RTAMT) on every recorded drive in results/traces."""
import csv, glob, os, sys, types, typing

# Some Python versions lack typing.io, which older RTAMT dependencies import.
if "typing.io" not in sys.modules:
    m = types.ModuleType("typing.io"); m.TextIO = typing.TextIO
    sys.modules["typing.io"] = m; typing.io = m
import rtamt

LANE_LIMIT = 2.0  # lane edge is 1.75 m off center, so 2.0 allows touching the line but not leaving the lane
SEVERE_LIMIT = 3.5  # a full lane width away: wrong lane or off the road
SKIP = {"cnn_clean", "cnn_render", "autopilot_render", "cnn_mac", "cnn_mac9"}  # old runs


def robustness(dev, limit):
    """Margin of 'always |offset| <= limit' over a whole drive. Positive = held, negative = violated."""
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var("dev", "float")
    spec.spec = f"always (dev <= {limit})"
    spec.parse()
    times = list(range(len(dev)))
    try:
        out = spec.evaluate({"time": times, "dev": dev})
    except (TypeError, KeyError):  # older/newer RTAMT versions take a list instead of a dict
        out = spec.evaluate(["dev", list(zip(times, dev))])
    return float(out[0][1])  # value at time 0 covers the whole drive


rows = []
for path in sorted(glob.glob("results/traces/*_route_*.csv")):
    name = os.path.basename(path)[:-4]
    tag, route = name.split("_route_")
    if tag in SKIP:
        continue
    with open(path) as f:
        data = list(csv.DictReader(f))
    dev = [abs(float(r["lat_offset"])) for r in data]
    failed = any(int(float(r["crash"])) or int(float(r["out_of_road"])) for r in data)
    rows.append(dict(tag=tag, route=route, steps=len(data),
                     completion=round(float(data[-1]["route_completion"]), 3),
                     failed=int(failed),
                     margin_lane=round(robustness(dev, LANE_LIMIT), 3),
                     margin_severe=round(robustness(dev, SEVERE_LIMIT), 3)))

with open("results/rtamt_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
print("saved results/rtamt_summary.csv")
