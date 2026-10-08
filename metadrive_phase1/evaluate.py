"""Steps 2, 4, 5: drive the FIXED routes with either the autopilot or the trained CNN and log metrics.

    python evaluate.py --policy autopilot            # step 2 checkpoint: car drives the fixed routes
    python evaluate.py --policy cnn                  # step 4/5: CNN drives, baseline numbers
    python evaluate.py --policy cnn --corruption fog --severity 3   # Phase 2 hook

Writes:
    results/<tag>_summary.csv   one row per route (completion, crash, lane deviation, ...)
    results/traces/<tag>_<route>.csv   per-step signals (speed, lateral offset, crash flags),
                                       ready to feed into RTAMT/STL monitors in Phase 2
    results/frames/<tag>_<route>.png   a few camera frames from the run
"""
import argparse, os, csv
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy
from common import (make_env, obs_to_input, to_continuous, lateral_offset, FIXED_ROUTES,
                    touching_solid_line, hit_curb_or_left_road)


def make_corruption(name, severity):
    if not name:
        return None
    from imagecorruptions import corrupt
    return lambda f: corrupt(f, corruption_name=name, severity=severity)


def run_route(env, route, seed, policy, model, corrupt_fn):
    obs, info = env.reset(seed=seed)
    expert = IDMPolicy(env.agent, seed) if policy == "autopilot" else None
    trace, snaps = [], []
    step = 0
    while True:
        image, extra = obs_to_input(obs, env.agent, corrupt_fn)
        if policy == "autopilot":
            action = np.clip(np.array(expert.act(env.agent.id), dtype=np.float32), -1, 1)
        else:
            action = to_continuous(*model.act(image, extra))
        if step % 60 == 30 and len(snaps) < 4:
            snaps.append(image[-3:].transpose(1, 2, 0))  # newest frame, as the model saw it
        obs, r, term, trunc, info = env.step(action)
        step += 1
        x, y = env.agent.position
        trace.append(dict(t=step, x=x, y=y, speed_kmh=env.agent.speed_km_h,
                          lat_offset=lateral_offset(env.agent), steer=action[0], throttle=action[1],
                          crash=int(info["crash"]), out_of_road=int(hit_curb_or_left_road(env.agent, info)),
                          solid_line=int(touching_solid_line(env.agent)),
                          route_completion=float(info["route_completion"])))
        if term or trunc or trace[-1]["out_of_road"]:
            break
    lat = np.abs([s["lat_offset"] for s in trace])
    summary = dict(route=route, seed=seed, steps=step,
                   route_completion=round(float(info["route_completion"]), 4),
                   success=int(bool(info["arrive_dest"])),
                   collision=int(bool(info["crash"])),
                   out_of_road=trace[-1]["out_of_road"],
                   solid_line_touches=int(sum(1 for a, b in zip([{"solid_line": 0}] + trace, trace)
                                              if b["solid_line"] and not a["solid_line"])),
                   mean_lane_dev_m=round(float(lat.mean()), 3),
                   max_lane_dev_m=round(float(lat.max()), 3),
                   pct_time_dev_over_1m=round(float((lat > 1.0).mean() * 100), 2),
                   mean_speed_kmh=round(float(np.mean([s["speed_kmh"] for s in trace])), 2))
    return summary, trace, snaps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["autopilot", "cnn"], required=True)
    ap.add_argument("--model", default="models/cnn_policy.pt")
    ap.add_argument("--routes", nargs="+", default=list(FIXED_ROUTES))
    ap.add_argument("--corruption", default=None, help="imagecorruptions name, e.g. fog, gaussian_noise")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    model = None
    if args.policy == "cnn":
        import torch
        from train_cnn import DrivingCNN
        model = DrivingCNN(); model.load_state_dict(torch.load(args.model, map_location="cpu")); model.eval()
    corrupt_fn = make_corruption(args.corruption, args.severity)
    tag = args.tag or (args.policy + (f"_{args.corruption}{args.severity}" if args.corruption else "_clean"))

    for d in ["results/traces", "results/frames"]:
        os.makedirs(d, exist_ok=True)
    rows = []
    seeds = sorted(FIXED_ROUTES.values())
    env = make_env(seeds[0], num_scenarios=seeds[-1] - seeds[0] + 1)
    for route in args.routes:
        summary, trace, snaps = run_route(env, route, FIXED_ROUTES[route], args.policy, model, corrupt_fn)
        rows.append(summary); print(summary, flush=True)
        with open(f"results/traces/{tag}_{route}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0])); w.writeheader(); w.writerows(trace)
        if snaps:
            from PIL import Image
            strip = np.concatenate(snaps, axis=1)
            Image.fromarray(strip).resize((strip.shape[1] * 3, strip.shape[0] * 3), Image.NEAREST) \
                .save(f"results/frames/{tag}_{route}.png")
    with open(f"results/{tag}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("saved", f"results/{tag}_summary.csv")
    os._exit(0)
