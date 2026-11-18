# Drives one route with the GUI on so it can be shown live. Writes no result files.
import argparse, os
import numpy as np
from evaluate import make_corruption, run_route
from common import make_env, FIXED_ROUTES

ap = argparse.ArgumentParser()
ap.add_argument("--policy", choices=["autopilot", "cnn"], required=True)
ap.add_argument("--model", default="models/cnn_mac9f.pt")
ap.add_argument("--route", default="B")
ap.add_argument("--corruption", default=None)
ap.add_argument("--severity", type=int, default=3)
args = ap.parse_args()

# Load the trained driver only when it is needed
model = None
if args.policy == "cnn":
    import torch
    from train_cnn import DrivingCNN
    model = DrivingCNN()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

corrupt_fn = make_corruption(args.corruption, args.severity)
seeds = sorted(FIXED_ROUTES.values())
env = make_env(seeds[0], num_scenarios=seeds[-1] - seeds[0] + 1)
summary, trace, snaps = run_route(env, args.route, FIXED_ROUTES[args.route], args.policy, model, corrupt_fn)
print(summary)
os._exit(0)
