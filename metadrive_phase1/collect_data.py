"""Step 3a: record MetaDrive's built-in autopilot (IDM policy) driving, to train the CNN by imitation.

The autopilot labels every frame. To get recovery examples (otherwise the CNN only
ever sees perfect driving and falls apart once it drifts), we sometimes add a short
steering disturbance to what the car actually does, while the label stays the
autopilot's correct action. This is the DART noise-injection trick.

DAgger mode (--model): the trained CNN drives instead, and the autopilot still labels every
frame with what it WOULD do. This captures exactly the situations the CNN gets itself into
(too slow, drifting at a junction) together with the correct response.

Usage:
    python collect_data.py --out data/autopilot                      # autopilot drives
    python collect_data.py --out data/dagger1 --model models/cnn_policy.pt --seeds 0-19
Each route is saved as <out>/seed_<n>.npz as soon as it finishes; re-running skips finished routes.
"""
import argparse, os, time
import numpy as np
from metadrive.policy.idm_policy import IDMPolicy
from common import make_env, obs_to_input, discretize, to_continuous, TRAIN_SEEDS, hit_curb_or_left_road


def parse_seeds(s):
    if s is None:
        return TRAIN_SEEDS
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def run_episode(env, seed, rng, perturb_prob=0.02, driver=None):
    obs, info = env.reset(seed=seed)
    expert = IDMPolicy(env.agent, seed)
    images, extras, steer_y, thr_y, raw = [], [], [], [], []
    perturb_left, perturb = 0, 0.0
    while True:
        a = np.clip(np.array(expert.act(env.agent.id), dtype=np.float32), -1, 1)
        image, extra = obs_to_input(obs, env.agent)
        s, t = discretize(a)
        images.append(image); extras.append(extra); steer_y.append(s); thr_y.append(t); raw.append(a)

        executed = a.copy()
        if driver is not None:  # DAgger: CNN drives, autopilot only labels
            executed = to_continuous(*driver.act(image, extra))
        elif perturb_left == 0 and rng.random() < perturb_prob:
            perturb_left, perturb = int(rng.integers(3, 8)), float(rng.normal(0, 0.2))
        if perturb_left > 0:
            executed[0] = np.clip(executed[0] + perturb, -1, 1)
            perturb_left -= 1
        obs, r, term, trunc, info = env.step(executed)
        if term or trunc or hit_curb_or_left_road(env.agent, info):
            break
    result = dict(seed=seed, steps=len(images), completion=float(info["route_completion"]),
                  crash=bool(info["crash"]), out_of_road=bool(info["out_of_road"]))
    return (np.stack(images), np.stack(extras), np.array(steer_y), np.array(thr_y), np.stack(raw)), result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="e.g. 0-11 (default: all TRAIN_SEEDS)")
    ap.add_argument("--out", default="data/autopilot", help="folder for per-route files")
    ap.add_argument("--model", default=None, help="DAgger: let this trained CNN drive")
    args = ap.parse_args()
    driver = None
    if args.model:
        import torch
        from train_cnn import DrivingCNN
        driver = DrivingCNN(); driver.load_state_dict(torch.load(args.model, map_location="cpu")); driver.eval()
    rng = np.random.default_rng(0)
    seeds = parse_seeds(args.seeds)
    ep_dir = args.out
    os.makedirs(ep_dir, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(f"{ep_dir}/seed_{s}.npz")]
    if todo:
        env = make_env(seeds[0], num_scenarios=len(seeds))
    for seed in todo:  # each route saved as soon as it finishes, so a crash loses at most one
        t0 = time.time()
        (im, ex, st, th, ra), res = run_episode(env, seed, rng, driver=driver)
        np.savez_compressed(f"{ep_dir}/seed_{seed}.npz", images=im, extras=ex, steer=st, throttle=th,
                            actions=ra)  # raw continuous actions, so bins can be changed later without recollecting
        print(f"seed {seed}: {res} ({time.time()-t0:.0f}s)", flush=True)
    print("done, saved to", ep_dir)
    os._exit(0)  # MetaDrive can segfault on interpreter exit; data is already saved
