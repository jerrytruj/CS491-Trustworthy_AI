# Phase 1: MetaDrive + CNN driving baseline (Group 08)

Pipeline: autopilot drives -> record camera frames + the autopilot's actions -> train a CNN to imitate it
-> CNN drives the fixed routes -> baseline metrics.

## Setup (once)

Use **Python 3.11** (3.12+ won't work). The PyPI version of MetaDrive is broken (old `gym` dependency
fails to install), so install it from GitHub:

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install "git+https://github.com/metadriverse/metadrive.git"
pip install -r requirements.txt
```

First run downloads MetaDrive's 3D assets (~200 MB), that's normal.
Linux with no display: put `xvfb-run -a` in front of every `python` command. Mac/Windows: not needed.
Each simulator uses ~4 GB of RAM with software rendering, so only run one script at a time.

**Quick test that setup worked:** `python evaluate.py --policy autopilot --routes route_A`
(should finish with `success: 1` in a few minutes).

## Run it (from this folder)

| Step | Command | Time on a 2-core cloud CPU |
|---|---|---|
| 1. Autopilot drives the 4 fixed routes (checkpoint) | `python evaluate.py --policy autopilot` | ~10 min |
| 2. Collect autopilot data on 30 training routes | `python collect_data.py --out data/autopilot` | ~75 min |
| 3. Train the CNN | `python train_cnn.py --data data/autopilot --epochs 20` | ~8 min |
| 4. CNN drives the 4 fixed routes | `python evaluate.py --policy cnn` | ~10 min |
| 5. (optional) DAgger round: CNN drives, autopilot labels | `python collect_data.py --out data/dagger1 --model models/cnn_policy.pt --seeds 0-19` | ~50 min |
| 6. (optional) Retrain with DAgger data, then redo step 4 | `python train_cnn.py --data data/autopilot data/dagger1 --epochs 20` | ~10 min |
| 7. Table + figures for the report | `python make_report_figs.py` | seconds |

Your machine should be a lot faster than these times. Collection saves each route as its own file in the
`--out` folder, so if it gets interrupted just rerun the same command and it skips finished routes.

Trained models are included so you can skip straight to step 4:
- `models/cnn_policy_round0.pt`: trained on autopilot data only
- `models/cnn_policy.pt`: after one DAgger round
Use one with `python evaluate.py --policy cnn --model models/cnn_policy_round0.pt --tag cnn_round0`.

## Outputs
- `results/<tag>_summary.csv`: one row per route (route completion, success, collision, out of road,
  solid line touches, lane deviation, speed)
- `results/traces/*.csv`: per-step speed, lane offset, crash/line flags (what RTAMT will read in Phase 2)
- `results/frames/*.png`: what the camera saw
- `results/figs/`: report table + figures (`make_report_figs.py` reads the `autopilot_clean` and `cnn_clean` tags)

## Where things stand (Oct 7)

| | Autopilot | CNN round 0 | CNN after DAgger |
|---|---|---|---|
| route_A | completes | completes | completes, cleaner (4% of time >1 m off center vs 11%) |
| route_B | completes | hits curb at 28% | hits curb at 24% |
| route_C | completes | leaves road at 41% (roundabout) | stalls, 35% at time limit (over-brakes) |
| route_D | completes | leaves road at 49% (T-junction) | stalls at 50% at time limit (over-brakes) |

So the pipeline works end to end, the autopilot baseline is solid (4/4), and the CNN baseline is weak
outside the easy route. That's a legit Phase 1 baseline as is, but if you want it stronger, ideas in
rough order of payoff:
1. **Fix over-braking from DAgger.** The autopilot slams the brake (-1) any time speed passes ~30 km/h,
   so DAgger data is ~25% brake labels. Try a lower brake weight in `train_cnn.py` (throttle class weight
   power 1.0 -> 0.5), or relabel brake as cruise when speed is within a few km/h of 30.
2. **More DAgger rounds** (round 2 using the new model), the textbook fix for junction failures.
3. **Look at route_B's start**: the CNN creeps from 0 km/h for a long time (autopilot floors it at the
   start); check `results/traces/cnn_clean_route_B.csv`.
4. More training routes or bigger images (84x84 is small for spotting junctions).

## Design notes (for the methodology section)
- **Routes:** procedurally generated 4-block MetaDrive roads. Training on seeds 0-29; validation during
  training uses whole held-out routes (27-29). Fixed evaluation routes, never trained on:
  route_A=1000, route_B=1002, route_C=1005, route_D=1006 (mix of intersections, roundabouts, T-junction,
  ramps). Seeds 1001, 1003, 1007 were screened out because the autopilot itself crashes on them
  (3 of the 10 screened). No traffic in Phase 1.
- **Inputs:** last 3 front-camera frames (84x84 RGB), speed, and a route target: positions of the next two
  route checkpoints relative to the car (same idea as TCP's target point). MetaDrive's on-road guide line
  is turned off so the CNN has to read the actual road. MetaDrive's built-in left/straight/right flags
  were tried first but only describe road curvature, never which way to turn at a junction.
- **Actions:** 21 discrete = 7 steering bins x 3 throttle bins (brake -0.3 / cruise 0.05 / accelerate 0.5).
- **Teacher:** MetaDrive's built-in IDM autopilot. During collection the steering is occasionally nudged
  off for a few steps while the label stays the autopilot's action (DART-style noise injection), so the
  CNN sees recoveries. Optional DAgger rounds as above.
- **Episode end:** leaving the drivable road, hitting the curb, collision, or reaching the destination.
  MetaDrive's default also ends an episode on touching any solid line, which even its own autopilot trips,
  so that's turned off and line touches are counted as a metric instead.
- **Metrics:** route completion %, success, collision, out-of-road, solid line touches, mean/max offset
  from lane center (m, measured against the route's own lanes), % of time >1 m off center, mean speed.

If anything errors, paste the full error message into chat.
