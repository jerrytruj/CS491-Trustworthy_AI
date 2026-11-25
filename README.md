# CS 491/591 Final Project - Group 08

Brandon Pham, Diego Uriarte, Jerry Trujillo

Robustness testing of an end-to-end driving model in MetaDrive. We trained a CNN to drive by imitating
MetaDrive's autopilot (with DAgger), then tested it with image noise and brightness changes, checked the
drives with RTAMT, and ran auto_LiRPA (CROWN) on the steering decision.

All the code is in `metadrive_phase1/`, run everything from inside that folder.
The final model from the report is `models/cnn_mac9f.pt`. `RESULTS.md` has all the numbers.

## Setup

Python 3.11 only (3.12 doesn't work). MetaDrive from pip is broken so install it from GitHub:

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install "git+https://github.com/metadriverse/metadrive.git"
pip install "git+https://github.com/Verified-Intelligence/auto_LiRPA.git"
pip install -r requirements.txt
```

The first run downloads about 200 MB of MetaDrive assets. On Linux without a display put `xvfb-run -a`
before each python command. Each simulator takes around 4 GB of RAM so run one script at a time.

To check it works: `python evaluate.py --policy autopilot --routes route_A` should finish with `success: 1`.

Note: everything in the report was run on one Mac. Our models didn't give the same results across machines.

## Models

- `cnn_mac9c.pt` - imitation only
- `cnn_mac9d.pt` - DAgger round 1
- `cnn_mac9e.pt` - DAgger round 2
- `cnn_mac9f.pt` - DAgger round 3, final model

`cnn_policy.pt` and `cnn_policy_round0.pt` are old Phase 1 models, not used in the final report.

## Reproducing the report

### Clean drives (Table 1, Fig. 1)

```bash
python evaluate.py --policy autopilot --tag autopilot_clean
python evaluate.py --policy cnn --model models/cnn_mac9c.pt --tag cnn_mac9c
python evaluate.py --policy cnn --model models/cnn_mac9d.pt --tag cnn_mac9d
python evaluate.py --policy cnn --model models/cnn_mac9e.pt --tag cnn_mac9e
python evaluate.py --policy cnn --model models/cnn_mac9f.pt --tag cnn_mac9f
```

### Noise and brightness (Table 2, Fig. 2)

For noise, severity is the max change per pixel out of 255. For brightness it's a percent of normal.
Use these tag names because the figure scripts look for them.

```bash
for k in 1 2 4 8 16 32 64 128; do
  python evaluate.py --policy cnn --model models/cnn_mac9f.pt --corruption noise --severity $k --tag cnn_mac9f_noise$k
done
for p in 10 20 40 60 80 120 140 180 220 300 450; do
  python evaluate.py --policy cnn --model models/cnn_mac9f.pt --corruption brightness --severity $p --tag cnn_mac9f_brightness$p
done
```

On Windows run them one at a time or use Git Bash.

### RTAMT (Fig. 3)

```bash
python rtamt_monitor.py
```

Goes through every drive in `results/traces/` and checks that the car stays within 2.0 m of the lane center
the whole time, and that there's no collision. Output is `results/rtamt_summary.csv`, positive margin means
the rule held.

### Verification (Table 3, Fig. 4)

This needs camera frames from roads 27-29, which aren't in the zip because of size. Record them first
(only takes a few minutes):

```bash
python collect_data.py --out data/autopilot_mac --seeds 27-29
python verify_cnn.py --model models/cnn_mac9f.pt --n 60 --out results/verify_summary.csv
python verify_cnn.py --model models/cnn_mac9e.pt --n 60 --out results/verify_summary_9e.csv
python verify_cnn.py --model models/cnn_mac9f.pt --n 5 --method alpha-CROWN --out results/verify_alpha_test.csv
```

Each frame ends up verified, falsified (the attack found an image that flips the steering), or unknown.
Collection randomly nudges the steering, so a new recording won't have the exact same 60 frames we used.

### Figures

```bash
python make_phase2_figs.py
python make_threshold_fig.py
```

They go in `results/figs_phase2/`.

## Training from scratch

Optional and slow (times below are on a 2-core cloud CPU, a laptop is faster).

```bash
python collect_data.py --out data/autopilot                       # ~75 min
python train_cnn.py --data data/autopilot --epochs 20 --out models/my_round0.pt
python collect_data.py --out data/dagger1 --model models/my_round0.pt --seeds 0-29   # DAgger, ~50 min
python train_cnn.py --data data/autopilot data/dagger1 --epochs 20 --out models/my_round1.pt
```

Repeat the last two lines for each extra DAgger round, with a new data folder each time.
`train_cnn.py` uses roads 27-29 for validation and keeps the best epoch. If collection gets interrupted,
rerun the same command and it skips roads it already finished.

## Setup details

- Training roads are seeds 0-26, validation 27-29. Test routes are A=1000, B=1002, C=1005, D=1006 and were
  never trained on. No traffic.
- Input is the last 3 camera frames (84x84), speed, and the next two route checkpoints. The on-road guide
  line is turned off so the CNN has to actually see the road.
- The CNN only picks steering, 9 options from -1 to 1. Speed is kept around 30 km/h by a simple rule.
- The autopilot (IDM) is the teacher. While collecting we nudge the steering 2% of the time but keep the
  autopilot's label, so the data has recoveries in it (DART). Steering labels get a median filter because
  the autopilot sometimes spikes to full lock for one step.
- A drive ends when the car leaves the road, hits the curb, crashes, or reaches the end. Touching a solid
  line doesn't end it since even the autopilot does that, we just count it.

## Output files

- `results/<tag>_summary.csv` - one row per route (completion, success, crashes, lane deviation, speed)
- `results/traces/` - step by step logs that RTAMT reads
- `results/frames/` - camera snapshots
- `results/rtamt_summary.csv`, `results/verify_summary*.csv` - monitoring and verification results
- `results/figs_phase2/` - report figures
