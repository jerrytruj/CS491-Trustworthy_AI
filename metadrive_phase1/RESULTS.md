# Results summary (for the final report)

All runs: MetaDrive 0.4.3 on a Mac, final CNN steering policy `models/cnn_mac9f.pt`.
Training roads are seeds 0-29. Test routes: A=1000, B=1002, C=1005, D=1006 (never trained on).

## What changed from Phase 1 (disclose in the report)
- The network now only chooses steering (9 steering slots). Speed is held near 30 km/h by a fixed rule.
- Steering labels are cleaned with a window-3 median filter, because the autopilot emits one-step full-lock spikes on straight roads.
- The best epoch (by held-out accuracy) is saved instead of the last one.
- Three DAgger rounds were added (the CNN drives, the autopilot labels what it should have done).
- Diego's Phase 1 model did not reproduce on a Mac, so everything was retrained and re-evaluated here.
- Formal checks use auto_LiRPA (CROWN bounds, the engine inside alpha,beta-CROWN), not the full alpha,beta-CROWN program.

## 1. Driver quality by training round (clean drives, route completion)
| Model | A | B | C | D | Routes finished |
|---|---|---|---|---|---|
| Autopilot (expert) | 0.99 | 0.99 | 0.99 | 0.99 | 4/4 |
| Imitation only (cnn_mac9c) | 0.52 | 0.28 | 0.46 | 0.49 | 0/4 |
| DAgger round 1 (cnn_mac9d) | 0.99 | 0.91 | 0.49 | 0.99 | 2/4 |
| DAgger round 2 (cnn_mac9e) | 0.99 | 0.96 | 0.74 | 0.99 | 2/4 |
| DAgger round 3 (cnn_mac9f, final) | 0.99 | 0.99 | 0.64 | 0.99 | 3/4 |

Imitation alone finished 0 of 4 routes. Each DAgger round improved it, ending at 3 of 4.
On A, B, and D the final CNN's lane deviation (0.09 to 0.17 m mean) is close to the autopilot's (0.09 to 0.27 m).
Route C is the hard one: the network leaves the road at about 64% every time on clean drives.

## 2. Closed-loop drives with disturbances (final model, success per route)
| Setting | A | B | C | D | Success |
|---|---|---|---|---|---|
| Clean | ok | ok | fail | ok | 3/4 |
| Brightness 60% | ok | ok | fail | ok | 3/4 |
| Brightness 80% | ok | ok | fail | ok | 3/4 |
| Brightness 120% | ok | fail | ok | ok | 3/4 |
| Brightness 140% | ok | ok | fail | ok | 3/4 |
| Noise +-1/255 | ok | ok | fail | ok | 3/4 |
| Noise +-2/255 | ok | ok | fail | ok | 3/4 |
| Noise +-4/255 | ok | fail | ok | ok | 3/4 |
| Noise +-8/255 | ok | ok | ok | ok | 4/4 |

Routes A and D finished in all 9 runs. B finished 7 of 9 and C finished 3 of 9.
Within this range the success count did not fall as the disturbance got bigger. Section 2b pushes much harder.
Route C flips between pass and fail with no pattern tied to disturbance size, so treat B and C as indicative only (one run per setting).

## 2b. Finding the driving breaking points (final model, routes finished out of 4)
Noise (out of 255): 0 gives 3, 1 gives 3, 2 gives 3, 4 gives 3, 8 gives 4, 16 gives 3, 32 gives 3, 64 gives 4, 128 gives 1.
Brightness (% of normal): 10 gives 0, 20 gives 3, 40 gives 2, 60 gives 3, 80 gives 3, 100 gives 3, 120 gives 3, 140 gives 3, 180 gives 3, 220 gives 4, 300 gives 3, 450 gives 0.
- Route D finishes from 20% to 300% brightness and through noise of +-128. It fails at 10% and 450% brightness.
- Route A finishes from 20% to 300% brightness and through noise of +-64. It fails at 10% and 450% brightness and at noise of +-128.
- Route B first fails at 40% brightness (a crash at 45% of the route). It also fails at 10% and 450% brightness and at noise of +-128.
- Route C fails in most runs but passes now and then (for example at noise +-64 and brightness 220%), so it stays indicative only.
- Lane quality gets worse before the car fails. At 180% to 300% brightness routes A and B spend 26 to 43% of the drive more than 1 m off center, compared with under 6% at normal brightness.
- The gap with verification: random noise needs about +-64 to +-128 out of 255 to break the driver, while the formal check can only guarantee the steering decision up to about +-0.25 out of 255. That is roughly 250 to 500 times smaller.
- Figure: `results/figs_phase2/fig4_thresholds.png` (script: `make_threshold_fig.py`).

## 3. RTAMT (runtime monitoring on the recorded drives)
Property: always |lateral offset| <= 2.0 m (touching the lane line is allowed, leaving the lane is not). A second tier at 3.5 m flags a full lane departure.
Margin is in meters. Positive means the rule held for the whole drive.
- Routes A, B, and D hold a margin of about +0.2 to +0.8 m in almost every setting.
- Successful drives that still strayed: route A at brightness 140% (-3.7 m, the offset ramped up smoothly, so it is real drift), and route C at noise +-4 and +-8 (about -1 m). The simulator's success flag does not capture this.
- Route B's two crashes (brightness 120%, noise +-4, both at about 93%) show a positive margin (+0.3 m): the car crashed while inside its lane, so a lateral-offset rule cannot see that failure. Limitation: our traces have no distance-to-obstacle signal.
- Full table: `results/rtamt_summary.csv` (script: `rtamt_monitor.py`).

## 4. Formal verification (auto_LiRPA CROWN bounds, 60 frames, roads 27-29)
Question per frame: for every image inside the disturbance box, does the steering slot stay within one slot of the clean choice?
Each frame ends as verified (bounds prove it), falsified (an attack found a flipping image), or unknown.
| Pixel noise (out of 255) | Verified | Falsified | Unknown |
|---|---|---|---|
| +-0 (control) | 60 | 0 | 0 |
| +-0.25 | 45 | 12 | 3 |
| +-0.5 | 21 | 25 | 14 |
| +-1 | 0 | 40 | 20 |
| +-2 | 0 | 59 | 1 |
| +-4, +-8 | 0 | 60 | 0 |

Main finding: the guarantee only holds for very small noise. In simulation, random noise up to +-8/255 did not break routes A and D, but a worst-case pattern flips the decision at +-1/255 on most frames.
Random testing gave a false sense of safety.
The round 2 model (cnn_mac9e) gave nearly the same picture (47, 10, 0 verified at +-0.25, +-0.5, +-1, in `results/verify_summary_9e.csv`). Better driving did not make the network more robust.
Brightness caveat: the brightness box lets every pixel change independently, which is far larger than a real lighting change (all pixels scale together), so those falsified rows must not be read as "brightness breaks it".
The strict version (same slot, not just within one slot) is also in `results/verify_summary.csv` (script: `verify_cnn.py`).

## Limits to state in the report
- One simulator run per setting. Results on B and C are noisy.
- We looked at the 4 test routes after every training round, so they are slightly tuned. Fresh roads would be a cleaner test.
- Verification frames come from the autopilot's drives on three held-out roads, not from the CNN's own drives.
- A verified frame is a one-frame guarantee, not a closed-loop guarantee.
- The attack is a short 20-step search, so "unknown" frames could go either way.
- Perturbed images are real-valued, not rounded to whole pixel values.

## Files
- `results/*_summary.csv` drive scores per setting, `results/traces/` per-step logs, `results/frames/` snapshots
- `results/rtamt_summary.csv`, `results/verify_summary.csv` (final model), `results/verify_summary_9e.csv` (round 2)
- `results/figs_phase2/` the four report figures
- `models/cnn_mac9c.pt` (imitation only), `cnn_mac9d.pt`, `cnn_mac9e.pt`, `cnn_mac9f.pt` (final)
