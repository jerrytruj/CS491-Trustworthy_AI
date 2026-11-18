# Results summary (for the final report)

All runs: MetaDrive 0.4.3 on a Mac, CNN steering policy `models/cnn_mac9e.pt`.
Training roads are seeds 0-29. Test routes: A=1000, B=1002, C=1005, D=1006 (never trained on).

## What changed from Phase 1 (disclose in the report)
- The network now only chooses steering (9 steering slots). Speed is held near 30 km/h by a fixed rule.
- Steering labels are cleaned with a window-3 median filter, because the autopilot emits one-step full-lock spikes on straight roads.
- The best epoch (by held-out accuracy) is saved instead of the last one.
- Two DAgger rounds were added (the CNN drives, the autopilot labels what it should have done).
- Diego's Phase 1 model did not reproduce on a Mac, so everything was retrained and re-evaluated here.
- Formal checks use auto_LiRPA (CROWN bounds, the engine inside alpha,beta-CROWN), not the full alpha,beta-CROWN program.

## 1. Driver quality by training round (clean drives, route completion)
| Model | A | B | C | D | Routes finished |
|---|---|---|---|---|---|
| Autopilot (expert) | 0.99 | 0.99 | 0.99 | 0.99 | 4/4 |
| Imitation only (cnn_mac9c) | 0.52 | 0.28 | 0.46 | 0.49 | 0/4 |
| DAgger round 1 (cnn_mac9d) | 0.99 | 0.91 | 0.49 | 0.99 | 2/4 |
| DAgger round 2 (cnn_mac9e) | 0.99 | 0.96 | 0.74 | 0.99 | 2/4 |

Mean completion: 0.44, then 0.84, then 0.92. On A and D the CNN's lane deviation (0.08 to 0.20 m mean) matches the autopilot's (0.09 to 0.27 m).
Route B always crashed at the same spot (about 28%) in early models, and C is the hardest route.

## 2. Closed-loop drives with disturbances (success out of 4 routes, cnn_mac9e)
| Setting | A | B | C | D | Success |
|---|---|---|---|---|---|
| Clean | ok | fail | fail | ok | 2/4 |
| Brightness 60% | ok | ok | fail | ok | 3/4 |
| Brightness 80% | ok | fail | ok | ok | 3/4 |
| Brightness 120% | ok | fail | fail | ok | 2/4 |
| Brightness 140% | ok | ok | fail | ok | 3/4 |
| Noise +-1/255 | ok | fail | ok | ok | 3/4 |
| Noise +-2/255 | ok | fail | fail | ok | 2/4 |
| Noise +-4/255 | ok | ok | fail | ok | 3/4 |
| Noise +-8/255 | ok | fail | fail | ok | 2/4 |

Routes A and D never failed under any disturbance. B and C flip between success and failure with no pattern tied to disturbance size, so they sit on the edge (one run per setting, so treat B and C as indicative only).

## 3. RTAMT (runtime monitoring on the recorded drives)
Property: always |lateral offset| <= 2.0 m (stay in own lane, touching the line is allowed). A second tier at 3.5 m flags a full lane departure.
Margin is in meters. Positive means the rule held for the whole drive.
- Autopilot and good CNN drives: about +0.25 to +1.5 m.
- Several drives the simulator calls successes still left their lane: brightness 140% on B (-8.9 m), brightness 80% on C (-5.9 m), noise +-1 on C (-4.3 m), noise +-4 on B (-5.2 m).
- Route B's 28% crash shows a positive margin (+1.2 to +1.5 m): the car crashed while inside its lane, so a lateral-offset rule cannot see that failure. Limitation: our traces have no distance-to-obstacle signal.
- Full table: `results/rtamt_summary.csv` (script: `rtamt_monitor.py`).

## 4. Formal verification (auto_LiRPA CROWN bounds, 60 frames, roads 27-29)
Question per frame: for every image inside the disturbance box, does the steering slot stay within one slot of the clean choice?
Each frame ends as verified (bounds prove it), falsified (an attack found a flipping image), or unknown.
| Pixel noise (out of 255) | Verified | Falsified | Unknown |
|---|---|---|---|
| +-0 (control) | 60 | 0 | 0 |
| +-0.25 | 47 | 7 | 6 |
| +-0.5 | 10 | 19 | 31 |
| +-1 | 0 | 48 | 12 |
| +-2, +-4, +-8 | 0 | 60 | 0 |

Main finding: the guarantee only holds for very small noise. In simulation, random noise up to +-8/255 never broke routes A and D, but the worst-case pattern flips the decision at +-1/255. Random testing gave a false sense of safety.
Brightness caveat: the brightness box lets every pixel change independently, which is far larger than a real lighting change (all pixels scale together), so those falsified rows must not be read as "brightness breaks it".
Strict version (same slot, not just within one slot) is also in `results/verify_summary.csv` (script: `verify_cnn.py`).

## Limits to state in the report
- One simulator run per setting. Results on B and C are noisy.
- Verification frames come from the autopilot's drives on three held-out roads, not from the CNN's own drives.
- A verified frame is a one-frame guarantee, not a closed-loop guarantee.
- The attack is a short 20-step search, so "unknown" frames could go either way.
- Perturbed images are real-valued, not rounded to whole pixel values.

## Files
- `results/*_summary.csv` drive scores per setting, `results/traces/` per-step logs, `results/frames/` snapshots
- `results/rtamt_summary.csv`, `results/verify_summary.csv`
- `models/cnn_mac9c.pt` (round 0), `cnn_mac9d.pt` (round 1), `cnn_mac9e.pt` (round 2, final)
