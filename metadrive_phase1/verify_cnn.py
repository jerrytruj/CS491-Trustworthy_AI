"""Checks the CNN's steering decision on sampled frames under bounded image disturbances.
For each frame: does steering stay in the same slot (strict) or within one slot (loose)
for every image inside the disturbance box? Uses auto_LiRPA bounds plus an attack search."""
import argparse, csv, glob, time
import numpy as np, torch, torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from train_cnn import DrivingCNN


class SteerNet(nn.Module):
    """Image (0 to 1 floats) and extras in, one score per steering slot out."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x, extra):
        h = torch.cat([self.m.conv(x), extra], dim=1)
        h = self.m.fc[1](self.m.fc[0](h))  # linear + ReLU (dropout does nothing when evaluating)
        return self.m.steer_head(h)


def make_box(x, kind, level):
    """Lowest and highest allowed value for every pixel."""
    if kind == "noise":   # every pixel may move by up to +-level (out of 255)
        lo, hi = x - level / 255.0, x + level / 255.0
    else:                 # brightness: every pixel scaled by a factor in [1-level, 1+level]
        lo, hi = x * (1 - level), x * (1 + level)
    return lo.clamp(0, 1), hi.clamp(0, 1)


def attack(net, x, e, lo, hi, c, targets, steps=20):
    """Search the box for an image that flips the decision to one of the target slots."""
    adv = (lo + (hi - lo) * torch.rand_like(x)).detach()
    step = (hi - lo) / 4
    for _ in range(steps + 1):
        adv.requires_grad_(True)
        out = net(adv, e)[0]
        gap = out[targets].max() - out[c]   # above 0 means the decision flipped
        if gap.item() > 0:
            return True
        grad, = torch.autograd.grad(gap, adv)
        adv = torch.min(torch.max(adv.detach() + step * grad.sign(), lo), hi)
    return False


def status(lb_rows, rows, net, x, e, lo, hi, c, targets):
    """verified: bounds prove no flip. falsified: a flipping image was found. else unknown."""
    if all(lb_rows[r] > 0 for r in rows):
        return "verified"
    return "falsified" if attack(net, x, e, lo, hi, c, targets) else "unknown"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cnn_mac9e.pt")
    ap.add_argument("--data", default="data/autopilot_mac")
    ap.add_argument("--n", type=int, default=2, help="frames to check per setting")
    ap.add_argument("--method", default="CROWN", help="CROWN or alpha-CROWN")
    ap.add_argument("--out", default="results/verify_summary.csv")
    args = ap.parse_args()
    torch.manual_seed(0)

    cnn = DrivingCNN(); cnn.load_state_dict(torch.load(args.model, map_location="cpu")); cnn.eval()
    net = SteerNet(cnn).eval()

    files = [f"{args.data}/seed_{s}.npz" for s in (27, 28, 29)]  # roads the network never trained on
    ds = [np.load(f) for f in files]
    images = np.concatenate([d["images"] for d in ds]); extras = np.concatenate([d["extras"] for d in ds])
    pick = np.linspace(0, len(images) - 1, args.n).astype(int)

    x0 = torch.from_numpy(images[pick[0]]).float()[None] / 255.0; e0 = torch.from_numpy(extras[pick[0]])[None]
    bm = BoundedModule(net, (x0, e0), device="cpu",
                       bound_opts={"optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}})
    k = 9  # steering slots

    settings = [("noise", 0), ("noise", 0.25), ("noise", 0.5), ("noise", 1), ("noise", 2),
                ("noise", 4), ("noise", 8), ("brightness", 0.02), ("brightness", 0.05),
                ("brightness", 0.1), ("brightness", 0.2), ("brightness", 0.4)]
    rows_out = []
    for kind, level in settings:
        tally = {"strict": {"verified": 0, "falsified": 0, "unknown": 0},
                 "loose": {"verified": 0, "falsified": 0, "unknown": 0}}
        t0 = time.time()
        for i in pick:
            x = torch.from_numpy(images[i]).float()[None] / 255.0; e = torch.from_numpy(extras[i])[None]
            with torch.no_grad():
                c = int(net(x, e).argmax())
            lo, hi = make_box(x, kind, level)
            targets = [j for j in range(k) if j != c]  # one row per other slot: score(c) - score(j)
            C = torch.zeros(1, len(targets), k)
            for r, j in enumerate(targets):
                C[0, r, c], C[0, r, j] = 1.0, -1.0
            bx = BoundedTensor(x, PerturbationLpNorm(norm=np.inf, x_L=lo, x_U=hi))
            be = BoundedTensor(e, PerturbationLpNorm(norm=np.inf, eps=0.0))
            lb, _ = bm.compute_bounds(x=(bx, be), C=C, method=args.method)
            lb = lb[0].detach()
            far = [r for r, j in enumerate(targets) if abs(j - c) > 1]
            far_slots = [targets[r] for r in far]
            tally["strict"][status(lb, range(len(targets)), net, x, e, lo, hi, c, targets)] += 1
            tally["loose"][status(lb, far, net, x, e, lo, hi, c, far_slots)] += 1 if far else 0
            if not far:
                tally["loose"]["verified"] += 1
        row = dict(disturbance=kind, level=level, frames=len(pick), method=args.method,
                   seconds=round(time.time() - t0, 1))
        for p in ("strict", "loose"):
            for s in ("verified", "falsified", "unknown"):
                row[f"{p}_{s}"] = tally[p][s]
        rows_out.append(row); print(row, flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    print("saved", args.out)
