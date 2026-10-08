"""Step 3b: train the CNN driving policy by imitating the autopilot.

Input: last 3 camera frames (84x84 RGB) + speed + route target points (next 2 checkpoints).
Output: two classification heads, steering bin (7) and throttle bin (3) = 21 discrete actions.

Usage:
    python train_cnn.py --data data/part0.npz data/part1.npz --out models/cnn_policy.pt
"""
import argparse, os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from common import STEER_BINS, THROTTLE_BINS, STACK


class DrivingCNN(nn.Module):
    def __init__(self, in_ch=3 * STACK, n_extra=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7 + n_extra, 256), nn.ReLU(), nn.Dropout(0.2))
        self.steer_head = nn.Linear(256, len(STEER_BINS))
        self.throttle_head = nn.Linear(256, len(THROTTLE_BINS))

    def forward(self, image_uint8, extra):
        x = self.conv(image_uint8.float() / 255.0)
        h = self.fc(torch.cat([x, extra], dim=1))
        return self.steer_head(h), self.throttle_head(h)

    @torch.no_grad()
    def act(self, image, extra):
        """numpy inputs for one timestep -> (steer_idx, throttle_idx)"""
        s, t = self(torch.from_numpy(image)[None], torch.from_numpy(extra)[None])
        return int(s.argmax()), int(t.argmax())


def load_data(paths):
    """paths: folders of per-route seed_*.npz files (or single .npz files)."""
    import glob
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "seed_*.npz")), key=lambda f: int(f.split("_")[-1][:-4])) \
            if os.path.isdir(p) else [p]
    ds = [np.load(f) for f in files]
    cat = lambda k: np.concatenate([d[k] for d in ds])
    seed_of = lambda f: int(f.split("_")[-1][:-4]) if os.path.basename(f).startswith("seed_") else -1
    seeds = np.concatenate([np.full(len(d["steer"]), seed_of(f)) for d, f in zip(ds, files)])
    return cat("images"), cat("extras"), cat("steer"), cat("throttle"), seeds


def augment(img):
    """Small brightness/contrast jitter so the model isn't tuned to one exact lighting."""
    b = img.shape[0]
    scale = torch.empty(b, 1, 1, 1).uniform_(0.85, 1.15)
    shift = torch.empty(b, 1, 1, 1).uniform_(-15, 15)
    return (img.float() * scale + shift).clamp(0, 255)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/autopilot"],
                    help="data folders of per-route seed_*.npz files")
    ap.add_argument("--out", default="models/cnn_policy.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    X, E, S, T, seeds = load_data(args.data)
    n = len(X)
    print(f"{n} samples | steer dist {np.bincount(S, minlength=len(STEER_BINS))} "
          f"| throttle dist {np.bincount(T, minlength=len(THROTTLE_BINS))}")
    # Validate on whole held-out routes (the last 3 training seeds), never trained on.
    # (A random frame split would leak near-duplicate neighboring frames.)
    val_seeds = sorted(set(seeds))[-3:]
    is_val = np.isin(seeds, val_seeds)
    val, tr = np.where(is_val)[0], np.where(~is_val)[0]
    print(f"validation routes: {val_seeds} ({len(val)} frames)")

    # class weights so rare hard-turn / accelerate actions aren't ignored
    def weights(y, k, power):
        c = np.bincount(y[tr], minlength=k).astype(np.float32)
        w = np.where(c > 0, (c.sum() / np.maximum(c, 1)) ** power, 0.0)
        return torch.tensor(w / w[c > 0].mean(), dtype=torch.float32)
    ws, wt = weights(S, len(STEER_BINS), 0.5), weights(T, len(THROTTLE_BINS), 1.0)

    Xt, Et, St, Tt = map(torch.from_numpy, (X, E, S.astype(np.int64), T.astype(np.int64)))
    model = DrivingCNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    history = []
    for ep in range(args.epochs):
        model.train(); perm = torch.from_numpy(np.random.permutation(tr)); tot = 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            ps, pt = model(augment(Xt[b]), Et[b])
            loss = F.cross_entropy(ps, St[b], weight=ws) + 0.5 * F.cross_entropy(pt, Tt[b], weight=wt)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vb = torch.from_numpy(val)
            ps, pt = model(Xt[vb], Et[vb])
            acc_s = (ps.argmax(1) == St[vb]).float().mean().item()
            acc_t = (pt.argmax(1) == Tt[vb]).float().mean().item()
            # "within one bin" steering accuracy, a fairer measure for ordered bins
            near_s = ((ps.argmax(1) - St[vb]).abs() <= 1).float().mean().item()
        history.append(dict(epoch=ep + 1, train_loss=tot / len(tr), val_steer_acc=acc_s,
                            val_steer_within1=near_s, val_throttle_acc=acc_t))
        print(history[-1], flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    with open(os.path.splitext(args.out)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=1)
    print("saved", args.out)
