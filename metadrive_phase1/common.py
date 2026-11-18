"""Shared setup: environment config, fixed routes, discrete action set, observation preprocessing.

Everything else (collect, train, evaluate) imports from here so the routes and
action definitions stay identical across the whole project.
"""
import numpy as np
from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

IMG_SIZE = 84          # camera resolution (square)
STACK = 3              # MetaDrive stacks the last 3 camera frames for us
MAP_BLOCKS = 4         # each route is a procedurally generated 4-block road

# Training data comes from these seeds (different road layouts)
TRAIN_SEEDS = list(range(0, 30))

# The FIXED evaluation routes. Held out from training, never change these,
# Phase 2 disruption experiments must run on exactly these routes.
# Each number is a MetaDrive seed: a randomly built road that comes out the same every time.
FIXED_ROUTES = {
    "route_A": 1000,  # on-ramp, two 4-way intersections, curve
    "route_B": 1002,  # straight, two curves, 4-way intersection
    "route_C": 1005,  # curve, two roundabouts, straight
    "route_D": 1006,  # off-ramp, T-junction, curve, on-ramp
}

# ---- Discrete action set ----
# 7 steering bins x 3 throttle bins = 21 actions.
# 0.5 (sharp turns), 0.2 (normal curves), 0.08 (small lane corrections) 
STEER_BINS = np.array([-1.0, -0.5, -0.2, -0.08, 0.0, 0.08, 0.2, 0.5, 1.0], dtype=np.float32)
# Autopilot holds speed with throttle around 0.03-0.18, so "cruise" sits at 0.05
# (0.1 made the CNN drive ~25% faster than its teacher).
THROTTLE_BINS = np.array([-0.3, 0.05, 0.5], dtype=np.float32)  # brake, cruise, accelerate

# Rounds the autopilot's real amounts to the nearest slot numbers (builds the training answer key).
# Example: [-0.3, 0.45] -> (1, 2) = steering slot 1 (-0.2), throttle slot 2 (accelerate).
def discretize(action):
    """Continuous autopilot action [steer, throttle] -> (steer_idx, throttle_idx)."""
    s = int(np.argmin(np.abs(STEER_BINS - np.clip(action[0], -1, 1))))
    t = int(np.argmin(np.abs(THROTTLE_BINS - np.clip(action[1], -1, 1))))
    return s, t

# Reverse of discretize: turns the network's slot picks into real amounts for the car.
# Example: (1, 2) -> [-0.2, 0.5] = steer slightly right, accelerate.
def to_continuous(steer_idx, throttle_idx):
    return np.array([STEER_BINS[steer_idx], THROTTLE_BINS[throttle_idx]], dtype=np.float32)


def make_env(start_seed, num_scenarios=1, traffic_density=0.0, horizon=1500):
    """One MetaDrive env covering routes start_seed .. start_seed+num_scenarios-1.
    Build it ONCE per process and switch routes with env.reset(seed=...); building a
    new env per route leaks memory. Camera-only observations.

    Navigation markers drawn on the road are turned off so the CNN has to read
    the actual road, not a painted guide line.
    """
    cfg = dict(
        num_scenarios=num_scenarios,
        start_seed=start_seed,
        map=MAP_BLOCKS,
        traffic_density=traffic_density,
        image_observation=True,
        norm_pixel=False,
        stack_size=STACK,
        sensors={"rgb_camera": (RGBCamera, IMG_SIZE, IMG_SIZE)},
        vehicle_config=dict(show_navi_mark=False, show_line_to_navi_mark=False, # Turn off marker guidelines
                            show_dest_mark=False, show_line_to_dest=False),
        horizon=horizon,
        # MetaDrive's default ends the episode the moment a tire touches ANY solid line,
        # which even its own autopilot trips. We only end on actually leaving the road
        # (plus curb hits, handled in hit_curb_or_left_road) and log line touches instead.
        on_continuous_line_done=False,
        use_render=__import__("os").environ.get("DEMO_RENDER") == "1", # GUI is on only when DEMO_RENDER=1
        log_level=50,
    )
    return MetaDriveEnv(cfg)

# The network's GPS: 4 numbers for where the next 2 route checkpoints are
def route_target(agent):
    """Where the route goes: positions of the next two route checkpoints relative to the car,
    as (ahead, sideways) pairs in [-1, 1] (50 m scale). Same idea as TCP's "target point" input.

    MetaDrive's own left/forward/right flags only compare the heading of the current and next
    road piece (road curvature), and never signal which way to go at an intersection, so the
    CNN couldn't learn turns from them.
    """
    navi = agent.navigation.get_navi_info()
    return (np.array([navi[0], navi[1], navi[5], navi[6]], dtype=np.float32) * 2 - 1)


def obs_to_input(obs, agent, corrupt_fn=None):
    """Build model inputs.

    image: (STACK*3, H, W) uint8, channels-first, oldest frame first
    extra: [speed/50, ckpt1_ahead, ckpt1_side, ckpt2_ahead, ckpt2_side]
    corrupt_fn: optional function applied to each HxWx3 uint8 frame (Phase 2 hook
    for imagecorruptions, so disruptions plug in without touching anything else).
    """
    img = obs["image"].astype(np.uint8)  # (H, W, 3, STACK)
    frames = []
    for k in range(img.shape[-1]):
        f = img[..., k]
        if corrupt_fn is not None:
            f = np.asarray(corrupt_fn(f)).astype(np.uint8)
        frames.append(f.transpose(2, 0, 1)) # regroup pixel-by-pixel -> red/green/blue layers (PyTorch's layout)
    image = np.concatenate(frames, axis=0) # stack 3 frames x 3 color layers = 9 layers (9, 84, 84)
    extra = np.concatenate([[agent.speed_km_h / 50.0], route_target(agent)]).astype(np.float32)
    return image, extra


def touching_solid_line(agent):
    return bool(agent.on_white_continuous_line or agent.on_yellow_continuous_line)


def hit_curb_or_left_road(agent, info):
    return bool(info["out_of_road"] or agent.crash_sidewalk)


def lateral_offset(agent):
    """Signed distance (m) from the center of the nearest lane ON THE ROUTE.

    Measured against the navigation's reference lanes for the current/next road segment,
    not agent.lane (which can snap to an unrelated lane inside intersections and gave
    bogus 20 m offsets). Being in a different lane of the same road is not penalized.
    """
    nav = agent.navigation
    lanes = list(nav.current_ref_lanes or []) + list(nav.next_ref_lanes or [])
    best = None # "best" = closest route lane so far (the lane the car is in)
    for ln in lanes:
        lon, lat = ln.local_coordinates(agent.position) # lon = how far along the lane, lat = how far sideways from its center
        if -2.0 <= lon <= ln.length + 2.0 and (best is None or abs(lat) < abs(best)):
            best = lat
    if best is None:
        _, best = agent.lane.local_coordinates(agent.position)
    return float(best)
