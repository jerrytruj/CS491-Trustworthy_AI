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
# Screened seeds 1000-1009: the autopilot itself fails 1001, 1003, 1007 (curb hits at
# T-junctions / roundabouts), so those can't be fair tests of a policy that imitates it.
FIXED_ROUTES = {
    "route_A": 1000,  # on-ramp, two 4-way intersections, curve
    "route_B": 1002,  # straight, two curves, 4-way intersection
    "route_C": 1005,  # curve, two roundabouts, straight
    "route_D": 1006,  # off-ramp, T-junction, curve, on-ramp
}

# ---- Discrete action set ----
# 7 steering bins x 3 throttle bins = 21 actions.
STEER_BINS = np.array([-0.5, -0.2, -0.08, 0.0, 0.08, 0.2, 0.5], dtype=np.float32)
# Autopilot holds speed with throttle around 0.03-0.18, so "cruise" sits at 0.05
# (0.1 made the CNN drive ~25% faster than its teacher).
THROTTLE_BINS = np.array([-0.3, 0.05, 0.5], dtype=np.float32)  # brake, cruise, accelerate


def discretize(action):
    """Continuous autopilot action [steer, throttle] -> (steer_idx, throttle_idx)."""
    s = int(np.argmin(np.abs(STEER_BINS - np.clip(action[0], -1, 1))))
    t = int(np.argmin(np.abs(THROTTLE_BINS - np.clip(action[1], -1, 1))))
    return s, t


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
        vehicle_config=dict(show_navi_mark=False, show_line_to_navi_mark=False,
                            show_dest_mark=False, show_line_to_dest=False),
        horizon=horizon,
        # MetaDrive's default ends the episode the moment a tire touches ANY solid line,
        # which even its own autopilot trips. We only end on actually leaving the road
        # (plus curb hits, handled in hit_curb_or_left_road) and log line touches instead.
        on_continuous_line_done=False,
        use_render=False,
        log_level=50,
    )
    return MetaDriveEnv(cfg)


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
        frames.append(f.transpose(2, 0, 1))
    image = np.concatenate(frames, axis=0)
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
    best = None
    for ln in lanes:
        lon, lat = ln.local_coordinates(agent.position)
        if -2.0 <= lon <= ln.length + 2.0 and (best is None or abs(lat) < abs(best)):
            best = lat
    if best is None:
        _, best = agent.lane.local_coordinates(agent.position)
    return float(best)
