"""Dynamic pick task: DROID real robot, pick up the cube from a randomized drop pose.

Same `PickBlocksEnv` and same success detector as `pick` -- what makes it dynamic is the
reset: after every episode the cube is released at a fresh x/y offset (with a required
minimum, so it never lands twice in the same spot) and lower than the reset pose, and the
lift bar is higher. It runs at 30 Hz, fast enough that a pi0.5 forward pass no longer fits
inside one control step, which is what the real-time chunking pipeline
(`scripts/dynamic_pick/`) is for.

Everything not listed below is inherited from `pick`.
"""

from configs.task import pick


def get_config():
    config = pick.get_config()

    config.env_name = "dynamic_pick"

    # 30 Hz. `--delay` in scripts/dynamic_pick is sized off this: ceil(latency_ms * hz / 1000).
    config.control_hz = 30
    config.auto_reset_steps = 200  # ~6.7 s per episode at 30 Hz

    # Success = an object held between the fingers and lifted past success_height. Kept well
    # under the 0.45 z ceiling inherited from `pick`, so the lift cannot trip reached_boundary.
    config.success_height = 0.3
    # A step count, not a duration: 6 steps at 30 Hz is 200 ms of dwell (`pick` gets 600 ms at
    # 10 Hz). Raise it if a brush past the height threshold is being scored as a grasp.
    config.success_consecutive_steps = 6

    # Re-randomize the drop pose after each success: |offset| drawn in [min, max] per axis with
    # a random sign, and released 8 cm below the reset pose so the cube is not dropped from height.
    config.success_reset_randomize_magnitude = 0.055
    config.success_reset_randomize_min_magnitude = 0.02
    config.success_reset_drop_height_offset = -0.08

    # HQ clip camera: the one ZED the policy does not read, opened by the recorder itself at
    # record quality. Bare serial, not an image key.
    # REPLACE with your own ZED camera serial (see README "DROID Setup").
    config.record_camera = "29838012"
    config.record_resolution = (1280, 720)
    config.record_fps = 30
    config.record_post_roll = 3.0         # seconds of clip after the verdict
    config.eval_post_done_sleep_s = 3.0   # same wait on every eval episode

    # spacemouse scaling: more rotation authority than `pick` for chasing a moved cube
    config.collect_max_rot_vel = 0.3

    return config
