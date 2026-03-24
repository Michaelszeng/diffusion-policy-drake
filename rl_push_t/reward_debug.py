"""
Interactive reward debugger for the Drake Push-T environment.

Use Meshcat sliders to move the pusher in real time and watch reward
components update live in the terminal.

The pusher position is set directly from the sliders (bypassing the RL
delta-action integration), so you can freely explore the reward landscape.
The T-block is simulated normally and will react to pusher contact.
Episode resets when max_episode_steps is reached or when the "Reset" button
is clicked in Meshcat.

Usage:
    python rl_push_t/test_reward_debug.py
    python rl_push_t/test_reward_debug.py --cfg_path rl_push_t/configs/rl_env.yaml
"""

import argparse
import time

import numpy as np
from pydrake.all import Meshcat

from rl_push_t.envs.push_t_gym_env import PushTDrakeEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_path", default="rl_push_t/configs/rl_env.yaml")
    args = parser.parse_args()

    meshcat = Meshcat()
    print(f"Meshcat: {meshcat.web_url()}\n")

    env = PushTDrakeEnv(cfg_path=args.cfg_path, meshcat=meshcat)

    goal_x = env._slider_goal_pose.x
    goal_y = env._slider_goal_pose.y
    control_dt = env._rl_cfg.get("control_dt", 0.1)
    max_steps = env._rl_cfg.get("max_episode_steps", 200)

    # Sliders expressed relative to goal (matches observation space)
    RANGE = 0.25
    meshcat.AddSlider(
        "pusher_rel_x",
        min=-RANGE, max=RANGE, step=0.005,
        value=env._pusher_start_pose.x - goal_x,
    )
    meshcat.AddSlider(
        "pusher_rel_y",
        min=-RANGE, max=RANGE, step=0.005,
        value=env._pusher_start_pose.y - goal_y,
    )
    meshcat.AddButton("Reset Episode")

    env.reset()
    meshcat.StartRecording()

    print("Sliders control the pusher position (relative to goal).")
    print("Click 'Reset Episode' in Meshcat to reset. Ctrl+C to exit.\n")
    print(f"{'rot':>8}  {'trans':>8}  {'ee':>8}  {'total':>8}  {'overlap':>8}  success")

    reset_clicks = 0

    try:
        while True:
            # ── Check for manual reset ─────────────────────────────────────
            clicks = meshcat.GetButtonClicks("Reset Episode")
            if clicks > reset_clicks:
                reset_clicks = clicks
                env.reset()
                meshcat.StopRecording()
                meshcat.StartRecording()

            # ── Set pusher position from sliders ──────────────────────────
            rel_x = meshcat.GetSliderValue("pusher_rel_x")
            rel_y = meshcat.GetSliderValue("pusher_rel_y")
            env._current_position[:] = [goal_x + rel_x, goal_y + rel_y]
            env._rl_source.set_action(env._current_position)

            # ── Advance simulation one control step ───────────────────────
            env._current_sim_time += control_dt
            env._simulator.AdvanceTo(env._current_sim_time)
            env._step_count += 1

            # ── Compute and display reward ─────────────────────────────────
            obs = env._get_obs()
            env._compute_reward(obs, debug=True)

            # ── Auto-reset at episode end ──────────────────────────────────
            if env._step_count >= max_steps:
                env.reset()
                meshcat.StopRecording()
                meshcat.StartRecording()

    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
