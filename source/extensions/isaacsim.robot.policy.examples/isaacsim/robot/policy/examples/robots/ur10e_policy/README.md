# UR10e policy assets

This directory contains config and checkpoint for the UR10e reach/lift policy:

- **agent.yaml** – RSL-RL training config (actor/critic layout, PPO, etc.).
- **env.yaml** – Environment config (physics dt, decimation, render_interval, scene, robot init_state).
- **ur10e_policy.pt** – Trained policy checkpoint (copy from your standalone install if not present).

The extension resolves this directory by walking up from `robots/ur10e.py` to find a folder named `ur10e_policy`, so it works from both source and built layouts.
