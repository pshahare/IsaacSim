# Migrating changes from standalone Isaac Sim to this fork (public 5.1.0)

This document describes the modifications that were brought from the **standalone Isaac Sim install** (`ds/isaacsim`) into this **public 5.1.0 fork** (`ds/ps/IsaacSim`) so you can use the same UR10e policy and examples when building from source.

## What was added

All of the following live in the **fork** and are part of the repo:

### 1. UR10e policy extension (source tree)

- **`source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/ur10e.py`**  
  UR10e + Robotiq 2f-140 policy class: loads an RSL-RL checkpoint and env config, runs reach/lift with differential IK.

- **`source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/ur10e_policy/`**  
  Policy assets used by `ur10e.py`:
  - `agent.yaml` – training config (actor/critic, PPO).
  - `env.yaml` – environment config (physics dt, decimation, scene, robot init_state).
  - `ur10e_policy.pt` – trained checkpoint (copied from your standalone install).
  - `README.md` – short description of the folder.

- **`source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/__init__.py`**  
  Updated to export `UR10eReachTargetPolicy`.

### 2. Standalone examples

- **`source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py`**  
  Standalone script: table + DexCube + UR10e, goal position from args or default.

- **`source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_standalone.py`**  
  Simpler standalone: ground + cube + UR10e, fixed goal.

### 3. Tests

- **`source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/tests/test_ur10e.py`**  
  Async tests: add UR10e, check DOF count and articulation API, run reach and check joint motion.

## What was not copied (and why)

- **`.vscode/launch.json` and `tasks.json`**  
  The standalone uses paths like `kit/`, `apps/`, `exts/`. This repo is build-from-source and uses `_build/linux-x86_64/release/` (or debug). Keep using the **fork’s** existing `.vscode` config for building and running from the built tree; no need to overwrite with the standalone versions.

- **`VERSION`**  
  The fork keeps its own version (e.g. `5.1.0-rc.19`). The standalone’s `5.1.0-rc.19+release....` is for the prebuilt package.

- **`ur10e_reach_scene.py` (ROS2)**  
  Lives under `isaacsim.ros2.bridge` in the standalone. Can be added to the fork in the same path under `source/standalone_examples/api/isaacsim.ros2.bridge/` if you need it.

- **Large/binary or install-specific content**  
  Things like `kit/`, `exts/` (prebuilt), `python_packages/`, launcher scripts, `pick.txt`, `sim_logs.txt`, etc., are not part of the fork and stay in the standalone install.

## How to run in the fork (after build)

1. **Build the repo** (see main README / build docs):
   ```bash
   ./build.sh
   ```

2. **Run a UR10e standalone example** using the built kit Python (paths may differ by platform):
   ```bash
   # From repo root
   ./_build/linux-x86_64/release/kit/python/bin/python3 \
     source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py
   ```
   Or with a goal:
   ```bash
   ./_build/linux-x86_64/release/kit/python/bin/python3 \
     source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py --goal 0.5 0.0 0.35
   ```

3. **Run UR10e tests** (from repo root, using the built env):
   ```bash
   # Use the test runner and args from the extension’s config; the test module is
   # isaacsim.robot.policy.examples.tests.test_ur10e
   ```

Policy assets are resolved by walking up from `robots/ur10e.py` to find the `ur10e_policy` directory, so they work both in the source layout and in the built `exts` output.

## Summary

| Item | Standalone (ds/isaacsim) | Fork (ds/ps/IsaacSim) |
|------|--------------------------|------------------------|
| UR10e policy class | `exts/.../robots/ur10e.py` | `source/extensions/.../robots/ur10e.py` |
| Policy data | Repo root `ur10e_policy/` | `.../robots/ur10e_policy/` |
| Standalone examples | `standalone_examples/.../ur10e_*.py` | `source/standalone_examples/.../ur10e_*.py` |
| Tests | `exts/.../tests/test_ur10e.py` | `source/extensions/.../tests/test_ur10e.py` |
| .vscode / VERSION | Use for standalone install | Use fork’s for build-from-source |

Your changes from the standalone are now reflected in the public 5.1.0 fork so you can build from source and keep the same UR10e policy behavior and examples.
