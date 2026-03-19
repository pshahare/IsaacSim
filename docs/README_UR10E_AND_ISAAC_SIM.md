# UR10e Use Case & Isaac Sim vs Isaac Lab Arena — Detailed README

**Start here** for the UR10e use case: this doc gives the full picture. For step-by-step run instructions, see **docs/RUN_UR10E_USECASE.md**.

This document explains (1) everything we did for the UR10e policy use case in this repo, (2) **full technical details** (formulas, observation/action layout, physics, coordinates), and (3) how **Isaac Sim** and **Isaac Lab / Isaac Lab Arena** differ, with a **dedicated section on how inference differs** between the two.

---

# Part 1: What We Did (UR10e Use Case)

## 1.1 Goal

Bring the **UR10e reach/lift policy** (trained and run in a **standalone Isaac Sim install**) into this **public 5.1.0 fork** so that:

- Others can clone the fork, build or use an existing install, and run the same use case.
- The code lives in version control (e.g. `feature/stackbot` or `feature/ur10e-policy`) instead of only in a local standalone tree.

## 1.2 Repositories Involved

| Location | Role |
|----------|------|
| **ds/isaacsim** | Your **modified** side: a **prebuilt/standalone** Isaac Sim install (kit/, exts/, apps/, `python.sh`). Contains the UR10e policy code and `ur10e_policy/` data you developed/ran. |
| **ds/ps/IsaacSim** | Your **fork** of the public Isaac Sim repo (build-from-source: `source/`, premake, `build.sh`, `_build/`). No prebuilt kit until you build. |

The two trees have **different layouts**; only a few paths (e.g. `.vscode/launch.json`, `VERSION`) exist in both, and those differ by design (standalone vs built paths).

## 1.3 Changes We Made in the Fork

All of the following were added or updated **in the fork** (`ds/ps/IsaacSim`):

### Code & data

1. **UR10e policy extension**
   - `source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/robots/ur10e.py`  
     Class `UR10eReachTargetPolicy`: loads an RSL-RL checkpoint and env config, runs reach/lift with differential IK for UR10e + Robotiq 2f-140.
   - `source/extensions/.../robots/ur10e_policy/`  
     Directory with:
     - `agent.yaml` (training config: actor/critic, PPO),
     - `env.yaml` (environment: physics dt, decimation, scene, robot init_state),
     - `ur10e_policy.pt` (trained checkpoint),
     - `README.md`.
   - `source/extensions/.../robots/__init__.py`  
     Updated to export `UR10eReachTargetPolicy`.

2. **Standalone examples**
   - `source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py`  
     Table + DexCube + UR10e, goal from args or default.
   - `source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_standalone.py`  
     Simpler scene: ground + cube + UR10e, fixed goal.

3. **Tests**
   - `source/extensions/.../tests/test_ur10e.py`  
     Async tests: add UR10e, check DOF/articulation API, run reach and assert joint motion.

### Docs (this repo)

- `docs/MIGRATION_FROM_STANDALONE.md` — What was copied from standalone to fork and what was not (and why).
- `docs/PUSH_UR10E_CHANGES.md` — How to commit and push UR10e changes to your fork (branch, push, what others clone).
- `docs/RUN_UR10E_USECASE.md` — Step-by-step: clone, build (or use standalone), set env, run examples and tests.
- `docs/README_UR10E_AND_ISAAC_SIM.md` — This file (overview + Isaac Sim vs Isaac Lab Arena).

## 1.4 How to Run the Use Case

- **From the fork after build:**  
  See **docs/RUN_UR10E_USECASE.md** (source env, then run with the built kit Python and script path).
- **From your standalone install (no build):**  
  From `ds/isaacsim`:  
  `./python.sh standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py`

## 1.5 Sharing with Others

- Branch with the changes: e.g. `feature/stackbot` or `feature/ur10e-policy`.
- Others clone the fork, checkout that branch, then either:
  - **Build from source** and run as in RUN_UR10E_USECASE.md, or  
  - **Copy** the extension + standalone scripts into their own Isaac Sim install and run with their `python.sh`.

Details: **docs/PUSH_UR10E_CHANGES.md**.

## 1.6 What We Did *Not* Copy

- **.vscode/launch.json, tasks.json** — Fork uses `_build/...` paths; standalone uses `kit/`, `apps/`, `exts/`. Keep the fork’s config for build-from-source.
- **VERSION** — Fork keeps its own version string; standalone uses a release build id.
- **ur10e_reach_scene.py (ROS2)** — Can be added under `source/standalone_examples/api/isaacsim.ros2.bridge/` if needed.
- **Prebuilt kit, exts, python_packages, launcher scripts** — Those stay in the standalone install; the fork only has source and build output after `./build.sh`.

---

## 1.7 Technical specification of the UR10e policy (formulas and details)

This section documents every technical detail of the UR10e reach/lift policy used in this repo: robot model, observation/action spaces, Euclidean distance and frame transforms, differential IK formula, network layout, and simulation timing.

### 1.7.1 Robot and DOFs

- **Robot:** UR10e arm + Robotiq 2f-140 gripper (USD variant selected in code).
- **Total actuated DOFs:** 14  
  - Arm: 6 (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3).  
  - Gripper: 1 (finger_joint); remaining DOFs are passive/coupled in the gripper model.
- **Policy action dimension:** 7 (6 arm-related + 1 gripper command).
- **End-effector frame:** wrist_3_link with a fixed offset from the link origin: position offset **p_offset = (0, 0, 0.24)** m, orientation offset **quat_offset = (1, 0, 0, 0)** (wxyz, identity). Used to compute “effective” EE pose for observations and Jacobian.

All positions and distances in the simulation use **Isaac Sim world frame: Z-up, +X forward, right-handed, meters**.

### 1.7.2 Observation vector (53 dimensions)

The policy is trained on a **53-dimensional** observation vector. Layout (exact indices and meaning):

| Index range | Dimension | Symbol / name | Description |
|-------------|-----------|----------------|-------------|
| 0–6 | 7 | last_action | Previous step’s action (7D). |
| 7–20 | 14 | joint_pos_rel | Joint position relative to default: **q − q_default** (all 14 DOFs). |
| 21–34 | 14 | joint_vel_rel | Joint velocity relative to default: **v − v_default** (v_default = 0). |
| 35–37 | 3 | eef_pos | End-effector position in **world frame** (x, y, z) in meters. |
| 38–41 | 4 | eef_quat | End-effector orientation in **world frame**, quaternion (w, x, y, z). |
| 42 | 1 | gripper_pos | Current finger_joint position (scalar). |
| 43–45 | 3 | goal_position | Target lift position (x, y, z) in world frame (command). |
| 46–49 | 4 | goal_quat | Target orientation for command; fixed to (1, 0, 0, 0) in this task. |
| 50–52 | 3 | object_pos_in_robot_frame | Object position expressed in **robot base frame**: **p_object_world − p_robot_world**. |

So the observation is **o = (last_action, joint_pos_rel, joint_vel_rel, eef_pos, eef_quat, gripper_pos, goal_position, goal_quat, object_pos_in_robot_frame)**.

**End-effector pose:** If the wrist_3_link prim is valid, EE pose is computed from that prim plus the fixed offset; otherwise it falls back to robot base pose + offset. All in **world frame** (Z-up).

### 1.7.3 Euclidean distance (formula and where it appears)

**Euclidean distance** between two points **p** and **q** in 3D (e.g. world frame, in meters):

\[
d(p, q) = \|p - q\| = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2 + (p_z - q_z)^2}.
\]

In code/numpy: `np.linalg.norm(p - q)` or `np.sqrt(np.sum((p - q)**2))`.

In this policy:

- The **observation** does not pass a single “distance” scalar; it passes **goal_position**, **eef_pos**, and **object_pos_in_robot_frame**. Downstream, the network can implicitly use distances (e.g. goal–EE or object–EE) from these vectors.
- **Object position in robot frame** is the vector difference (not the Euclidean distance):
  \[
  \texttt{object\_pos\_in\_robot\_frame} = p_{\text{object}}^{\text{world}} - p_{\text{robot}}^{\text{world}}.
  \]
  So the policy sees the **direction and magnitude** (which includes distance) of the object relative to the robot base.
- If you need the **distance from robot to object** (e.g. for logging or reward in training):  
  \( d_{\text{robot–object}} = \|p_{\text{object}}^{\text{world}} - p_{\text{robot}}^{\text{world}}\| \).  
  Similarly, **distance from EE to goal**: \( d_{\text{EE–goal}} = \|p_{\text{goal}}^{\text{world}} - p_{\text{EE}}^{\text{world}}\| \).

All positions are in **meters**; distances are in **meters**.

### 1.7.4 Action space and scaling

- **Policy output (raw):** 7D vector.  
  - **[0:6]** — End-effector pose deltas in world frame: **(Δx, Δy, Δz, Δroll, Δpitch, Δyaw)** (in meters and radians).  
  - **[6]** — Gripper command: **> 0** → close (target 0.7), **≤ 0** → open (target 0.0).

- **Action scale:** Raw EE deltas are multiplied by **action_scale = 0.1** before differential IK:
  \[
  \Delta\xi_{\text{used}} = 0.1 \cdot a_{0:6}.
  \]

- **Differential IK** converts the 6D EE delta into **joint position deltas** for the 6 arm joints; the gripper is set directly from the 7th action.

### 1.7.5 Differential IK (Damped Least Squares, DLS)

The policy outputs **end-effector pose deltas**; we convert them to **joint deltas** using the **Jacobian** of the EE pose w.r.t. joint angles and **Damped Least Squares (DLS)** to avoid singularities.

- **Jacobian:** **J** = (6 × 6) for the 6 arm joints at the end-effector body (wrist_3_link). Taken from `robot._articulation_view.get_jacobians()`; row corresponding to the EE body index, columns for the 6 arm joint indices.

- **DLS formula:** Solve for joint delta **Δq** given EE delta **Δξ**:
  \[
  (J J^T + \lambda^2 I)\, y = \Delta\xi, \qquad \Delta q = J^T y.
  \]
  Equivalently: **Δq = J^T (J J^T + λ² I)⁻¹ Δξ**, with **λ = 0.01** (damping).

- **Clamping:** Each component of **Δq** is clamped to **±0.1** rad before applying. Joint targets are then clamped to **±2π**.

- **Safety:** If Jacobian or solve fails (e.g. singular), **Δq** is set to zero. If any joint position is NaN/Inf, the robot is reset to the training default pose.

### 1.7.6 Actor network (MLP)

- **Architecture:** Fully connected MLP: **Linear(53, 256) → ELU → Linear(256, 128) → ELU → Linear(128, 64) → ELU → Linear(64, 7)**. So hidden dims **(256, 128, 64)** and **ELU** activation.
- **Input:** 53D observation (same layout as above).  
- **Output:** 7D raw action (EE deltas + gripper).  
- **Checkpoint:** Only the **actor** weights are loaded from the RSL-RL checkpoint (keys starting with `"actor."`); the critic is not used at inference.

### 1.7.7 Training config (from agent.yaml / env)

- **Algorithm:** PPO (Proximal Policy Optimization).  
- **Policy:** ActorCritic; actor/critic hidden dims (256, 128, 64), ELU.  
- **Decimation:** Policy is evaluated every **4** physics steps (decimation = 4).  
- **Physics dt:** **0.005** s (200 Hz). So effective policy rate = 200/4 = **50 Hz**.  
- **Render interval:** 2 (every 2nd physics step for rendering, if applicable).

So **simulation timestep = 0.005 s**, **policy step = 0.02 s**.

### 1.7.8 Default pose (training init)

The robot’s default joint position used in the observation (joint_pos_rel, joint_vel_rel) is the **training default**:

- Arm (rad): **[π, −π/2, π/2, −π/2, −π/2, 0]** (shoulder_pan through wrist_3).  
- Gripper: 0.  
- Total 14D; velocities default to 0.

### 1.7.9 Coordinate frames (summary)

- **World (Isaac Sim/PhysX):** Z-up, +X forward, right-handed, meters. All **goal_position**, **eef_pos**, **object_position** from the stage are in this frame unless we explicitly convert (e.g. to robot frame).
- **Robot base frame:** Origin at robot base; same orientation as world in our examples. So **object_pos_in_robot_frame = p_object_world − p_robot_world**.
- **Quaternions:** Stored as **(w, x, y, z)** in the observation (eef_quat, goal_quat).

---

# Part 2: Isaac Sim vs Isaac Lab / Isaac Lab Arena (Detailed Comparison)

Below is a concise but detailed comparison of **Isaac Sim** (the simulation platform you used for the UR10e use case) and **Isaac Lab / Isaac Lab Arena** (NVIDIA’s robot-learning and policy-evaluation stack). The UR10e policy in this repo runs **in Isaac Sim** (standalone or built from this fork); it is not an Isaac Lab Arena app, but the comparison clarifies how the two ecosystems differ.

## 2.1 What Each One Is

| | Isaac Sim | Isaac Lab / Isaac Lab Arena |
|--|-----------|-----------------------------|
| **What it is** | NVIDIA’s **core robotics simulation platform** built on Omniverse (USD, Kit, PhysX). General-purpose: visualization, physics, sensors, ROS, synthetic data, scripting, extensions. | **Isaac Lab**: GPU-accelerated **robot-learning framework** (RL, imitation, motion planning) built *on top of* Isaac Sim. **Isaac Lab Arena**: A **policy-evaluation and environment-creation** layer on top of Isaac Lab (task curation, diversification, large-scale benchmarking). |
| **Primary use** | Simulation, rendering, ROS integration, SDG, digital twins, custom scripts (e.g. UR10e policy inference). | Training and evaluating learned policies at scale; standardized tasks, robots, and metrics. |
| **Open source** | Public source repo (e.g. isaac-sim/IsaacSim); also prebuilt “Isaac Sim” installers. | Isaac Lab and Isaac Lab Arena are open-source (e.g. isaac-sim/IsaacLab, isaac-sim/IsaacLab-Arena). |

So: **Isaac Sim** = the underlying simulator and app platform. **Isaac Lab** = a framework that uses Isaac Sim for robot learning. **Isaac Lab Arena** = a higher-level framework for composing tasks and evaluating policies, built on Isaac Lab.

## 2.2 Physics Engine

| | Isaac Sim | Isaac Lab (current) | Isaac Lab 3.0 Beta (direction) |
|--|-----------|----------------------|--------------------------------|
| **Default engine** | **PhysX** (NVIDIA GPU physics). Same engine used for rigid bodies, articulations, and contact in the Isaac Sim you run. | Historically **PhysX** (via Isaac Sim). | Transitioning to support **multiple** backends: PhysX and **Newton** (MuJoCo-Warp–style solver). Goal: pluggable physics via a common simulation interface. |
| **Performance** | Full scene fidelity, RTX rendering, sensors; single-scene overhead higher than minimal engines. | With PhysX: very high throughput (e.g. tens of thousands of FPS with thousands of parallel envs) when used in Isaac Lab’s parallel pipeline. | Newton integration aims for large speedups (e.g. 152× locomotion, 313× manipulation in some benchmarks on RTX 4090) vs current PhysX-based runs. |

So: **Inference in this repo (UR10e)** runs in **Isaac Sim with PhysX**. Isaac Lab (and Arena) can use the same PhysX through Isaac Sim, but add their own env APIs and training/eval loops; Arena adds task/object/embodiment composition and large-scale evaluation.

## 2.3 Coordinate System and Conventions

| | Isaac Sim (world / PhysX) | Isaac Sim (USD / some UI) | Isaac Lab 3.0 Beta |
|--|---------------------------|----------------------------|---------------------|
| **Up axis** | **+Z** | +Y (USD convention) | Aligning with **xyzw** quaternions and common engine conventions (e.g. Newton, Warp, PhysX). |
| **Forward** | **+X** | -Z (USD) | — |
| **Handedness** | Right-handed | — | — |
| **Units** | **Meters** (default length). | Same. | Same. |
| **Quaternions** | Often **wxyz** in older Isaac Sim/Isaac Lab code. | — | Switched to **xyzw** in Isaac Lab 3.0 Beta (breaking change if you assume wxyz). |

Important for your UR10e scripts: **physics and world logic in Isaac Sim use Z-up, +X forward**. Camera prims and some USD-facing UI may show Y-up. When you spawn the table, cube, and robot (e.g. in `ur10e_pick_cube.py`), positions and goals are in **world axes (Z-up)**. Isaac Lab Arena, when using Isaac Sim under the hood, follows the same underlying world convention unless a specific wrapper changes it.

## 2.4 Inference and Deployment

| | Isaac Sim (your UR10e use case) | Isaac Lab / Isaac Lab Arena |
|--|---------------------------------|------------------------------|
| **Inference** | You load a **PyTorch checkpoint** (e.g. `ur10e_policy.pt`) in your own script, use Isaac Sim’s **articulation API** and **physics callbacks**, and run the policy in the same process as the simulator (e.g. `ur10e.forward(step_size, goal_position, object_position)`). No separate “inference server”; it’s script-driven. | Policies are typically **trained** in Isaac Lab (or elsewhere) and **evaluated** in Isaac Lab / Arena in parallel envs. Inference is still “run policy in simulation,” but orchestrated by the framework (tasks, env wrappers, logging, metrics). Arena adds standardized evaluation and diversification. |
| **Deployment** | **In-sim only** for this repo: run `ur10e_pick_cube.py` or `ur10e_standalone.py` with the kit Python. For real robots you’d deploy the same policy (e.g. via ROS or another stack) outside this doc’s scope. | Focus on **simulation-side** training and evaluation; deployment to real hardware is a separate step (e.g. export policy, use another stack). |

So: **This repo’s “inference” is “run the UR10e policy inside Isaac Sim from our scripts.”** Isaac Lab / Arena inference is “run a policy inside Isaac Lab’s envs and record metrics,” often at scale.

## 2.5 Rendering, Sensors, and Scene

| | Isaac Sim | Isaac Lab Arena |
|--|-----------|------------------|
| **Rendering** | **RTX-based**, high-quality (Omniverse/Kit). Supports headless and GUI. | Uses Isaac Sim’s rendering when running on top of it; can run headless for large-scale eval. |
| **Sensors** | Rich (cameras, lidar, IMU, contact, etc.) via Isaac Sim extensions. | Can use the same sensors through Isaac Sim; Arena focuses on task/scene/object composition rather than defining new sensor types. |
| **Scene** | **USD** stages; you add references (e.g. table, DexCube, UR10e USD), set poses, and run physics. | Same USD/Isaac Sim foundation; Arena adds **modular building blocks** (Object, Scene, Embodiment, Task) to compose many environments and tasks without hand-coding each. |

Your UR10e examples **build the scene in USD** (ground, table, cube, robot) and step the simulation; that’s the “Isaac Sim way.” Arena would instead compose such a setup from its blocks and run many variants for evaluation.

---

## 2.7 How inference in Isaac Sim is different from Isaac Lab Arena

This section states explicitly how **running a policy at inference time** in **Isaac Sim** (as in this repo) differs from doing so in **Isaac Lab Arena**.

### 2.7.1 Control flow and who drives the loop

| | Isaac Sim (this repo) | Isaac Lab Arena |
|--|------------------------|------------------|
| **Who drives** | **Your script** owns the loop. You create the stage, spawn the robot and objects, register a **physics callback**, and in that callback you call `ur10e.forward(step_size, goal_position, object_position)` every (or every Nth) physics step. The same process runs both the simulator and the policy. | The **framework** drives the loop. You define an **environment** (or use a prebuilt task); the framework steps many envs in parallel, calls your policy (or a wrapper) per env, applies actions, and advances simulation. You do not manually register physics callbacks for policy execution. |
| **Entry point** | A **standalone Python script** (e.g. `ur10e_pick_cube.py`) that you run with the Isaac Sim kit Python. Script creates `World`, adds prims, then `my_world.add_physics_callback("physics_step", on_physics_step)`; inside the callback you compute observations, call the policy, and apply actions. | A **framework entry point** (e.g. `isaaclab` or Arena CLI). You specify a task, robot, and (optionally) policy; the framework loads configs, builds envs, and runs the evaluation loop. Policy inference is invoked inside the framework’s step function. |

So: **Isaac Sim inference = “your script + physics callback + your policy class.”** **Isaac Lab Arena inference = “framework loop + env.step() + policy inside the framework.”**

### 2.7.2 Process and execution model

| | Isaac Sim (this repo) | Isaac Lab Arena |
|--|------------------------|------------------|
| **Process** | **Single process**: one Isaac Sim (Kit) process, one Python interpreter, one scene (or a few). Policy runs in the same process as the simulator. | Typically **batched**: many environments in one (or more) GPU-backed buffers; one or more policy forward passes per step across envs. Can run thousands of envs in parallel. |
| **Concurrency** | No inherent batching of envs. You can run multiple scenarios only by building them yourself (e.g. multiple robots in one stage or multiple scripts). | **Massive parallelism**: many envs stepped together; policy often evaluated in batches (e.g. obs shape `[N, obs_dim]`). |
| **Rendering** | Optional; you can show a window (GUI) or run headless. Rendering is tied to the same simulation process. | Often headless for large-scale eval; rendering can be disabled or limited to a subset of envs for efficiency. |

So: **Isaac Sim inference = single (or few) scene(s), single policy call per step, same process.** **Isaac Lab Arena inference = many envs, batched policy, optimized for throughput and metrics.**

### 2.7.3 APIs and integration

| | Isaac Sim (this repo) | Isaac Lab Arena |
|--|------------------------|------------------|
| **Simulation API** | **Isaac Sim / Omniverse APIs**: `World`, `SingleArticulation`, `get_joint_positions()`, `get_world_pose()`, `apply_action()`, `get_jacobians()`, USD stage, etc. You use these directly in your policy class and in the callback. | **Isaac Lab env API**: typically `env.step(action)`, `env.observation`, `env.reset()`. The framework wraps Isaac Sim (or another backend); you interact with the env interface, not necessarily with raw articulation APIs in your policy code. |
| **Policy loading** | **You** load the checkpoint (e.g. `torch.load("ur10e_policy.pt")`), build the actor MLP, and call `policy(obs_tensor)` in your callback. No standard “policy wrapper” required. | Policy is often provided as part of the **task or agent config**; the framework loads it and calls it each step. May support multiple policy formats (e.g. RSL-RL, other checkpoints) through adapters. |
| **Observation construction** | **You** compute the observation in your code (e.g. `_compute_observation(goal_position, object_position)`), matching the 53D layout the policy was trained on. | Observations are usually defined by the **task/env**; the framework provides the observation buffer. You may still define custom obs for custom tasks, but within the framework’s observation system. |

So: **Isaac Sim inference = direct use of Isaac Sim APIs and your own policy/observation code.** **Isaac Lab Arena inference = framework env API and framework-managed policy/observation.**

### 2.7.4 Batching and performance

| | Isaac Sim (this repo) | Isaac Lab Arena |
|--|------------------------|------------------|
| **Batch size** | Effectively **1** (one scene, one policy forward per step or per decimation). | **N** envs: policy often receives `[N, obs_dim]` and returns `[N, action_dim]`; simulation steps N envs in one go. |
| **Throughput** | Suited for **single-scene** demos, debugging, or deployment-style testing. Not optimized for “millions of steps per second” across envs. | Optimized for **throughput**: many envs, vectorized physics, batched policy, to maximize samples per second for evaluation or training. |
| **Use case** | “Run this one policy in this one scene and watch it (or log it).” | “Evaluate this policy across many tasks/robots/objects and aggregate success rates or metrics.” |

So: **Isaac Sim inference = one (or few) env(s), human-in-the-loop or single-scene eval.** **Isaac Lab Arena inference = large-scale, batched, automated evaluation.**

### 2.7.5 Logging, metrics, and evaluation

| | Isaac Sim (this repo) | Isaac Lab Arena |
|--|------------------------|------------------|
| **Logging** | **You** add any print/logging or file writes (e.g. cube position, joint positions). No built-in “evaluation report” for the policy. | Framework typically **aggregates** rewards, success flags, or custom metrics across envs and steps; can produce reports, logs, or leaderboard-ready outputs. |
| **Success / failure** | You can define and log your own (e.g. distance to goal, object height). Not standardized. | **Standardized** task termination and success criteria; Arena is built to compare policies and report metrics consistently. |
| **Reproducibility** | You control seeds and script; no standard “run id” or experiment tracking unless you add it. | Often integrated with **experiment tracking** (e.g. run ids, configs, seeds) for reproducible benchmarks. |

So: **Isaac Sim inference = ad-hoc logging and your own metrics.** **Isaac Lab Arena inference = structured evaluation and standardized metrics.**

### 2.7.6 Summary: inference in one sentence each

- **Isaac Sim (this repo):** Inference is **your script** calling **your policy** inside a **physics callback** in a **single Isaac Sim process**, using **Isaac Sim APIs** directly, with **no batching** and **no framework-managed evaluation**.
- **Isaac Lab Arena:** Inference is the **framework** stepping **many envs**, calling your policy in **batches**, using **env APIs** and **framework-managed** observations and metrics, for **large-scale, standardized policy evaluation**.

---

## 2.6 Summary Table (Quick Reference)

| Aspect | Isaac Sim | Isaac Lab / Isaac Lab Arena |
|--------|-----------|-----------------------------|
| **Role** | Core simulator (physics, rendering, USD, ROS, scripts). | Robot learning + policy evaluation framework (on top of Isaac Sim). |
| **Physics** | PhysX (default). | PhysX; Isaac Lab 3.0 Beta adding Newton (and pluggable backends). |
| **Coordinates** | World: Z-up, +X forward, right-handed, meters. | Same base; Isaac Lab 3.0 Beta: xyzw quaternions. |
| **Inference (this repo)** | Run policy in Python with kit (e.g. UR10e `.pt` in `ur10e_pick_cube.py`). | Run policies in framework-managed envs; Arena adds standardized eval. |
| **Use case in this repo** | UR10e reach/lift policy inference in a single (or few) scene(s). | Not used in this repo; would be used for training and large-scale policy evaluation. |

---

# Part 3: Where to Go From Here

- **Run the UR10e use case:**  
  **docs/RUN_UR10E_USECASE.md**  
- **Push/share changes:**  
  **docs/PUSH_UR10E_CHANGES.md**  
- **What was migrated from standalone:**  
  **docs/MIGRATION_FROM_STANDALONE.md**  
- **Isaac Sim conventions (axes, units):**  
  [Isaac Sim Conventions](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html)  
- **Isaac Lab:**  
  [NVIDIA Isaac Lab](https://developer.nvidia.com/isaac/lab)  
- **Isaac Lab Arena:**  
  [NVIDIA Isaac Lab-Arena](https://developer.nvidia.com/isaac/lab-arena), [IsaacLab-Arena (GitHub)](https://github.com/isaac-sim/IsaacLab-Arena)

This README and the linked docs give a full picture of what we did and how Isaac Sim differs from Isaac Lab Arena for physics, coordinates, inference, and usage.
