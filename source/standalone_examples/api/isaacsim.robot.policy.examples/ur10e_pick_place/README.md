# Deploying an IsaacLab-Arena RL Policy in Raw IsaacSim

## A practical guide to sim-to-sim policy transfer for UR10e pick-and-place

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Repository Map](#2-repository-map)
3. [The Three Layers](#3-the-three-layers-isaacsim--isaaclab--isaaclab-arena)
4. [What We Trained](#4-what-we-trained)
5. [The Core Challenge](#5-the-core-challenge-why-not-just-load-and-run)
6. [Architecture of the Deployment Script](#6-architecture-of-the-deployment-script)
7. [Step-by-Step: What We Built](#7-step-by-step-what-we-built)
8. [Bugs, Crashes, and How We Fixed Them](#8-bugs-crashes-and-how-we-fixed-them)
9. [The Sim-to-Sim Transfer Gap](#9-the-sim-to-sim-transfer-gap-deep-dive)
10. [Current Status and Remaining Gaps](#10-current-status-and-remaining-gaps)
11. [How to Run](#11-how-to-run)
12. [FAQ](#12-faq)

---

## 1. The Big Picture

We have a trained RL policy for a **UR10e robot performing pick-and-place** with a Robotiq 2F-140 gripper. The policy was trained inside **IsaacLab-Arena** (a high-level RL framework). Our goal is to deploy and run this same policy in **raw IsaacSim** (the low-level simulation platform), without any IsaacLab dependency at runtime.

**Why?** Because the ultimate target is real hardware. IsaacLab-Arena is great for training, but the deployment path goes through IsaacSim (and eventually ROS/real robot). We need to prove the policy works in the raw simulator first.

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE DEPLOYMENT PATH                         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  IsaacLab-   │    │   Raw        │    │   Real       │       │
│  │  Arena       │───>│   IsaacSim   │───>│   Hardware   │       │
│  │  (training)  │    │  (this work) │    │  (future)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│        RL policy          Same policy         Same policy       │
│        trains here        runs here           runs here         │
│                                                                 │
│  High-level APIs     Low-level APIs       Physical robot        │
│  Env managers        Direct PhysX         ROS2 / drivers        │
│  Auto observations   Manual obs build     Sensor fusion         │
│  Auto actions        Manual IK + grip     Motor commands        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Map

```
~/workspace/stbot/
├── IsaacSim/                          # NVIDIA Isaac Sim 5.1 (core simulator)
│   └── source/standalone_examples/
│       └── api/isaacsim.robot.policy.examples/
│           └── ur10e_pick_place/       # <── OUR WORK LIVES HERE
│               ├── run_policy.py       # Main deployment script
│               ├── differential_ik.py  # Self-contained DiffIK solver
│               ├── policy/
│               │   ├── policy.pt       # Trained TorchScript model
│               │   └── env.yaml        # Training configuration (reference)
│               └── README.md           # This file
│
├── IsaacLab/                          # IsaacLab v2.3.1 (RL framework)
│   └── source/isaaclab/               # Controllers, math utils, sensors
│
├── IsaacLab-Arena/                    # Arena (composable RL environments)
│   └── release/0.1.1                  # Branch with training environment
│
└── stackbot_rl_policy/                # Exported trained policy artifacts
    ├── policy.pt                      # TorchScript model (53→7 MLP)
    ├── env.yaml                       # Full training env configuration
    └── agent.yaml                     # PPO agent configuration
```

---

## 3. The Three Layers: IsaacSim → IsaacLab → IsaacLab-Arena

Understanding the stack is essential to understanding the deployment challenge.

```
┌───────────────────────────────────────────────────────────┐
│                   IsaacLab-Arena                           │
│  Composable environment builder for RL research           │
│  - Task definitions (pick-and-place, stacking, etc.)      │
│  - Embodiment configs (UR10e + Robotiq, Franka, etc.)     │
│  - Custom observations, rewards, terminations             │
│  - One-line training: "train policy for Task X on Robot Y"│
├───────────────────────────────────────────────────────────┤
│                     IsaacLab                               │
│  GPU-accelerated robotics research framework              │
│  - Environment managers (obs, actions, rewards, events)   │
│  - Controllers (DiffIK, OSC, joint-space)                 │
│  - Sensors (FrameTransformer, camera, contact)            │
│  - Math utilities (quaternion ops, frame transforms)      │
│  - Articulation wrappers with actuator models             │
├───────────────────────────────────────────────────────────┤
│                      IsaacSim                              │
│  NVIDIA's core simulation platform (Omniverse/Kit)        │
│  - PhysX 5 rigid-body and articulation physics            │
│  - USD scene representation                               │
│  - RTX rendering                                          │
│  - Low-level Python API (prims, articulations, worlds)    │
└───────────────────────────────────────────────────────────┘
```

**Key insight**: IsaacLab and Arena add ~10 layers of abstraction on top of IsaacSim. When you call `env.step(action)` in Arena, dozens of things happen automatically (observation computation, IK solving, action decimation, joint drive application). In raw IsaacSim, we must do **all of this manually**.

---

## 4. What We Trained

**Task**: Pick up a DexCube from a table and move it to a goal position.

**Robot**: UR10e (6-DOF arm) + Robotiq 2F-140 (parallel gripper) — 14 total DOFs.

**Policy architecture** (from `agent.yaml`):
```
Input (53D) → Normalizer → MLP [256, 128, 64] (ELU) → Output (7D)
                                                        ├── 6D: arm DiffIK command
                                                        └── 1D: gripper command
```

**Observation vector** (53 dimensions):
```
┌─────────────────────────────────────────────────────────────────┐
│  Observation (53D)                                              │
├──────────────────────┬──────┬───────────────────────────────────┤
│  Component           │ Dims │ Description                       │
├──────────────────────┼──────┼───────────────────────────────────┤
│  last_action         │   7  │ Previous policy output            │
│  joint_pos_rel       │  14  │ All joint pos − default pos       │
│  joint_vel_rel       │  14  │ All joint vel − default vel       │
│  eef_pos             │   3  │ End-effector world position       │
│  eef_quat            │   4  │ End-effector orientation (wxyz)   │
│  gripper_pos         │   1  │ finger_joint position             │
│  target_pose         │   7  │ Goal position + orientation       │
│  object_pos_rel      │   3  │ Cube position relative to robot   │
├──────────────────────┼──────┼───────────────────────────────────┤
│  Total               │  53  │                                   │
└──────────────────────┴──────┴───────────────────────────────────┘
```

**Action vector** (7 dimensions):
```
action[0:6] → 6D relative pose delta (dx, dy, dz, droll, dpitch, dyaw)
              Scaled by 0.1, then applied via Differential IK → joint targets
action[6]   → Gripper command (>0 = attempt close if near cube)
```

**Training distribution**:
```
Goal:  X ∈ [0.40, 0.60]  Y ∈ [−0.25, 0.25]  Z ∈ [0.255, 0.455]
Cube:  Fixed at (0.5, 0.0, 0.055) — no randomization
Robot: Default pose + Gaussian noise (σ = 0.02 rad)
```

---

## 5. The Core Challenge: Why Not Just "Load and Run"?

**Q: The policy is a TorchScript `.pt` file. Can't we just load it and call `policy(obs)`?**

A: Loading the model is trivial. The hard part is everything around it:

```
┌─────────────────────────────────────────────────────────────────┐
│            What Arena does automatically (hidden)                │
│                                                                 │
│  1. Build scene with correct physics parameters                 │
│  2. Configure articulation (solver iters, drives, gravity)      │
│  3. Read 53D observation from 7 different sources               │
│  4. Apply action through DiffIK controller                      │
│  5. Handle decimation (4 physics steps per policy step)         │
│  6. Re-solve IK every physics step within decimation            │
│  7. Apply proximity gripper logic                               │
│  8. Manage episode resets and terminations                      │
│                                                                 │
│            What raw IsaacSim gives you                           │
│                                                                 │
│  - world.step()                                                 │
│  - robot.get_joint_positions()                                  │
│  - robot.apply_action(joint_targets)                            │
│  - That's about it.                                             │
└─────────────────────────────────────────────────────────────────┘
```

We had to **reverse-engineer and reimplement** every abstraction layer that Arena/IsaacLab provides, using only the raw IsaacSim API.

---

## 6. Architecture of the Deployment Script

```
run_policy.py
│
├── Scene Setup ──────────── setup_scene()
│   ├── Ground plane, table, UR10e, DexCube, lights
│   └── Matches training USD assets and positions
│
├── Physics Config ───────── configure_physics()
│   ├── Articulation solver: 16 pos / 1 vel iterations
│   ├── Disable gravity on robot bodies
│   ├── Set max depenetration velocity
│   └── Apply joint drive stiffness/damping (from env.yaml)
│
├── Inference Manager ────── UR10ePickPlaceInference
│   ├── __init__(): Load policy, create DiffIK solver
│   ├── initialize(): Map joints, read defaults, setup Jacobian
│   ├── _get_ee_pose(): FK via wrist_3_link + offset
│   ├── _get_jacobian(): PhysX Jacobian + body-offset correction
│   ├── _compute_observation(): Build 53D vector
│   ├── step(): Full inference cycle
│   │   ├── Read observation
│   │   ├── Policy forward pass
│   │   ├── DiffIK: command → desired EE → joint targets
│   │   ├── Proximity gripper logic
│   │   └── Apply joint targets
│   └── check_ood_status(): Detect out-of-distribution states
│
├── DiffIK Solver ────────── differential_ik.py (standalone module)
│   ├── Quaternion math (conjugate, multiply, axis-angle)
│   ├── Pose error computation
│   ├── Damped Least Squares IK
│   └── No IsaacLab dependency
│
└── Main Loop ────────────── main()
    ├── Validate & clamp goal/cube positions
    ├── Scene warmup (50 physics steps)
    ├── Decimated inference loop
    │   ├── Policy runs once
    │   └── Same action held for 4 physics steps
    └── Diagnostic logging (--verbose, CSV output)
```

---

## 7. Step-by-Step: What We Built

### Step 1: Understanding the Policy

We started by examining the exported policy artifacts:
- `policy.pt` — a TorchScript MLP (53 → 256 → 128 → 64 → 7)
- `env.yaml` — 3600+ lines of training configuration
- `agent.yaml` — PPO hyperparameters

From `env.yaml`, we extracted every detail needed to reproduce the environment: physics timestep, robot configuration, observation terms, action pipeline, and training distribution bounds.

### Step 2: Self-Contained Differential IK

The policy outputs 6D relative pose deltas, not joint positions. These must be converted to joint targets via Differential Inverse Kinematics. Rather than importing all of IsaacLab at runtime, we extracted the core math into `differential_ik.py`:

```
Policy output (6D)
    │
    ▼
┌──────────────────────────┐
│  Scale by 0.1            │  action_scale from env.yaml
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Apply as relative delta │  current_EE_pose + scaled_delta
│  to current EE pose      │  → desired_EE_pose
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Compute pose error      │  desired − current (pos + axis-angle)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Damped Least Squares    │  Δq = Jᵀ(JJᵀ + λ²I)⁻¹ · error
│  (λ = 0.01)              │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  q_target = q_current+Δq │  New joint position targets
└──────────────────────────┘
```

### Step 3: Observation Pipeline

We manually reconstructed the 53D observation vector by reading 7 different data sources from IsaacSim:

```python
obs = concatenate([
    prev_action,                    # 7D  — stored from last step
    joint_pos - default_joint_pos,  # 14D — from robot.get_joint_positions()
    joint_vel - default_joint_vel,  # 14D — from robot.get_joint_velocities()
    ee_pos,                         # 3D  — FK: wrist_3_link + offset
    ee_quat,                        # 4D  — FK: wrist_3_link + offset
    finger_joint_pos,               # 1D  — from joint positions
    goal_pos + goal_quat,           # 7D  — user-provided goal
    cube_pos - robot_pos,           # 3D  — from rigid body pose
])
```

### Step 4: Proximity Gripper

Arena uses a custom `ProximityGripperAction` — the gripper only closes when the end-effector is within `close_threshold` (1.5cm) of the cube AND the policy commands a positive gripper value. We replicated this logic:

```python
if ee_to_cube_distance < 0.015 and gripper_command > 0:
    finger_target = 0.7   # close
else:
    finger_target = 0.0   # open
```

### Step 5: Training Distribution Safety

RL policies behave unpredictably outside their training distribution. We added validation and clamping for user-provided goal and cube positions:

```
Goal X: [0.40, 0.60]  — clamped by default
Goal Y: [−0.25, 0.25] — clamped by default
Goal Z: [0.255, 0.455] — clamped by default
Cube:   fixed (0.5, 0.0, 0.055) during training — warn on deviation
```

---

## 8. Bugs, Crashes, and How We Fixed Them

### Crash 1: `XFormPrim.__init__() got an unexpected keyword argument 'prim_path'`

**Root cause**: IsaacSim 5.1 renamed `XFormPrim` to a "multi-prim view" class. The single-prim version is now `SingleXFormPrim`.

**Fix**: Changed all prim instantiations:
```python
# Before (crashed)
XFormPrim(prim_path="/World/Table", ...)
RigidPrim(prim_path="/World/DexCube", ...)

# After (works)
SingleXFormPrim(prim_path="/World/Table", ...)
SingleRigidPrim(prim_path="/World/DexCube", ...)
```

### Crash 2: `'Articulation' object has no attribute '_articulation'`

**Root cause**: The experimental `Articulation` API in IsaacSim 5.1 does not expose internal `_articulation` attributes. We were trying to access link paths through a private member.

**Fix**: Used the public `get_link_indices("wrist_3_link")` method instead:
```python
# Before (crashed)
self._jac_robot._articulation.link_paths[0]

# After (works)
self._jac_robot.get_link_indices("wrist_3_link").list()[0]
```

### Crash 3: `mat1 and mat2 shapes cannot be multiplied (1x54 and 53x256)`

**Root cause**: The observation vector was 54D instead of 53D. The `gripper_pos` observation was producing 2 values (mimicking a Franka 2-finger gripper) instead of 1 (the UR10e custom single-finger position).

**Fix**: Changed from both finger joints to just the single `finger_joint`:
```python
# Before (54D — wrong)
gripper_obs = joint_pos[[finger_idx, other_finger_idx]]  # 2D

# After (53D — correct)
gripper_obs = np.array([joint_pos[finger_joint_idx]])     # 1D
```

### Behavioral Bug: Robot pushes cube instead of grasping

**Root cause**: Multiple sim-to-sim transfer gaps (see Section 9).

---

## 9. The Sim-to-Sim Transfer Gap (Deep Dive)

**Q: Both Arena and raw IsaacSim use PhysX. Why would behavior differ?**

A: The physics engine is identical. The gaps come from the **software layers between the policy and PhysX**:

```
                    IsaacLab-Arena                    Raw IsaacSim
                    ──────────────                    ────────────
Policy output       policy(obs) → 7D action           SAME
     │                   │                              │
     ▼                   ▼                              ▼
Action scaling      × 0.1 (automatic)                 × 0.1 (manual) ✅
     │                   │                              │
     ▼                   ▼                              ▼
DiffIK set_command  Once per policy step               SAME ✅
     │                   │                              │
     ▼                   ▼                              ▼
╔════╧═══════════════════╧══════════════════════════════╧════════╗
║              DECIMATION LOOP (4 physics steps)                 ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Arena:                        Raw IsaacSim (current):         ║
║  ┌─────────────────────┐       ┌─────────────────────┐         ║
║  │ Re-read EE pose     │       │                     │         ║
║  │ Re-read joint pos   │       │ Hold same joint     │         ║
║  │ Re-read Jacobian    │       │ targets for all     │         ║
║  │ Re-solve IK         │──vs──>│ 4 physics steps     │         ║
║  │ Set NEW joint target│       │                     │         ║
║  │ Physics step        │       │ Physics step ×4     │         ║
║  │ (repeat ×4)         │       │                     │         ║
║  └─────────────────────┘       └─────────────────────┘         ║
║                                                                ║
║  = Closed-loop IK servo        = Open-loop target hold         ║
╚════════════════════════════════════════════════════════════════╝
     │                   │                              │
     ▼                   ▼                              ▼
Read observation    After 4 steps                      SAME ✅
     │                   │                              │
     ▼                   ▼                              ▼
Policy input        53D observation                    SAME ✅
```

### Gap 1 (FIXED): Decimation not applied

**Problem**: Our script called the policy every physics step (every 0.005s). Arena calls it every 4 physics steps (every 0.02s).

**Impact**: The policy saw observations that barely changed between calls. The IK deltas were 1/4 the expected magnitude. The robot moved 4× slower than intended.

**Fix**: Restructured the main loop to run 4 physics steps per policy call.

### Gap 2 (FIXED): Jacobian body-offset correction missing

**Problem**: PhysX returns the Jacobian for `wrist_3_link`, but the end-effector is 24cm above it. IsaacLab applies a skew-symmetric correction:

```
J_translational += −skew([0, 0, 0.24]) × J_rotational
```

Without this, the IK solver optimizes for the wrong point in space.

**Fix**: Added the skew-symmetric correction to `_get_jacobian()`.

### Gap 3 (FIXED): Joint drives never applied

**Problem**: `JOINT_DRIVE_CONFIG` was defined but never written to the USD. The robot's joint response didn't match training.

**Fix**: Iterate through all joints in `configure_physics()` and set `UsdPhysics.DriveAPI` stiffness/damping.

### Gap 4 (FIXED): No scene warmup

**Problem**: The DexCube spawned at z=0.055 but immediately sank to z=0.027 (collision geometry settling). The policy saw a cube position that differed from training.

**Fix**: Added 50 warmup physics steps before starting inference, with a warning if the cube drops >1cm.

### Gap 5 (FIXED): Gripper joint defaults assumed zero

**Problem**: `joint_pos_rel` was computed using hand-coded defaults that assumed all non-arm joints default to 0. If the Robotiq USD has different defaults, the observation would be slightly wrong.

**Fix**: Read actual defaults from the articulation after initialization, then override only the 6 arm joints.

### Gap 6 (KNOWN — not yet fixed): IK servo within decimation

**Problem**: Arena re-solves the IK every physics step (closed-loop servo). We solve once and hold (open-loop).

**Impact**: Less accurate EE tracking per policy step.

### Gap 7 (KNOWN — minor): Velocity limits not set

**Problem**: Arena sets `velocity_limit_sim` per joint group. We don't enforce these.

---

## 10. Current Status and Remaining Gaps

### Parity Checklist

| Component | Arena | Our Script | Status |
|-----------|-------|-----------|--------|
| Physics engine | PhysX 5 | PhysX 5 | ✅ Match |
| Physics dt | 0.005s | 0.005s | ✅ Match |
| Decimation | 4 | 4 | ✅ Fixed |
| Joint drives (stiffness/damping) | From actuator config | Via DriveAPI | ✅ Fixed |
| Gravity disabled on robot | Yes | Yes | ✅ Match |
| Solver iterations (16/1) | Yes | Yes | ✅ Match |
| EE body offset (0,0,0.24) | Yes | Yes | ✅ Match |
| Jacobian body-offset correction | skew-symmetric | skew-symmetric | ✅ Fixed |
| Obs: last_action (7D) | Auto | Manual | ✅ Match |
| Obs: joint_pos_rel (14D) | Auto | Manual | ✅ Fixed |
| Obs: joint_vel_rel (14D) | Auto | Manual | ✅ Match |
| Obs: eef_pos (3D) | FrameTransformer | FK + offset | ✅ Match |
| Obs: eef_quat (4D) | FrameTransformer | FK + offset | ✅ Match |
| Obs: gripper_pos (1D) | Custom func | finger_joint | ✅ Fixed |
| Obs: target_pose (7D) | Command manager | Manual | ✅ Match |
| Obs: object_pos_rel (3D) | Custom func | Manual | ✅ Match |
| DiffIK solver (DLS, λ=0.01) | Built-in | Standalone | ✅ Match |
| Action scale (0.1) | Config | Constant | ✅ Match |
| Proximity gripper (1.5cm) | Custom class | Manual logic | ✅ Match |
| Scene warmup | Implicit | 50 steps | ✅ Fixed |
| **IK servo within decimation** | **4 re-solves/step** | **1 solve/step** | ⚠️ Gap |
| **Velocity limits** | **Per-joint** | **Not set** | ⚠️ Minor |
| **Gripper sub-joint drives** | **All configured** | **Only 7 joints** | ⚠️ Minor |

---

## 11. How to Run

### Prerequisites

- IsaacSim 5.1 built from source at `~/workspace/stbot/IsaacSim`
- Policy file at `ur10e_pick_place/policy/policy.pt`

### Basic run (default goal and cube positions)

```bash
cd ~/workspace/stbot/IsaacSim/_build/linux-x86_64/release
./python.sh ../../../source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_place/run_policy.py
```

### Custom goal position

```bash
./python.sh .../run_policy.py --goal_x 0.55 --goal_y 0.1 --goal_z 0.3
```

### Verbose diagnostics with CSV output

```bash
./python.sh .../run_policy.py --verbose --log_interval 5
# Creates diagnostics_<timestamp>.csv alongside the script
```

### Headless mode (no GUI)

```bash
./python.sh .../run_policy.py --headless --num_steps 2000
```

### All options

```
--headless          Run without GUI
--num_steps N       Max policy steps (default: 10000)
--goal_x/y/z F     Goal position (clamped to training range)
--cube_x/y/z F     Cube spawn position
--no_clamp          Disable safety clamping (OOD risk)
--verbose           Enable per-step diagnostic logging
--log_interval N    Steps between verbose log lines (default: 10)
```

---

## 12. FAQ

### Q: Why not just run inference in IsaacLab-Arena directly?

We can, and we did — it works great there. But the deployment path to real hardware doesn't go through Arena. We need a standalone IsaacSim script that we can later replace the simulator with a real robot driver. This work proves the policy is portable.

### Q: Why reimplement the DiffIK solver instead of importing IsaacLab?

IsaacLab is a heavy dependency (~100+ extensions). For deployment, we want a minimal, self-contained script. The DiffIK solver is ~135 lines of pure PyTorch math. No external dependencies.

### Q: Why does the cube sink from z=0.055 to z=0.027?

The DexCube's collision geometry doesn't perfectly sit on the table surface at the configured spawn height. After physics settles (our warmup step), it drops ~3cm. This is a USD asset geometry detail, not a bug in our code. The warmup step detects and reports this.

### Q: What is "decimation" and why does it matter so much?

Decimation = the ratio of physics steps to policy steps. With `decimation=4`, the policy runs at 50 Hz (0.02s) while physics runs at 200 Hz (0.005s). The policy was trained expecting its actions to be applied for 4 physics steps. Without decimation, the robot saw the policy running 4× too fast and each action having 1/4 the expected effect.

### Q: What is the Jacobian body-offset correction?

The PhysX Jacobian maps joint velocities to the velocity of `wrist_3_link`. But our end-effector frame is 24cm above that link. Due to the lever arm, a rotation at `wrist_3_link` causes a translation at the EE point. The skew-symmetric correction accounts for this:

```
J_trans_EE = J_trans_link − skew(offset) × J_rot_link
```

Without it, the IK solver computes joint motions that move the wrong point in space.

### Q: What is "proximity gripper" and why can't we use a normal gripper?

The training environment uses a custom gripper action: the gripper only actuates when the EE is physically close to the cube (within 1.5cm). This prevents the policy from closing the gripper in mid-air (which would waste time and confuse the grasp). We replicate this distance-based gating in raw IsaacSim.

### Q: Why did we remove episode resets?

Initially we added automatic resets (teleport robot and cube back) when the cube fell off the table. But on real hardware, you can't teleport objects. Removing resets makes the simulation match real-world deployment: if the policy fails, it must deal with the consequences. We kept OOD detection as warning logs.

### Q: What's the "IK servo within decimation" gap?

In Arena, the IK solver recomputes joint targets every physics sub-step (4× per policy step), reading the latest robot state each time. This creates a closed-loop servo that converges accurately. In our script, we compute targets once and hold them. The PD controller does the rest, but without IK feedback, tracking is less precise.

---

## Appendix: Key Constants from Training (env.yaml)

```python
PHYSICS_DT          = 0.005    # 200 Hz physics
DECIMATION          = 4        # Policy at 50 Hz
ACTION_SCALE        = 0.1      # Scale before IK
DLS_LAMBDA          = 0.01     # Damped least squares regularization
GRIPPER_THRESHOLD   = 0.015    # 1.5cm proximity trigger
GRIPPER_OPEN        = 0.0      # finger_joint open position
GRIPPER_CLOSE       = 0.7      # finger_joint close position
EE_BODY_OFFSET      = [0, 0, 0.24]  # From wrist_3_link to EE
SOLVER_POS_ITERS    = 16       # PhysX position iterations
SOLVER_VEL_ITERS    = 1        # PhysX velocity iterations
```
