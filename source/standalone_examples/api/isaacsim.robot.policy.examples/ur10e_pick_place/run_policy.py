# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone IsaacSim script to run a trained RL policy for UR10e pick-and-place.
# The policy was trained in IsaacLab-Arena and is deployed here using IsaacSim APIs
# with a self-contained Differential IK solver.
#
# TRAINING DISTRIBUTION (from env.yaml):
#   Goal X:  [0.40, 0.60]   Goal Y: [-0.25, 0.25]   Goal Z: [0.255, 0.455]
#   Cube:    fixed at (0.5, 0.0, 0.055) -- no randomization during training
#   Robot:   fixed default pose + tiny gaussian noise (std=0.02 rad)
#
# Usage:
#   cd IsaacSim/_build/linux-x86_64/release
#   ./python.sh <path>/run_policy.py [--goal_x 0.5] [--goal_y 0.1] [--goal_z 0.35]
#   ./python.sh <path>/run_policy.py --cube_x 0.5 --cube_y 0.05 --cube_z 0.055
#   ./python.sh <path>/run_policy.py --no_clamp   # disable safety clamping

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

parser = argparse.ArgumentParser(
    description="Run UR10e pick-and-place RL policy in IsaacSim",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Training-safe ranges (from env.yaml):
  Goal position:  X=[0.40, 0.60]  Y=[-0.25, 0.25]  Z=[0.255, 0.455]
  Cube position:  X=0.50  Y=0.00  Z=0.055  (fixed during training)

Values outside these ranges will be clamped unless --no_clamp is passed.
""",
)
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num_steps", type=int, default=10000, help="Max simulation steps")
parser.add_argument("--goal_x", type=float, default=0.5, help="Goal X [safe: 0.40-0.60]")
parser.add_argument("--goal_y", type=float, default=0.0, help="Goal Y [safe: -0.25-0.25]")
parser.add_argument("--goal_z", type=float, default=0.35, help="Goal Z [safe: 0.255-0.455]")
parser.add_argument("--cube_x", type=float, default=0.5, help="Cube spawn X [trained: 0.5]")
parser.add_argument("--cube_y", type=float, default=0.0, help="Cube spawn Y [trained: 0.0]")
parser.add_argument("--cube_z", type=float, default=0.055, help="Cube spawn Z [trained: 0.055]")
parser.add_argument("--no_clamp", action="store_true", help="Disable clamping to training distribution (use at own risk)")
parser.add_argument("--verbose", action="store_true", help="Enable per-step diagnostic logging")
parser.add_argument("--log_interval", type=int, default=10, help="Steps between verbose log lines (default: 10)")
parser.add_argument("--episode_length", type=int, default=600, help="Per-episode timeout in steps (default: 600)")
args = parser.parse_args()

# --- Launch IsaacSim ---
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# All Omniverse imports must come after SimulationApp is created
import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.transformations import get_world_pose_from_relative
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics, PhysxSchema, Gf

# Local DiffIK solver (no IsaacLab dependency)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from differential_ik import DifferentialIKSolver

# ============================================================================
# Constants from env.yaml (training configuration)
# ============================================================================
PHYSICS_DT = 0.005
DECIMATION = 4
ACTION_SCALE = 0.1
DLS_LAMBDA = 0.01
GRIPPER_CLOSE_THRESHOLD = 0.015
GRIPPER_OPEN_POS = 0.0
GRIPPER_CLOSE_POS = 0.7

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Default arm pose from init_ur10e_arm_pose event in env.yaml
DEFAULT_ARM_POSE = [0.0, -2.3562, 1.5708, -0.7854, -1.5708, 0.0]

# EE body offset from wrist_3_link (env.yaml: body_offset)
EE_BODY_OFFSET_POS = np.array([0.0, 0.0, 0.24])
EE_BODY_OFFSET_ROT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity

JOINT_DRIVE_CONFIG = {
    "shoulder_pan_joint": {"stiffness": 2000.0, "damping": 100.0},
    "shoulder_lift_joint": {"stiffness": 2000.0, "damping": 100.0},
    "elbow_joint": {"stiffness": 1000.0, "damping": 60.0},
    "wrist_1_joint": {"stiffness": 500.0, "damping": 40.0},
    "wrist_2_joint": {"stiffness": 500.0, "damping": 40.0},
    "wrist_3_joint": {"stiffness": 500.0, "damping": 40.0},
    "finger_joint": {"stiffness": 2000.0, "damping": 100.0},
}

# ============================================================================
# Training Distribution Bounds (from env.yaml command ranges / initial state)
# ============================================================================
GOAL_X_RANGE = (0.40, 0.60)
GOAL_Y_RANGE = (-0.25, 0.25)
GOAL_Z_RANGE = (0.255, 0.455)

CUBE_TRAINED_POS = (0.5, 0.0, 0.055)
CUBE_X_RANGE = (0.35, 0.65)
CUBE_Y_RANGE = (-0.20, 0.20)
CUBE_Z_RANGE = (0.055, 0.055)

# Episode reset thresholds
CUBE_FALL_Z = -0.05
CUBE_GOAL_MAX_DIST = 1.0
WARMUP_STEPS = 50


def validate_and_clamp(value: float, lo: float, hi: float, name: str, clamp: bool) -> float:
    """Check *value* against [lo, hi]. Clamp or warn depending on *clamp*."""
    if lo <= value <= hi:
        return value
    if clamp:
        clamped = max(lo, min(hi, value))
        print(f"[WARN] {name}={value:.4f} outside training range [{lo}, {hi}], clamped to {clamped:.4f}")
        return clamped
    print(f"[WARN] {name}={value:.4f} outside training range [{lo}, {hi}] -- OOD behavior possible!")
    return value


def validate_goal(x: float, y: float, z: float, clamp: bool) -> tuple[float, float, float]:
    x = validate_and_clamp(x, *GOAL_X_RANGE, "goal_x", clamp)
    y = validate_and_clamp(y, *GOAL_Y_RANGE, "goal_y", clamp)
    z = validate_and_clamp(z, *GOAL_Z_RANGE, "goal_z", clamp)
    return x, y, z


def validate_cube(x: float, y: float, z: float, clamp: bool) -> tuple[float, float, float]:
    x = validate_and_clamp(x, *CUBE_X_RANGE, "cube_x", clamp)
    y = validate_and_clamp(y, *CUBE_Y_RANGE, "cube_y", clamp)
    z = validate_and_clamp(z, *CUBE_Z_RANGE, "cube_z", clamp)
    return x, y, z


# Isaac Sim 5.1 asset URLs
ASSETS_BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"
UR10E_USD = f"{ASSETS_BASE}/Robots/UniversalRobots/ur10e/ur10e.usd"
TABLE_USD = f"{ASSETS_BASE}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
DEXCUBE_USD = f"{ASSETS_BASE}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
GROUND_USD = f"{ASSETS_BASE}/Environments/Grid/default_environment.usd"


# ============================================================================
# Scene Setup
# ============================================================================
def setup_scene(cube_pos: np.ndarray):
    """Spawn all scene elements matching the training environment.

    Args:
        cube_pos: (3,) world position for the DexCube.
    """

    # Ground plane at z=-1.05
    add_reference_to_stage(usd_path=GROUND_USD, prim_path="/World/GroundPlane")
    ground = SingleXFormPrim(prim_path="/World/GroundPlane", name="ground_plane")
    ground.set_world_pose(position=np.array([0.0, 0.0, -1.05]))

    # Seattle Lab Table at (0.5, 0, 0) with 90-deg rotation
    add_reference_to_stage(usd_path=TABLE_USD, prim_path="/World/Table")
    table = SingleXFormPrim(prim_path="/World/Table", name="table")
    table.set_world_pose(
        position=np.array([0.5, 0.0, 0.0]),
        orientation=np.array([0.707, 0.0, 0.0, 0.707]),
    )

    # UR10e robot with Robotiq 2F-140 gripper at origin
    add_reference_to_stage(usd_path=UR10E_USD, prim_path="/World/Robot")
    robot_prim = get_prim_at_path("/World/Robot")
    vsets = robot_prim.GetVariantSets()
    if vsets.HasVariantSet("Gripper"):
        vsets.GetVariantSet("Gripper").SetVariantSelection("Robotiq_2f_140")

    # DexCube on the table
    add_reference_to_stage(usd_path=DEXCUBE_USD, prim_path="/World/DexCube")
    cube = SingleXFormPrim(prim_path="/World/DexCube", name="dex_cube")
    cube.set_world_pose(position=cube_pos)
    print(f"[INFO] Cube spawned at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")

    # Dome light
    light_prim = define_prim("/World/DomeLight", "DomeLight")
    light_prim.GetAttribute("inputs:intensity").Set(3000.0)
    light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(0.75, 0.75, 0.75))


def configure_physics():
    """Set articulation solver properties and joint drives to match training config."""
    stage = simulation_app.context.get_stage()
    robot_prim = get_prim_at_path("/World/Robot")

    # Articulation root: solver iterations, self-collisions
    if robot_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        artic_api = PhysxSchema.PhysxArticulationAPI(robot_prim)
    else:
        artic_api = PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
    artic_api.GetSolverPositionIterationCountAttr().Set(16)
    artic_api.GetSolverVelocityIterationCountAttr().Set(1)
    artic_api.GetEnabledSelfCollisionsAttr().Set(False)

    # Disable gravity on robot rigid bodies and set max depenetration velocity
    for prim in stage.Traverse():
        prim_path_str = str(prim.GetPath())
        if not prim_path_str.startswith("/World/Robot"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim) if not prim.HasAPI(
                PhysxSchema.PhysxRigidBodyAPI
            ) else PhysxSchema.PhysxRigidBodyAPI(prim)
            physx_rb.GetDisableGravityAttr().Set(True)
            physx_rb.GetMaxDepenetrationVelocityAttr().Set(5.0)

    # Apply joint drive stiffness/damping from training config
    for joint_name, drive_cfg in JOINT_DRIVE_CONFIG.items():
        joint_prim = None
        for prim in stage.Traverse():
            if prim.GetName() == joint_name and str(prim.GetPath()).startswith("/World/Robot"):
                joint_prim = prim
                break
        if joint_prim is None:
            print(f"[WARN] Joint prim not found for drive config: {joint_name}")
            continue
        drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if not drive_api:
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive_api.GetStiffnessAttr().Set(drive_cfg["stiffness"])
        drive_api.GetDampingAttr().Set(drive_cfg["damping"])
    print(f"[INFO] Applied joint drives for {len(JOINT_DRIVE_CONFIG)} joints")


# ============================================================================
# Policy Inference Manager
# ============================================================================
class UR10ePickPlaceInference:
    """Manages observation computation, policy inference, and action application
    for the UR10e pick-and-place task deployed in IsaacSim."""

    def __init__(self, policy_path: str, goal_pos: np.ndarray, cube_spawn_pos: np.ndarray,
                 episode_length: int = 600, verbose: bool = False, log_interval: int = 10,
                 csv_path: str | None = None, device: str = "cpu"):
        self.device = device

        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        self.ik_solver = DifferentialIKSolver(
            action_scale=ACTION_SCALE, lambda_val=DLS_LAMBDA, device=device
        )

        self.goal_pos = goal_pos.astype(np.float32)
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.cube_spawn_pos = cube_spawn_pos.copy()

        self.episode_length = episode_length
        self.verbose = verbose
        self.log_interval = log_interval

        # State buffers (initialized in initialize())
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.default_joint_pos = None
        self.default_joint_vel = None
        self.robot = None
        self.cube = None
        self.arm_joint_indices = []
        self.finger_joint_idx = -1
        self.num_joints = 0
        self.wrist_3_link_prim = None

        self._jac_robot = None
        self._jac_link_idx = None

        # Episode tracking
        self.episode_step = 0
        self.episode_count = 0
        self.total_steps = 0

        # CSV logging
        self._csv_file = None
        self._csv_writer = None
        if csv_path:
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "total_step", "episode", "ep_step",
                "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
                "cube_x", "cube_y", "cube_z",
                "ee_cube_dist", "cube_goal_dist",
                "finger_target", "ee_to_cube_raw", "gripper_cmd",
                "action_0", "action_1", "action_2", "action_3", "action_4", "action_5", "action_6",
                "ik_target_j0", "ik_target_j1", "ik_target_j2", "ik_target_j3", "ik_target_j4", "ik_target_j5",
                "reset_reason",
            ])

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def initialize(self, robot: SingleArticulation, cube_prim_path: str):
        """Initialize after world.reset(). Sets up joint mappings and default poses."""
        self.robot = robot
        self.cube = SingleRigidPrim(prim_path=cube_prim_path, name="dex_cube_rigid")

        joint_names = list(self.robot.dof_names)
        self.num_joints = len(joint_names)
        print(f"[INFO] Robot DOFs ({self.num_joints}): {joint_names}")

        self.arm_joint_indices = [joint_names.index(n) for n in ARM_JOINT_NAMES]
        self.finger_joint_idx = joint_names.index("finger_joint")

        self.default_joint_pos = np.zeros(self.num_joints, dtype=np.float32)
        for i, name in enumerate(ARM_JOINT_NAMES):
            self.default_joint_pos[self.arm_joint_indices[i]] = DEFAULT_ARM_POSE[i]
        self.default_joint_vel = np.zeros(self.num_joints, dtype=np.float32)

        self.robot.set_joint_positions(self.default_joint_pos)

        self.wrist_3_link_prim = get_prim_at_path("/World/Robot/wrist_3_link")

        from isaacsim.core.experimental.prims import Articulation as ExpArticulation
        self._jac_robot = ExpArticulation("/World/Robot")
        self._jac_link_idx = self._jac_robot.get_link_indices("wrist_3_link").list()[0]
        print(f"[INFO] wrist_3_link index for Jacobian: {self._jac_link_idx}")
        print(f"[INFO] Articulation links: {self._jac_robot.link_names}")

        obs = self._compute_observation()
        total = obs.shape[-1]
        print(f"[INFO] Observation dimension: {total} (expected: 53)")
        expected = 7 + self.num_joints * 2 + 3 + 4 + 1 + 7 + 3
        print(f"[INFO] Breakdown: 7 + {self.num_joints}*2 + 3 + 4 + 1 + 7 + 3 = {expected}")
        if total != 53:
            print(f"[WARN] Observation dim mismatch! Got {total}, policy expects 53.")

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get end-effector world pose (position, quaternion w,x,y,z)."""
        return get_world_pose_from_relative(
            self.wrist_3_link_prim, EE_BODY_OFFSET_POS, EE_BODY_OFFSET_ROT
        )

    def _get_jacobian(self) -> torch.Tensor:
        """Get geometric Jacobian for the 6 arm joints w.r.t. wrist_3_link.

        Returns shape (1, 6, 6): one env, 6 task-space dims, 6 arm joints.
        """
        jac_all = self._jac_robot.get_jacobian_matrices()
        jac_np = jac_all.numpy()
        jac_ee = jac_np[0, self._jac_link_idx - 1, :, :]
        jac_arm = jac_ee[:, self.arm_joint_indices]
        return torch.tensor(jac_arm, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _compute_observation(self) -> torch.Tensor:
        """Build observation vector matching the training env.

        Layout: [last_action(7), joint_pos_rel(N), joint_vel_rel(N),
                 eef_pos(3), eef_quat(4), gripper_pos(1),
                 target_pose(7), object_pos_rel(3)]
        """
        joint_pos = np.array(self.robot.get_joint_positions(), dtype=np.float32)
        joint_vel = np.array(self.robot.get_joint_velocities(), dtype=np.float32)

        joint_pos_rel = joint_pos - self.default_joint_pos
        joint_vel_rel = joint_vel - self.default_joint_vel

        ee_pos, ee_quat = self._get_ee_pose()
        ee_pos = np.array(ee_pos, dtype=np.float32)
        ee_quat = np.array(ee_quat, dtype=np.float32)

        gripper_obs = np.array([joint_pos[self.finger_joint_idx]], dtype=np.float32)

        target_obs = np.concatenate([self.goal_pos, self.goal_quat])

        cube_pos, _ = self.cube.get_world_pose()
        robot_pos, _ = self.robot.get_world_pose()
        object_pos_rel = np.array(cube_pos - robot_pos, dtype=np.float32)

        obs = np.concatenate([
            self.prev_action,
            joint_pos_rel,
            joint_vel_rel,
            ee_pos,
            ee_quat,
            gripper_obs,
            target_obs,
            object_pos_rel,
        ])

        return torch.from_numpy(obs).unsqueeze(0).to(self.device)

    def reset_episode(self, world, reason: str = ""):
        """Reset robot and cube to initial state for a new episode."""
        self.episode_count += 1
        print(f"[RESET] Episode {self.episode_count} starting (reason: {reason}) "
              f"at total step {self.total_steps}")

        self.prev_action[:] = 0.0
        self.episode_step = 0

        self.robot.set_joint_positions(self.default_joint_pos)
        self.robot.set_joint_velocities(self.default_joint_vel)

        self.cube.set_world_pose(
            position=self.cube_spawn_pos,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.cube.set_linear_velocity(np.zeros(3))
        self.cube.set_angular_velocity(np.zeros(3))

        for _ in range(5):
            world.step(render=False)

        if self._csv_writer:
            self._csv_writer.writerow([
                self.total_steps, self.episode_count, 0,
                "", "", "", "", "", "", "",
                "", "", "", "", "", "", "", "",
                "", "", "", "", "", "", "",
                "", "", "", "", "", "",
                reason,
            ])

    def check_reset_needed(self) -> str | None:
        """Return a reset reason string if the episode should end, else None."""
        cube_pos, _ = self.cube.get_world_pose()

        if cube_pos[2] < CUBE_FALL_Z:
            return f"cube_fell (z={cube_pos[2]:.3f})"

        goal_dist = np.linalg.norm(np.array(cube_pos) - self.goal_pos)
        if goal_dist > CUBE_GOAL_MAX_DIST:
            return f"cube_too_far (dist={goal_dist:.3f})"

        if self.episode_step >= self.episode_length:
            return f"timeout ({self.episode_length} steps)"

        return None

    def step(self, world):
        """Execute one policy inference step and apply actions to the robot.

        Returns a dict of diagnostics for the current step.
        """
        self.episode_step += 1
        self.total_steps += 1

        obs = self._compute_observation()

        with torch.no_grad():
            action = self.policy(obs).cpu().numpy().flatten()

        self.prev_action = action.copy().astype(np.float32)

        arm_cmd = action[:6]
        gripper_cmd = action[6]

        # --- Arm: Differential IK ---
        ee_pos, ee_quat = self._get_ee_pose()
        ee_pos_t = torch.tensor(ee_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        ee_quat_t = torch.tensor(ee_quat, dtype=torch.float32, device=self.device).unsqueeze(0)
        arm_cmd_t = torch.tensor(arm_cmd, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.ik_solver.set_command(arm_cmd_t, ee_pos_t, ee_quat_t)

        jacobian = self._get_jacobian()
        current_arm_pos = np.array(
            [self.robot.get_joint_positions()[i] for i in self.arm_joint_indices], dtype=np.float32
        )
        current_arm_pos_t = torch.tensor(current_arm_pos, device=self.device).unsqueeze(0)

        target_arm_pos = self.ik_solver.compute(
            ee_pos_t, ee_quat_t, jacobian, current_arm_pos_t
        ).cpu().numpy().flatten()

        # --- Gripper: proximity-based ---
        cube_pos, _ = self.cube.get_world_pose()
        ee_to_cube = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
        if ee_to_cube < GRIPPER_CLOSE_THRESHOLD and gripper_cmd > 0:
            target_finger = GRIPPER_CLOSE_POS
        else:
            target_finger = GRIPPER_OPEN_POS

        # --- Apply joint targets ---
        joint_targets = np.array(self.robot.get_joint_positions(), dtype=np.float64)
        for i, arm_idx in enumerate(self.arm_joint_indices):
            joint_targets[arm_idx] = float(target_arm_pos[i])
        joint_targets[self.finger_joint_idx] = target_finger

        self.robot.apply_action(ArticulationAction(joint_positions=joint_targets))

        # --- Diagnostics ---
        ee_pos_np = np.array(ee_pos, dtype=np.float32)
        ee_quat_np = np.array(ee_quat, dtype=np.float32)
        cube_pos_np = np.array(cube_pos, dtype=np.float32)
        cube_goal_dist = float(np.linalg.norm(cube_pos_np - self.goal_pos))

        diag = {
            "ee_pos": ee_pos_np, "ee_quat": ee_quat_np,
            "cube_pos": cube_pos_np,
            "ee_cube_dist": float(ee_to_cube), "cube_goal_dist": cube_goal_dist,
            "finger_target": target_finger, "ee_to_cube_raw": float(ee_to_cube),
            "gripper_cmd": float(gripper_cmd), "action": action,
            "ik_target": target_arm_pos,
        }

        should_log = self.verbose and (self.episode_step % self.log_interval == 0)
        if should_log:
            jp_rel = np.array(self.robot.get_joint_positions(), dtype=np.float32) - self.default_joint_pos
            arm_jp_rel = jp_rel[self.arm_joint_indices]
            ik_des_pos = self.ik_solver.ee_pos_des
            ik_des_quat = self.ik_solver.ee_quat_des
            des_str = ""
            if ik_des_pos is not None:
                dp = ik_des_pos.cpu().numpy().flatten()
                dq = ik_des_quat.cpu().numpy().flatten()
                des_str = (f"  IK desired EE: pos=({dp[0]:.4f},{dp[1]:.4f},{dp[2]:.4f}) "
                           f"quat=({dq[0]:.3f},{dq[1]:.3f},{dq[2]:.3f},{dq[3]:.3f})")
            print(
                f"  [Ep{self.episode_count}|S{self.episode_step:4d}] "
                f"act=[{action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f},"
                f"{action[3]:+.3f},{action[4]:+.3f},{action[5]:+.3f}|g{action[6]:+.3f}] "
                f"EE=({ee_pos_np[0]:.3f},{ee_pos_np[1]:.3f},{ee_pos_np[2]:.3f}) "
                f"Cube=({cube_pos_np[0]:.3f},{cube_pos_np[1]:.3f},{cube_pos_np[2]:.3f}) "
                f"dist_ee_cube={ee_to_cube:.4f} dist_cube_goal={cube_goal_dist:.4f} "
                f"finger={target_finger:.2f} "
                f"arm_jp_rel=[{arm_jp_rel[0]:+.3f},{arm_jp_rel[1]:+.3f},{arm_jp_rel[2]:+.3f},"
                f"{arm_jp_rel[3]:+.3f},{arm_jp_rel[4]:+.3f},{arm_jp_rel[5]:+.3f}]"
            )
            if des_str:
                print(des_str)

        if self._csv_writer:
            self._csv_writer.writerow([
                self.total_steps, self.episode_count, self.episode_step,
                f"{ee_pos_np[0]:.5f}", f"{ee_pos_np[1]:.5f}", f"{ee_pos_np[2]:.5f}",
                f"{ee_quat_np[0]:.5f}", f"{ee_quat_np[1]:.5f}", f"{ee_quat_np[2]:.5f}", f"{ee_quat_np[3]:.5f}",
                f"{cube_pos_np[0]:.5f}", f"{cube_pos_np[1]:.5f}", f"{cube_pos_np[2]:.5f}",
                f"{ee_to_cube:.5f}", f"{cube_goal_dist:.5f}",
                f"{target_finger:.3f}", f"{ee_to_cube:.5f}", f"{gripper_cmd:.5f}",
                f"{action[0]:.5f}", f"{action[1]:.5f}", f"{action[2]:.5f}",
                f"{action[3]:.5f}", f"{action[4]:.5f}", f"{action[5]:.5f}", f"{action[6]:.5f}",
                f"{target_arm_pos[0]:.5f}", f"{target_arm_pos[1]:.5f}", f"{target_arm_pos[2]:.5f}",
                f"{target_arm_pos[3]:.5f}", f"{target_arm_pos[4]:.5f}", f"{target_arm_pos[5]:.5f}",
                "",
            ])

        return diag


# ============================================================================
# Main
# ============================================================================
def main():
    do_clamp = not args.no_clamp

    print("=" * 68)
    print("  UR10e Pick-and-Place  |  RL Policy Inference in IsaacSim")
    print("=" * 68)
    print(f"  Training-safe goal range:")
    print(f"    X: [{GOAL_X_RANGE[0]:.2f}, {GOAL_X_RANGE[1]:.2f}]")
    print(f"    Y: [{GOAL_Y_RANGE[0]:.2f}, {GOAL_Y_RANGE[1]:.2f}]")
    print(f"    Z: [{GOAL_Z_RANGE[0]:.3f}, {GOAL_Z_RANGE[1]:.3f}]")
    print(f"  Training-safe cube range:")
    print(f"    X: [{CUBE_X_RANGE[0]:.2f}, {CUBE_X_RANGE[1]:.2f}]")
    print(f"    Y: [{CUBE_Y_RANGE[0]:.2f}, {CUBE_Y_RANGE[1]:.2f}]")
    print(f"    Z: [{CUBE_Z_RANGE[0]:.3f}, {CUBE_Z_RANGE[1]:.3f}]")
    print(f"  Trained cube default: {CUBE_TRAINED_POS}")
    print(f"  Clamping: {'ON' if do_clamp else 'OFF (--no_clamp)'}")
    print(f"  Verbose: {'ON (interval={args.log_interval})' if args.verbose else 'OFF'}")
    print(f"  Episode length: {args.episode_length} steps")
    print("=" * 68)

    gx, gy, gz = validate_goal(args.goal_x, args.goal_y, args.goal_z, do_clamp)
    cx, cy, cz = validate_cube(args.cube_x, args.cube_y, args.cube_z, do_clamp)

    goal_pos = np.array([gx, gy, gz], dtype=np.float64)
    cube_spawn_pos = np.array([cx, cy, cz], dtype=np.float64)

    cube_offset = np.linalg.norm(cube_spawn_pos - np.array(CUBE_TRAINED_POS))
    if cube_offset > 0.01:
        print(f"[WARN] Cube position deviates {cube_offset:.3f}m from the trained default "
              f"{CUBE_TRAINED_POS}. Policy was trained with a fixed cube position; "
              f"large deviations may cause unpredictable behavior.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_path = os.path.join(script_dir, "policy", "policy.pt")
    if not os.path.exists(policy_path):
        print(f"[ERROR] Policy not found at: {policy_path}")
        sys.exit(1)

    csv_path = os.path.join(script_dir, f"diagnostics_{int(time.time())}.csv") if args.verbose else None
    if csv_path:
        print(f"[INFO] CSV diagnostics will be written to: {csv_path}")

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT,
        rendering_dt=PHYSICS_DT * DECIMATION,
    )

    setup_scene(cube_pos=cube_spawn_pos)
    configure_physics()

    world.reset()
    simulation_app.update()

    robot = SingleArticulation(prim_path="/World/Robot", name="ur10e_robot")
    robot.initialize()

    inference = UR10ePickPlaceInference(
        policy_path=policy_path,
        goal_pos=goal_pos,
        cube_spawn_pos=cube_spawn_pos,
        episode_length=args.episode_length,
        verbose=args.verbose,
        log_interval=args.log_interval,
        csv_path=csv_path,
        device="cpu",
    )
    inference.initialize(robot, "/World/DexCube")

    # --- Scene warmup: let physics settle before inference ---
    print(f"[INFO] Running {WARMUP_STEPS} warmup steps to stabilize scene...")
    for _ in range(WARMUP_STEPS):
        world.step(render=False)
    cube_rest_pos, _ = inference.cube.get_world_pose()
    print(f"[INFO] Cube resting position after warmup: "
          f"({cube_rest_pos[0]:.4f}, {cube_rest_pos[1]:.4f}, {cube_rest_pos[2]:.4f})")
    cube_drop = abs(cube_spawn_pos[2] - cube_rest_pos[2])
    if cube_drop > 0.01:
        print(f"[WARN] Cube dropped {cube_drop:.4f}m during warmup. "
              f"Consider adjusting spawn Z to {cube_rest_pos[2]:.4f}.")

    print(f"[INFO] Goal position: ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f})")
    print(f"[INFO] Running inference (max {args.num_steps} total steps)...")

    inference.reset_episode(world, reason="initial")

    while simulation_app.is_running() and inference.total_steps < args.num_steps:
        world.step(render=True)

        if world.is_playing():
            diag = inference.step(world)

            reset_reason = inference.check_reset_needed()
            if reset_reason:
                inference.reset_episode(world, reason=reset_reason)

            if inference.total_steps % 200 == 0:
                print(
                    f"[Step {inference.total_steps:5d}|Ep{inference.episode_count}|"
                    f"S{inference.episode_step:4d}] "
                    f"EE->Cube: {diag['ee_cube_dist']:.4f}m | "
                    f"Cube->Goal: {diag['cube_goal_dist']:.4f}m | "
                    f"Cube: ({diag['cube_pos'][0]:.3f}, {diag['cube_pos'][1]:.3f}, "
                    f"{diag['cube_pos'][2]:.3f}) | Finger: {diag['finger_target']:.2f}"
                )

        if world.is_stopped():
            break

    print(f"[INFO] Done after {inference.total_steps} total steps, "
          f"{inference.episode_count} episodes.")
    inference.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
