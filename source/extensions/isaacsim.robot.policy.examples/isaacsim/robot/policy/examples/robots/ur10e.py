# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
from typing import Optional

import carb
import numpy as np
import torch
import torch.nn as nn
import yaml
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.transformations import get_world_pose_from_relative
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.robot.policy.examples.controllers.config_loader import get_physics_properties
from isaacsim.storage.native import get_assets_root_path


def _find_policy_dir():
    """Walk up from this file to find the directory containing ur10e_policy/."""
    path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        candidate = os.path.join(path, "ur10e_policy")
        if os.path.isdir(candidate):
            return candidate
        path = os.path.dirname(path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ur10e_policy")


_POLICY_DIR = _find_policy_dir()

# UR10e + Robotiq 2f-140 has 14 actuated DOFs total
_NUM_ARM_JOINTS = 6
_NUM_TOTAL_DOFS = 14
_NUM_ACTIONS = 7  # 6 arm joints + 1 gripper (finger_joint)

# Observation vector layout (53 total):
#   policy group (43):
#     last_action:    7
#     joint_pos_rel: 14
#     joint_vel_rel: 14
#     eef_pos:        3
#     eef_quat:       4
#     gripper_pos:    1
#   task_obs group (10):
#     target_object_position (command pose): 7  (pos xyz + quat wxyz)
#     object_position (in robot frame):      3
_OBS_DIM = 53


class _ActorMLP(nn.Module):
    """Reconstructs the RSL-RL ActorCritic actor network: Linear-ELU stack."""

    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)


class UR10eReachTargetPolicy(PolicyController):
    """The UR10e manipulator with Robotiq 2f-140 gripper running a lift-object policy."""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "ur10e",
        usd_path: Optional[str] = None,
        policy_path: Optional[str] = None,
        env_config_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize UR10e robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the robot
            usd_path (str) -- robot usd filepath in the directory
            policy_path (Optional[str]) -- path to the policy .pt checkpoint file
            env_config_path (Optional[str]) -- path to the environment config .yaml file
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        assets_root_path = get_assets_root_path()
        if usd_path is None:
            usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"

        # Manually replicate PolicyController.__init__ so we can set the
        # gripper variant between adding the USD reference and creating
        # the SingleArticulation.
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        # Apply Robotiq 2f-140 gripper variant before the articulation is created
        vset = prim.GetVariantSets().GetVariantSet("Gripper")
        if vset:
            vset.SetVariantSelection("Robotiq_2f_140")

        if root_path is None:
            self.robot = SingleArticulation(prim_path=prim_path, name=name, position=position, orientation=orientation)
        else:
            self.robot = SingleArticulation(prim_path=root_path, name=name, position=position, orientation=orientation)

        if policy_path is None:
            policy_path = os.path.join(_POLICY_DIR, "ur10e_policy.pt")
        if env_config_path is None:
            env_config_path = os.path.join(_POLICY_DIR, "env.yaml")

        self.load_policy(policy_path, env_config_path)
        self._action_scale = 0.1
        self._previous_action = np.zeros(_NUM_ACTIONS)
        self.action = np.zeros(_NUM_ACTIONS)
        self._policy_counter = 0

        # Initial joint pose from env.yaml init_state.joint_pos
        self._training_default_pos = np.array([
            3.141592653589793, -1.5707963267948966, 1.5707963267948966,
            -1.5707963267948966, -1.5707963267948966, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])

        # End-effector frame: wrist_3_link with offset (0, 0, 0.24) per env.yaml
        self._ee_prim = None
        self._ee_offset_pos = np.array([0.0, 0.0, 0.24])
        self._ee_offset_rot = np.array([1.0, 0.0, 0.0, 0.0])

    def load_policy(self, policy_file_path, policy_env_path) -> None:
        """
        Loads an RSL-RL checkpoint and env config from local filesystem paths.
        Reconstructs the actor MLP from the saved state dict.
        """

        class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
            def ignore_unknown(self, node) -> None:
                return None

            def tuple_constructor(loader, node) -> tuple:
                return tuple(loader.construct_sequence(node))

        SafeLoaderIgnoreUnknown.add_constructor(
            "tag:yaml.org,2002:python/tuple", SafeLoaderIgnoreUnknown.tuple_constructor
        )
        SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

        policy_file_path = os.path.realpath(policy_file_path)
        policy_env_path = os.path.realpath(policy_env_path)

        checkpoint = torch.load(policy_file_path, map_location="cpu", weights_only=False)
        self.policy = _ActorMLP(_OBS_DIM, _NUM_ACTIONS)
        actor_state = {k: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("actor.")}
        self.policy.load_state_dict(actor_state, strict=True)
        self.policy.eval()

        with open(policy_env_path, "r") as f:
            self.policy_env_params = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the articulation interface and cache the end-effector prim.
        """
        super().initialize(physics_sim_view=physics_sim_view)

        # Override default_pos with the training default so observations are in-distribution
        self.default_pos = self._training_default_pos.copy()
        self.default_vel = np.zeros(_NUM_TOTAL_DOFS)

        # Set the robot to the training default pose
        self.robot.set_joint_positions(self._training_default_pos)

        self._gripper_joint_idx = self.robot.get_dof_index("finger_joint")

        # Cache arm joint indices for the 6 controlled arm joints
        arm_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self._arm_joint_indices = [self.robot.get_dof_index(n) for n in arm_joint_names]

        # Find the body index of wrist_3_link in the articulation for Jacobian lookup
        body_names = self.robot._articulation_view.body_names
        self._ee_body_idx = None
        for i, bname in enumerate(body_names):
            if "wrist_3_link" in bname:
                self._ee_body_idx = i
                break

        from pxr import Usd

        robot_prim = get_prim_at_path(self.robot.prim_path)
        self._ee_prim = None
        for child in Usd.PrimRange(robot_prim):
            if child.GetName() == "wrist_3_link":
                self._ee_prim = child
                break


    def _compute_observation(self, goal_position, object_position):
        """
        Compute the 53-element observation vector matching the trained policy.

        Args:
        goal_position (np.ndarray) -- the goal lift position (x, y, z)
        object_position (np.ndarray) -- the current object world position (x, y, z)

        Returns:
        np.ndarray -- The observation vector.

        """
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()

        if self._ee_prim is not None and self._ee_prim.IsValid():
            ee_world_pos, ee_world_quat = get_world_pose_from_relative(
                self._ee_prim, self._ee_offset_pos, self._ee_offset_rot
            )
        else:
            robot_pos, robot_quat = self.robot.get_world_pose()
            ee_world_pos = robot_pos + self._ee_offset_pos
            ee_world_quat = robot_quat

        gripper_pos = current_joint_pos[self._gripper_joint_idx]

        obs = np.zeros(_OBS_DIM)
        idx = 0

        # last_action (7)
        obs[idx : idx + _NUM_ACTIONS] = self._previous_action
        idx += _NUM_ACTIONS

        # joint_pos_rel (14)
        obs[idx : idx + _NUM_TOTAL_DOFS] = current_joint_pos - self.default_pos
        idx += _NUM_TOTAL_DOFS

        # joint_vel_rel (14)
        obs[idx : idx + _NUM_TOTAL_DOFS] = current_joint_vel - self.default_vel
        idx += _NUM_TOTAL_DOFS

        # eef_pos (3)
        obs[idx : idx + 3] = ee_world_pos
        idx += 3

        # eef_quat (4)
        obs[idx : idx + 4] = ee_world_quat
        idx += 4

        # gripper_pos (1)
        obs[idx] = gripper_pos
        idx += 1

        # target_object_position: command pose (3 pos + 4 quat = 7)
        obs[idx : idx + 3] = goal_position
        idx += 3
        obs[idx : idx + 4] = np.array([1.0, 0.0, 0.0, 0.0])
        idx += 4

        # object_position in robot frame (3)
        robot_pos, _ = self.robot.get_world_pose()
        obs[idx : idx + 3] = np.array(object_position) - robot_pos
        idx += 3

        return obs

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded actor network.
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs_tensor).detach().view(-1).numpy()
        return action

    def _apply_diff_ik(self, ee_delta_pose):
        """
        Apply differential IK using damped least squares (DLS) to convert
        end-effector pose deltas to joint position deltas.

        Args:
            ee_delta_pose (np.ndarray): 6D EE pose delta (dx, dy, dz, droll, dpitch, dyaw)

        Returns:
            np.ndarray: joint position deltas for the 6 arm joints
        """
        jacobians = self.robot._articulation_view.get_jacobians()
        if jacobians is None:
            return np.zeros(_NUM_ARM_JOINTS)

        if hasattr(jacobians, 'numpy'):
            jacobians = jacobians.numpy()

        # Jacobian has shape (batch, num_bodies-1, 6, num_dofs) -- base_link is excluded,
        # so subtract 1 from the body index to get the Jacobian row.
        ee_idx = self._ee_body_idx - 1 if self._ee_body_idx is not None else -1

        arm_indices = self._arm_joint_indices
        num_dofs = jacobians.shape[3]
        valid_arm_indices = [i for i in arm_indices if i < num_dofs]
        if len(valid_arm_indices) != _NUM_ARM_JOINTS:
            return np.zeros(_NUM_ARM_JOINTS)

        J = jacobians[0, ee_idx, :, :][:, valid_arm_indices]  # (6, 6)

        lambda_val = 0.01
        JJT = J @ J.T + (lambda_val ** 2) * np.eye(6)
        try:
            delta_q = J.T @ np.linalg.solve(JJT, ee_delta_pose)
        except np.linalg.LinAlgError:
            return np.zeros(_NUM_ARM_JOINTS)

        if np.any(np.isnan(delta_q)) or np.any(np.isinf(delta_q)):
            return np.zeros(_NUM_ARM_JOINTS)

        max_delta = 0.1
        delta_q = np.clip(delta_q, -max_delta, max_delta)

        return delta_q

    def forward(self, dt, goal_position, object_position):
        """
        Compute the desired articulation action and apply them to the robot articulation.

        The policy outputs 7 actions:
          [0:6] - EE pose deltas (dx, dy, dz, droll, dpitch, dyaw) for differential IK
          [6]   - binary gripper command (>0 = close, <=0 = open)

        Args:
        dt (float) -- Timestep update in the world.
        goal_position (np.ndarray) -- the goal lift position (x, y, z)
        object_position (np.ndarray) -- the current object world position (x, y, z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(goal_position, object_position)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        # Arm: convert EE pose deltas to joint deltas via differential IK
        ee_delta = self.action[:_NUM_ARM_JOINTS] * self._action_scale
        joint_deltas = self._apply_diff_ik(ee_delta)

        current_joint_pos = self.robot.get_joint_positions()
        if np.any(np.isnan(current_joint_pos)) or np.any(np.isinf(current_joint_pos)):
            self.robot.set_joint_positions(self._training_default_pos)
            return

        joint_targets = current_joint_pos.copy()
        for i, idx in enumerate(self._arm_joint_indices):
            joint_targets[idx] = current_joint_pos[idx] + joint_deltas[i]

        # Gripper: binary action
        if self.action[_NUM_ARM_JOINTS] > 0:
            joint_targets[self._gripper_joint_idx] = 0.7
        else:
            joint_targets[self._gripper_joint_idx] = 0.0

        if np.any(np.isnan(joint_targets)) or np.any(np.isinf(joint_targets)):
            return

        # Clamp arm joints to safe range (±2π)
        for idx in self._arm_joint_indices:
            joint_targets[idx] = np.clip(joint_targets[idx], -2 * np.pi, 2 * np.pi)

        action = ArticulationAction(joint_positions=joint_targets)
        self.robot.apply_action(action)

        self._policy_counter += 1
