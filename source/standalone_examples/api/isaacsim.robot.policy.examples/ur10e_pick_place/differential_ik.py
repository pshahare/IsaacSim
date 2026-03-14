# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained Differential IK solver extracted from IsaacLab.
# Uses Damped Least Squares (DLS) method matching the training configuration.
# No dependency on IsaacLab at runtime.

from __future__ import annotations

import torch


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


def apply_delta_pose(
    source_pos: torch.Tensor, source_rot: torch.Tensor, delta_pose: torch.Tensor, eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a 6D delta (dx,dy,dz, droll,dpitch,dyaw) to a source pose."""
    num_poses = source_pos.shape[0]
    device = source_pos.device
    target_pos = source_pos + delta_pose[:, 0:3]
    rot_actions = delta_pose[:, 3:6]
    angle = torch.linalg.vector_norm(rot_actions, dim=1)
    axis = rot_actions / angle.unsqueeze(-1)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_poses, 1)
    rot_delta_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > eps, quat_from_angle_axis(angle, axis), identity_quat
    )
    target_rot = quat_mul(rot_delta_quat, source_rot)
    return target_pos, target_rot


def compute_pose_error(
    ee_pos: torch.Tensor, ee_quat: torch.Tensor, des_pos: torch.Tensor, des_quat: torch.Tensor
) -> torch.Tensor:
    """Compute 6D pose error (position + axis-angle orientation error)."""
    pos_error = des_pos - ee_pos
    source_quat_norm = quat_mul(ee_quat, quat_conjugate(ee_quat))[:, 0]
    source_quat_inv = quat_conjugate(ee_quat) / source_quat_norm.unsqueeze(-1)
    quat_error = quat_mul(des_quat, source_quat_inv)
    axis_angle_error = axis_angle_from_quat(quat_error)
    return torch.cat((pos_error, axis_angle_error), dim=1)


class DifferentialIKSolver:
    """Self-contained Differential IK controller using Damped Least Squares.

    Matches IsaacLab's DifferentialIKController with:
      - command_type = "pose"
      - use_relative_mode = True
      - ik_method = "dls"
      - lambda_val = 0.01
      - action_scale = 0.1
    """

    def __init__(self, action_scale: float = 0.1, lambda_val: float = 0.01, device: str = "cpu"):
        self.action_scale = action_scale
        self.lambda_val = lambda_val
        self.device = device
        self.ee_pos_des = None
        self.ee_quat_des = None

    def set_command(self, command: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor):
        """Set target pose from a 6D relative command (dx,dy,dz, droll,dpitch,dyaw).

        The command is scaled by action_scale before being applied as a delta.
        """
        scaled_command = command * self.action_scale
        self.ee_pos_des, self.ee_quat_des = apply_delta_pose(ee_pos, ee_quat, scaled_command)

    def compute(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute target joint positions using DLS IK.

        Args:
            ee_pos: Current end-effector position (N, 3).
            ee_quat: Current end-effector quaternion (w,x,y,z) (N, 4).
            jacobian: Geometric Jacobian (N, 6, num_joints).
            joint_pos: Current joint positions (N, num_joints).

        Returns:
            Target joint positions (N, num_joints).
        """
        pose_error = compute_pose_error(ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des)
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (self.lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self.device)
        delta_joint_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ pose_error.unsqueeze(-1)
        delta_joint_pos = delta_joint_pos.squeeze(-1)
        return joint_pos + delta_joint_pos
