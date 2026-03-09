# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone Isaac Sim example: UR10e pick-up-cube policy.
#
# Matches the training environment from env.yaml:
#   - SeattleLabTable at (0.5, 0, 0) with rot (0.707, 0, 0, 0.707)
#   - Ground plane at z=-1.05
#   - DexCube at (0.5, 0, 0.055) on the table
#   - Robot at origin on the table
#   - Goal cmd range: x=[0.4,0.6], y=[-0.25,0.25], z=[0.255,0.455]
#   - Action scale: 0.1
#   - Default pose (init_state.joint_pos): shoulder_pan=pi, shoulder_lift=-pi/2,
#                   elbow=pi/2, wrist_1=-pi/2, wrist_2=-pi/2, wrist_3=0
#
# Usage (after building the repo):
#   From workspace root, use the kit Python from the build output, e.g.:
#   ./_build/linux-x86_64/release/kit/python/bin/python3 source/standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py
#   Or with standalone Isaac Sim install: ./python.sh standalone_examples/api/isaacsim.robot.policy.examples/ur10e_pick_cube.py

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import define_prim
from isaacsim.robot.policy.examples.robots import UR10eReachTargetPolicy
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

first_step = True
reset_needed = False

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument(
    "--goal", type=float, nargs=3,
    default=[0.5, 0.0, 0.35],
    help="Goal position (x y z) - must be within training range: "
         "x=[0.4,0.6], y=[-0.25,0.25], z=[0.255,0.455]",
)
args, unknown = parser.parse_known_args()


def on_physics_step(step_size) -> None:
    global first_step
    global reset_needed
    if first_step:
        ur10e.initialize()
        joint_pos = ur10e.robot.get_joint_positions()
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow",
                       "wrist_1", "wrist_2", "wrist_3"]
        print("\n  Spawning arm joint positions:")
        for name, val in zip(joint_names, joint_pos[:6]):
            print(f"    {name:>15s}: {val: .4f}")
        print()
        cube_rigid.initialize()
        first_step = False
    elif reset_needed:
        my_world.reset(True)
        reset_needed = False
        first_step = True
    else:
        cube_pos, _ = cube_rigid.get_world_pose()
        ur10e.forward(step_size, goal_position, cube_pos)


my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

stage = my_world.stage

# Ground plane at z=-1.05 (matching training env)
ground_prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
ground_prim.GetReferences().AddReference(asset_path)
xf_ground = UsdGeom.Xformable(ground_prim)
existing_translate = UsdGeom.XformOp(ground_prim.GetAttribute("xformOp:translate"))
if existing_translate:
    existing_translate.Set(Gf.Vec3d(0.0, 0.0, -1.05))
else:
    xf_ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -1.05))

# SeattleLabTable at (0.5, 0, 0) with 90-deg rotation (matching training env)
table_prim = define_prim("/World/Table", "Xform")
table_usd = assets_root_path + "/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd"
table_prim.GetReferences().AddReference(table_usd)
xf_table = UsdGeom.Xformable(table_prim)
xf_table.ClearXformOpOrder()
xf_table.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
xf_table.AddOrientOp().Set(Gf.Quatf(0.707, 0.0, 0.0, 0.707))

# DexCube at (0.5, 0.0, 0.055) on the table
cube_prim_path = "/World/Cube"
cube_prim = define_prim(cube_prim_path, "Xform")
cube_usd = assets_root_path + "/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"
cube_prim.GetReferences().AddReference(cube_usd)
xf_cube = UsdGeom.Xformable(cube_prim)
xf_cube.ClearXformOpOrder()
xf_cube.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.055))
xf_cube.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

cube_rigid = SingleRigidPrim(
    prim_path=cube_prim_path,
    name="cube",
)

# UR10e robot at origin (on the table)
ur10e = UR10eReachTargetPolicy(
    prim_path="/World/UR10e",
    name="UR10e",
    position=np.array([0, 0, 0]),
)

my_world.reset()
my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

goal_position = np.array(args.goal)

print("\n" + "=" * 60)
print("  UR10e Pick Cube - Standalone Policy Example")
print("=" * 60)
print(f"  Table:   (0.5, 0.0, 0.0) - SeattleLabTable")
print(f"  Cube:    (0.5, 0.0, 0.055) - DexCube on table")
print(f"  Robot:   (0.0, 0.0, 0.0) - on table")
print(f"  Goal:    ({goal_position[0]}, {goal_position[1]}, {goal_position[2]})")
print(f"  Training cmd range:")
print(f"    X: [0.4, 0.6]")
print(f"    Y: [-0.25, 0.25]")
print(f"    Z: [0.255, 0.455]")
print(f"  Action scale: 0.1")
print(f"  Default pose: [3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0]")
print("=" * 60 + "\n")

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped():
        reset_needed = True
    if my_world.is_playing():
        if args.test and i >= 500:
            cube_pos, _ = cube_rigid.get_world_pose()
            print("Cube position: ", cube_pos)
            break
        i += 1

simulation_app.close()
