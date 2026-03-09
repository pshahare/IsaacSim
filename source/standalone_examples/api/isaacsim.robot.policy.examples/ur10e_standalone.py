# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
args, unknown = parser.parse_known_args()


def on_physics_step(step_size) -> None:
    global first_step
    global reset_needed
    if first_step:
        ur10e.initialize()
        cube_rigid.initialize()
        first_step = False
    elif reset_needed:
        my_world.reset(True)
        reset_needed = False
        first_step = True
    else:
        cube_pos, _ = cube_rigid.get_world_pose()
        ur10e.forward(step_size, goal_position, cube_pos)


my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# Ground plane
ground_prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
ground_prim.GetReferences().AddReference(asset_path)

# Cube (Rubik's cube sized ~5.5cm, placed in front of the robot)
stage = my_world.stage
cube_prim_path = "/World/Cube"
cube_geom = UsdGeom.Cube.Define(stage, cube_prim_path)
cube_geom.GetSizeAttr().Set(0.055)
cube_geom.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0275))
cube_geom.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
cube_geom.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

UsdPhysics.RigidBodyAPI.Apply(cube_geom.GetPrim())
UsdPhysics.CollisionAPI.Apply(cube_geom.GetPrim())
UsdPhysics.MassAPI.Apply(cube_geom.GetPrim()).CreateMassAttr().Set(0.1)

# Give the cube a color so it's visible
material_path = "/World/Cube/Material"
material = UsdShade.Material.Define(stage, material_path)
shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.2, 0.2))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(cube_geom.GetPrim()).Bind(material)

cube_rigid = SingleRigidPrim(
    prim_path=cube_prim_path,
    name="cube",
)

# UR10e robot at origin
ur10e = UR10eReachTargetPolicy(
    prim_path="/World/UR10e",
    name="UR10e",
    position=np.array([0, 0, 0]),
)

my_world.reset()
my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

# Goal: lift the cube to this position (above its starting location)
goal_position = np.array([1.0, 0.0, 0.35])

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
