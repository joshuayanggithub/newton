# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
# Example Sim Cloth
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the  newton.ModelBuilder INSTEAD of wp. sim. ModelBuilder().
#
###########################################################################
import math
import sys
import os
import random

import numpy as np

import warp as wp
import warp.examples

from pxr import Usd, UsdGeom, Gf

from enum import Enum

import newton
import newton.solvers.euler.solver_euler
import newton.utils
import newton.examples


class SolverType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"

    def __str__(self):
        return self.value

class Example:
    def __init__(
        self, stage_path="example_cloth.usd", solver: SolverType = SolverType.EULER, height=32, width=64, num_envs=1
    ): 
        self.solver_type = solver

        self.sim_height = height
        self.sim_width = width

        #simulation timing parameters
        self.fps = 60
        self.num_substeps = 32
        self.frame_dt = 1.0 / self.fps

        self.sim_dt = self.frame_dt / self.num_substeps
        self.sim_time = 0.0
        self.profiler = {}
        
        #original frames
        self.num_frames = 300

        cloth_builder = newton.ModelBuilder(up_axis='Y')
        if self.solver_type == SolverType.EULER:
            cloth_builder.add_cloth_grid(
                pos=wp.vec3(0.0, 4.0, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.sim_width,
                dim_y=self.sim_height,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.1,
                fix_left=True,
                tri_ke=1.0e3,
                tri_ka=1.0e3,
                tri_kd=1.0e1,
            )
        elif self.solver_type == SolverType.XPBD:
            cloth_builder.add_cloth_grid(
                pos=wp.vec3(0.0, 4.0, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.sim_width,
                dim_y=self.sim_height,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.1,
                fix_left=True,
                edge_ke=1.0e2,
                add_springs=True,
                spring_ke=1.0e3,
                spring_kd=0.0,
            )
        else:
            # VBD
            cloth_builder.add_cloth_grid(
                pos=wp.vec3(0.0, 4.0, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.sim_width,
                dim_y=self.sim_height,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.1,
                fix_left=True,
                tri_ke=1e4,
                tri_ka=1e4,
                tri_kd=1e-5,
                edge_ke=100,
            )

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        mesh = newton.Mesh(mesh_points, mesh_indices)

        cloth_builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            xform=wp.transform(
                wp.vec3(1.0, 0.0, 1.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5)
            ),
            scale=wp.vec3(2.0, 2.0, 2.0),
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=1.0e2,
                kd=1.0e2,
                kf=1.0e1,
            )
        )

        # only difference between newton and wp ModelBuilder
        builder = newton.ModelBuilder(up_axis='Y')

        self.num_envs = num_envs

        env_offsets = newton.examples.compute_env_offsets(num_envs, env_offset=(0.0, 0.0, 5.0), up_axis=newton.Axis.Y) #same as rewarped-internal...

        print(env_offsets)

        # do the same for each env
        for i in range(self.num_envs):
            builder.add_builder(cloth_builder, xform=wp.transform(env_offsets[i], wp.quat_identity()))
        
        # ground_cfg = builder.default_shape_cfg
        # ground_cfg.ke = 0    # default 1.0e5 WAY TOO LARGE???, anything > 1.0e2 instable
        # ground_cfg.kd = 1.0e3    # default 1000
        # # ground_cfg.kf = 1.0e3    # default 000
        # ground_cfg.restitution = 0.0
        # ground_cfg.mu = 0.25

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e2,
            kd=1.0e2,
            kf=1.0e1,
            mu=0.25,
            thickness=0.01
        )

        # ground_cfg = newton.ModelBuilder.ShapeConfig(
        #     ke=1.0e2,
        #     kd=1.0e3,
        #     kf=1.0e2,
        #     mu=0.25,
        #     thickness=0.01
        # )


        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=0,
            kd=1,
            kf=0,
            mu=0,
        )

        builder.add_ground_plane(cfg=ground_cfg)
        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e2

        if self.solver_type == SolverType.EULER:
            self.solver = newton.solvers.SemiImplicitSolver(self.model)
        elif self.solver_type == SolverType.XPBD:
            self.solver = newton.solvers.XPBDSolver(self.model, iterations=1)
        else:
            self.solver = newton.solvers.VBDSolver(self.model, iterations=1)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        if stage_path:
            self.renderer = newton.utils.SimRenderer(self.model, stage_path, scaling=40.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        

    def simulate(self):
        contacts = self.model.collide(self.state_0)

        for _ in range(self.num_substeps):
            self.state_0.clear_forces()

            self.solver.step(state_in=self.state_0, state_out=self.state_1, control=self.control, contacts=contacts, dt=self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)
# 
    def step(self):
        with wp.ScopedTimer("step", dict=self.profiler):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            # self._render_coordinate_axes()
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cloth.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=4, help="Total number of simulated environments.")
    parser.add_argument(
        "--solver",
        help="Type of solver to use.",
        type=SolverType,
        choices=list(SolverType),
        default=SolverType.EULER,
    )
    parser.add_argument("--width", type=int, default=64, help="Cloth resolution in x.")
    parser.add_argument("--height", type=int, default=32, help="Cloth resolution in y.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, solver=args.solver, num_envs=args.num_envs, height=args.height, width=args.width)

        for _i in range(args.num_frames):
            example.step()
            example.render()

        frame_times = example.profiler["step"]
        print(f"\nAverage frame sim time: {sum(frame_times) / len(frame_times):.2f} ms")

        if example.renderer:
            example.renderer.save()
