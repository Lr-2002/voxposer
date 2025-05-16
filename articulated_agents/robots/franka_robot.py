# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

from articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams, MobileManipulator, MobileManipulatorParams)

# 0        pedestal_link        base_joint0
# 1        panda_link0        base_joint1
# 2        panda_link1        panda_joint1
# 3        panda_link2        panda_joint2
# 4        panda_link3        panda_joint3
# 5        panda_link4        panda_joint4
# 6        panda_link5        panda_joint5
# 7        panda_link6        panda_joint6
# 8        panda_link7        panda_joint7
# 9        panda_link8        panda_joint8
# 10        panda_hand        panda_hand_joint
# 11        panda_leftfinger        panda_finger_joint1
# 12        panda_rightfinger        panda_finger_joint2


class FrankaRobot(MobileManipulator):
    def _get_franka_params(self):
        return MobileManipulatorParams(
            arm_joints=list(range(2, 9)),
            gripper_joints=[11, 12],
            wheel_joints=[],
            # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/robots/manipulators/panda_robot.py#L37C16-L37C107
            arm_init_params=np.array([
                0,
                np.pi / 16.0,
                0.00,
                -np.pi / 2.0 - np.pi / 3.0,
                0.00,
                np.pi - 0.2,
                np.pi / 4
                ],
                dtype=np.float32,
            ),
            # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/grippers/panda_gripper.py#L25
            gripper_init_params=np.array([0.04, 0.04]),
            ee_offset=[mn.Vector3(0.08, 0, 0)],  # zeroed
            ee_links=[10],
            ee_constraint=np.array([[[0.4, 1.2], [-0.7, 0.7], [0.25, 1.5]]]),
            cameras={
                "articulated_agent_arm": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                    attached_link_id=8,
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(90)),
                ),
                "head": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.35, 1.75, 0.0),
                    cam_look_at_pos=mn.Vector3(1.0, 0.8, 0.0),
                    attached_link_id=-1,
                ),
                # "head+depth": ArticulatedAgentCameraParams(
                #     cam_offset_pos=mn.Vector3(0, 0.0, 0.0),
                #     cam_look_at_pos=mn.Vector3(0.0, 0.0, 0.0),
                #     attached_link_id=-1,
                # ),
                # "head+semantic": ArticulatedAgentCameraParams(
                #     cam_offset_pos=mn.Vector3(0, 0.0, 0.0),
                #     cam_look_at_pos=mn.Vector3(0.0, 0.0, 0.0),
                #     attached_link_id=8,
                #     relative_transform=mn.Matrix4.rotation_x(mn.Deg(180)),
                # ),
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
                "side": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.1, 1.0, -0.5),
                    cam_look_at_pos=mn.Vector3(0.7, 0.3, 0.7),
                    attached_link_id=-1,
                ),
                "front": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(1.2, 1.4, 0.0),
                    cam_look_at_pos=mn.Vector3(-0.7, 0.3, 0.0),
                    attached_link_id=-1,
                ),
                "chest": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.0, 1.0, 0.0),
                    cam_look_at_pos=mn.Vector3(0.7, 1.0, 0.0),
                    attached_link_id=-1,
                ),
                "side2": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.1, 1.0, 0.5),
                    cam_look_at_pos=mn.Vector3(0.7, 0.3, -0.7),
                    attached_link_id=-1,
                ),
                "top": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.5, 1.7, 0.1),
                    cam_look_at_pos=mn.Vector3(0.5, 0.2, 0.1),
                    # cam_offset_pos=mn.Vector3(0.5, 0.8, -0.5),
                    # cam_look_at_pos=mn.Vector3(0.5, 0.8, 1),
                    attached_link_id=-1,
                ),
            },
            gripper_closed_state=np.array([0.0, 0.0], dtype=np.float32),
            gripper_open_state=np.array([0.04, 0.04], dtype=np.float32),
            gripper_state_eps=0.001,
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
            wheel_mtr_pos_gain=0.0,
            wheel_mtr_vel_gain=1.3,
            wheel_mtr_max_impulse=10.0,
            base_offset=mn.Vector3(0, 0, 0),
            base_link_names={
                'omron_base_link',
                'pedestal_link'
            }, # unused tho
        )
    def __init__(
        self, urdf_path, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(
            self._get_franka_params(),
            urdf_path,
            sim,
            limit_robo_joints,
            fixed_base,
        )
        self.head_pan_joint_id = 4
        self.head_tilt_joint_id = 5

    def reconfigure(self) -> None:
        super().reconfigure()
        super().update()

    def reset(self) -> None:
        super().reset()
        super().update()

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    def update(self):
        super().update()
