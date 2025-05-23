# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

from articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams, MobileManipulator, MobileManipulatorParams)

# 0        bellows_joint        bellows_link
# 1        estop_joint        estop_link
# 2        l_wheel_joint        l_wheel_link
# 3        laser_joint        laser_link
# 4        r_wheel_joint        r_wheel_link
# 5        torso_fixed_joint        torso_fixed_link
# 6        torso_lift_joint        torso_lift_link
# 7        bellows_joint2        bellows_link2
# 8        head_pan_joint        head_pan_link
# 9        head_tilt_joint        head_tilt_link
# 10        head_camera_joint        head_camera_link
# 11        head_camera_depth_joint        head_camera_depth_frame
# 12        head_camera_depth_optical_joint        head_camera_depth_optical_frame
# 13        head_camera_rgb_joint        head_camera_rgb_frame
# 14        head_camera_rgb_optical_joint        head_camera_rgb_optical_frame
# 15        shoulder_pan_joint        shoulder_pan_link
# 16        shoulder_lift_joint        shoulder_lift_link
# 17        upperarm_roll_joint        upperarm_roll_link
# 18        elbow_flex_joint        elbow_flex_link
# 19        forearm_roll_joint        forearm_roll_link
# 20        wrist_flex_joint        wrist_flex_link
# 21        wrist_roll_joint        wrist_roll_link
# 22        gripper_axis        gripper_link
# 23        suction_joint        suction_link


class FetchRobot(MobileManipulator):
    def _get_fetch_params(self):
        return MobileManipulatorParams(
            arm_joints=list(range(15, 22)),
            gripper_joints=[23, 24],
            wheel_joints=[2, 4],
            arm_init_params=np.array(
                [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005],
                dtype=np.float32,
            ),
            gripper_init_params=np.array([0.00, 0.00], dtype=np.float32),
            ee_offset=[mn.Vector3(0.08, 0, 0)],
            ee_links=[22],
            ee_constraint=np.array([[[0.4, 1.2], [-0.7, 0.7], [0.25, 1.5]]]),
            cameras={
                "articulated_agent_arm": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                    attached_link_id=22,
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(90)),
                ),
                "head": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.25, 1.2, 0.0),
                    cam_look_at_pos=mn.Vector3(0.75, 1.0, 0.0),
                    attached_link_id=-1, # -1 means base
                ),
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
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
                "base_link",
                "r_wheel_link",
                "l_wheel_link",
                "r_wheel_link",
                "bellows_link",
                "bellows_link2",
                "estop_link",
                "laser_link",
                "torso_fixed_link",
            },
        )

    def __init__(
        self, urdf_path, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(
            self._get_fetch_params(),
            urdf_path,
            sim,
            limit_robo_joints,
            fixed_base,
        )
        self.back_joint_id = 6
        self.head_rot_jid = 8
        self.head_tilt_jid = 9

    def reconfigure(self) -> None:
        super().reconfigure()

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    def reset(self) -> None:
        super().reset()

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    def update(self):
        super().update()
        # Fix the head.
        self._set_joint_pos(self.head_rot_jid, 0)
        self._set_motor_pos(self.head_rot_jid, 0)
        self._set_joint_pos(self.head_tilt_jid, np.pi / 2)
        self._set_motor_pos(self.head_tilt_jid, np.pi / 2)
        # Fix the back
        fix_back_val = 0.15
        self._set_joint_pos(self.back_joint_id, fix_back_val)
        self._set_motor_pos(self.back_joint_id, fix_back_val)


class FetchRobotNoWheels(FetchRobot):
    def __init__(
        self, urdf_path, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(urdf_path, sim, limit_robo_joints, fixed_base)
        self.back_joint_id -= 2
        self.head_rot_jid -= 2
        self.head_tilt_jid -= 2

    def _get_fetch_params(self):
        params = super()._get_fetch_params()
        # No wheel control
        params.arm_joints = [x - 2 for x in params.arm_joints]
        params.gripper_joints = [x - 2 for x in params.gripper_joints]
        params.wheel_joints = None
        params.ee_links = [params.ee_links[0] - 2]
        return params
