import magnum as mn
import numpy as np

from articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams, MobileManipulator, MobileManipulatorParams)

# 0        cliff.rear_left_mount        cliff.rear_left
# 1        cliff.rear_right_mount        cliff.rear_right
# 2        joint_base_imu_mount        link_base_imu_mount
# 3        joint_imu.base        imu.base
# 4        joint_head_pan        link_head_pan
# 5        joint_head_tilt        link_head_tilt
# 6        joint_camera_mount        link_camera
# 7        joint_neck_imu_mount        link_neck_imu_mount
# 8        joint_imu.neck        imu.neck
# 9        joint_rear_wheel_left        link_rear_wheel_left
# 10        joint_rear_wheel_right        link_rear_wheel_right
# 11        joint_torso        link_torso
# 12        joint_shoulder        link_shoulder
# 13        joint_bicep        link_bicep
# 14        joint_elbow        link_elbow
# 15        joint_forearm        link_forearm
# 16        joint_wrist        link_wrist
# 17        joint_gripper        link_gripper
# 18        gripper_tcp_joint        link_gripper_tcp
# 19        joint_finger_left        link_finger_left
# 20        joint_finger_tip_left        link_finger_tip_left
# 21        joint_finger_nail_left        link_finger_nail_left
# 22        joint_finger_right        link_finger_right
# 23        joint_finger_tip_right        link_finger_tip_right
# 24        joint_finger_nail_right        link_finger_nail_right
# 25        joint_wheel_left        link_wheel_left
# 26        joint_wheel_right        link_wheel_right
# 27        link_base_inertial_joint        link_base_inertial
# 28        time_of_flight.left_1_mount        time_of_flight.left_1
# 29        time_of_flight.left_2_mount        time_of_flight.left_2
# 30        time_of_flight.left_back_1_mount        time_of_flight.left_back_1
# 31        time_of_flight.left_back_2_mount        time_of_flight.left_back_2
# 32        time_of_flight.right_1_mount        time_of_flight.right_1
# 33        time_of_flight.right_2_mount        time_of_flight.right_2
# 34        time_of_flight.right_back_1_mount        time_of_flight.right_back_1
# 35        time_of_flight.right_back_2_mount        time_of_flight.right_back_2

class GoogleRobot(MobileManipulator):
    def _get_googlerobot_params(self):
        return MobileManipulatorParams(
            arm_joints=list(range(11, 18)),
            gripper_joints=[19, 22],
            wheel_joints=[], # [25, 26],
            arm_init_params=np.array([
                -0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774
                ],
                dtype=np.float32,
            ),
            gripper_init_params=np.array([0, 0], dtype=np.float32),
            ee_offset=[mn.Vector3(0.08, 0, 0)], # TBD
            ee_links=[18],
            ee_constraint=np.array([[[0.4, 1.2], [-0.7, 0.7], [0.25, 1.5]]]), # TBD, unused tho
            cameras={
                "articulated_agent_arm": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                    attached_link_id=18,
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(90)),
                ),
                "head": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.0),
                    cam_look_at_pos=mn.Vector3(0.0, 0.0, 0.0),
                    attached_link_id=6,
                    relative_transform=mn.Matrix4.rotation_x(mn.Deg(180)),
                ),
                # "head+depth": ArticulatedAgentCameraParams(
                #     cam_offset_pos=mn.Vector3(0, 0.0, 0.0),
                #     cam_look_at_pos=mn.Vector3(0.0, 0.0, 0.0),
                #     attached_link_id=6,
                #     relative_transform=mn.Matrix4.rotation_x(mn.Deg(180)),
                # ),
                # "head+semantic": ArticulatedAgentCameraParams(
                #     cam_offset_pos=mn.Vector3(0, 0.0, 0.0),
                #     cam_look_at_pos=mn.Vector3(0.0, 0.0, 0.0),
                #     attached_link_id=6,
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
                    cam_offset_pos=mn.Vector3(1.0, 1.4, 0.0),
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
            gripper_closed_state=np.array([1.0, 1.0], dtype=np.float32),
            gripper_open_state=np.array([0.0, 0.0], dtype=np.float32),
            gripper_state_eps=0.001,
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
            wheel_mtr_pos_gain=0.0,
            wheel_mtr_vel_gain=1.3,
            wheel_mtr_max_impulse=10.0,
            base_offset=mn.Vector3(0, 0, 0),
            base_link_names={ # unused tho
                "link_base",
                "link_base_inertial",
                "link_wheel_left",
                "link_wheel_right",
                "link_rear_wheel_left",
                "link_rear_wheel_right",
                "link_torso",
                "link_head_pan",
                "link_base_imu_mount",
                "link_neck_imu_mount",
            },
        )


    def __init__(
        self, urdf_path, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(
            self._get_googlerobot_params(),
            urdf_path,
            sim,
            limit_robo_joints,
            fixed_base,
        )
        self.head_pan_joint_id = 4
        self.head_tilt_joint_id = 5

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
        # Fix the head
        self._set_joint_pos(self.head_pan_joint_id, -0.00285961)
        self._set_motor_pos(self.head_pan_joint_id, -0.00285961)
        self._set_joint_pos(self.head_tilt_joint_id, 0.7851361)
        self._set_motor_pos(self.head_tilt_joint_id, 0.7851361)
