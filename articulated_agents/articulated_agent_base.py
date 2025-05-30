# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.physics import JointMotorSettings
from habitat_sim.simulator import Simulator

from articulated_agents.articulated_agent_interface import \
    ArticulatedAgentInterface


class ArticulatedAgentBase(ArticulatedAgentInterface):
    """Generic manupulator interface defines standard API functions. Robot with a controllable base."""

    def __init__(
        self,
        params,
        urdf_path: str,
        sim: Simulator,
        limit_robo_joints: bool = True,
        fixed_based: bool = True,
        maintain_link_order=False,
        base_type="mobile",
        sim_obj=None,
        **kwargs,
    ):
        r"""Constructor
        :param params: The parameter of the base articulated agent.
        :param urdf_path: The path to the articulated agent's URDF file.
        :param sim: The simulator.
        :param limit_robo_joints: If true, joint limits of articulated agent are always
            enforced.
        :param fixed_base: If the articulated agent's base is fixed or not.
        :param maintain_link_order: Whether to to preserve the order of
            links parsed from URDF files as link indices. Needed for
            compatibility with PyBullet.
        :param sim_obj: Pointer to the simulated object
        """
        assert base_type in [
            "mobile",
            "leg",
        ], f"'{base_type}' is invalid - valid options are [mobile, leg]. Or you write your own class."
        ArticulatedAgentInterface.__init__(self)
        # Assign the variables
        self.params = params
        self.urdf_path = urdf_path
        self._sim = sim
        self._limit_robo_joints = limit_robo_joints
        self._base_type = base_type
        self.sim_obj = sim_obj
        self._maintain_link_order = maintain_link_order

        # NOTE: the follow members cache static info for improved efficiency over querying the API
        # maps joint ids to motor settings for convenience
        self.joint_motors: Dict[int, Tuple[int, JointMotorSettings]] = {}
        # maps joint ids to position index
        self.joint_pos_indices: Dict[int, int] = {}
        # maps joint ids to velocity index
        self.joint_limits: Tuple[np.ndarray, np.ndarray] = None
        # maps joint ids to velocity index
        self.joint_dof_indices: Dict[int, int] = {}
        # set the base fixed or not
        self._fixed_base = fixed_based
        # set the fixed joint values
        self._fix_joint_values: Optional[np.ndarray] = None

        # set the camera parameters if provided
        self._cameras = None
        if hasattr(self.params, "cameras"):
            self._cameras = defaultdict(list)
            for camera_prefix in self.params.cameras:
                for sensor_name in self._sim._sensors:
                    if sensor_name.startswith(camera_prefix):
                        self._cameras[camera_prefix].append(sensor_name)

        self._base_action_mode = "smooth"
        # for smooth base movement
        self.base_lin_vel_target = mn.Vector3(0, 0, 0)
        self.base_ang_vel_target = mn.Vector3(0, 0, 0)
        # \delta_v / (\delta_v * acc_ratio) = step to reach target velocity
        self.lin_acc_ratio = self.ang_acc_ratio = 1/5

    def step_base(self, err=1e-2):
        if self._base_action_mode == "smooth":
            def equal(a, b, err=1e-5):
                if abs(np.array(a).sum() - np.array(b).sum()) < err:
                    return True
            step = False
            # adjust lin and ang velocity, once reached the target, gradually slow down until stop
            if not equal(self.sim_obj.root_linear_velocity, self.base_lin_vel_target, err):
                self.sim_obj.root_linear_velocity += self.lin_acc_ratio * (self.base_lin_vel_target - self.sim_obj.root_linear_velocity)
            else:
                # the current target is 0 and the current velocity is close to 0, then just stop
                if np.array(self.base_lin_vel_target).sum() == 0:
                    self.sim_obj.root_linear_velocity = mn.Vector3(0, 0, 0)
                else:
                    self.base_lin_vel_target = mn.Vector3(0, 0, 0)

            if not equal(self.sim_obj.root_angular_velocity, self.base_ang_vel_target, err):
                self.sim_obj.root_angular_velocity += self.ang_acc_ratio * (self.base_ang_vel_target - self.sim_obj.root_angular_velocity)
            else:
                # the current target is 0 and the current velocity is close to 0, then just stop
                if np.array(self.base_ang_vel_target).sum() == 0:
                    self.sim_obj.root_angular_velocity = mn.Vector3(0, 0, 0)
                else:
                    self.base_ang_vel_target = mn.Vector3(0, 0, 0)


    def base_action(self, lin_vel, ang_vel):
        if self._base_action_mode == "teleport":
            return self.base_action_teleport(lin_vel, ang_vel, self._sim)
        elif self._base_action_mode == "smooth":
            return self.base_action_smooth(lin_vel, ang_vel)
        else:
            raise NotImplementedError

    def base_action_smooth(self, lin_vel, ang_vel):
        lin_vel *= 6
        ang_vel *= 6
        lin_vel = np.clip(lin_vel, -1, 1)
        ang_vel = np.clip(ang_vel, -1, 1)
        self.base_lin_vel_target = self.sim_obj.rotation.transform_vector(mn.Vector3(lin_vel, 0, 0))
        self.base_ang_vel_target = mn.Vector3(0, ang_vel, 0)

    def base_action_teleport(self, lin_vel, ang_vel):
        lin_vel = np.clip(lin_vel, -1, 1) * 12.0
        ang_vel = np.clip(ang_vel, -1, 1) * 12.0

        base_vel_ctrl = habitat_sim.physics.VelocityControl()
        base_vel_ctrl.controlling_lin_vel = True
        base_vel_ctrl.lin_vel_is_local = True
        base_vel_ctrl.controlling_ang_vel = True
        base_vel_ctrl.ang_vel_is_local = True
        base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            ctrl_freq = 60
            trans = self.sim_obj.transformation
            # 获取当前机器人的状态
            rigid_state = habitat_sim.RigidState(
                mn.Quaternion.from_matrix(trans.rotation()), trans.translation
            )

            # 根据设定好的速度，手动算出1/ctrl_freq时间后的目标状态
            target_rigid_state = base_vel_ctrl.integrate_transform(
                1 / ctrl_freq, rigid_state
            )
            # 和navmesh做计算，找到正确的终止位置
            end_pos = self._sim.step_filter(
                rigid_state.translation, target_rigid_state.translation
            )

            print(end_pos, target_rigid_state.translation, rigid_state.translation)

            # Offset the base
            end_pos -= self.params.base_offset

            self.sim_obj.translation = end_pos
            self.sim_obj.rotation = target_rigid_state.rotation

    def reconfigure(self) -> None:
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        if self.sim_obj is None or not self.sim_obj.is_alive:
            ao_mgr = self._sim.get_articulated_object_manager()
            self.sim_obj = ao_mgr.add_articulated_object_from_urdf(
                self.urdf_path,
                fixed_base=self._fixed_base,
                maintain_link_order=self._maintain_link_order,
            )
        # set correct gains for wheels
        if (
            hasattr(self.params, "wheel_joints")
            and self.params.wheel_joints is not None
        ):
            jms = JointMotorSettings(
                0,  # position_target
                self.params.wheel_mtr_pos_gain,  # position_gain
                0,  # velocity_target
                self.params.wheel_mtr_vel_gain,  # velocity_gain
                self.params.wheel_mtr_max_impulse,  # max_impulse
            )
            # pylint: disable=not-an-iterable
            for i in self.params.wheel_joints:
                self.sim_obj.update_joint_motor(self.joint_motors[i][0], jms)
        self._update_motor_settings_cache()

        # set correct gains for legs
        if (
            hasattr(self.params, "leg_joints")
            and self.params.leg_joints is not None
        ):
            jms = JointMotorSettings(
                0,  # position_target
                self.params.leg_mtr_pos_gain,  # position_gain
                0,  # velocity_target
                self.params.leg_mtr_vel_gain,  # velocity_gain
                self.params.leg_mtr_max_impulse,  # max_impulse
            )
            # pylint: disable=not-an-iterable
            for i in self.params.leg_joints:
                self.sim_obj.update_joint_motor(self.joint_motors[i][0], jms)
            self.leg_joint_pos = self.params.leg_init_params
        self._update_motor_settings_cache()

    def update(self) -> None:
        self.step_base()

    def reset(self) -> None:
        if (
            hasattr(self.params, "leg_joints")
            and self.params.leg_init_params is not None
        ):
            self.leg_joint_pos = self.params.leg_init_params
        self._update_motor_settings_cache()
        self.update()

    @property
    def base_pos(self):
        """Get the robot base ground position"""
        # via configured local offset from origin
        if self._base_type in ["mobile", "leg"]:
            return (
                self.sim_obj.translation
                + self.sim_obj.transformation.transform_vector(
                    self.params.base_offset
                )
            )
        else:
            raise NotImplementedError("The base type is not implemented.")

    @base_pos.setter
    def base_pos(self, position: mn.Vector3):
        """Set the robot base to a desired ground position (e.g. NavMesh point)"""
        # via configured local offset from origin.
        if self._base_type in ["mobile", "leg"]:
            if len(position) != 3:
                raise ValueError("Base position needs to be three dimensions")
            self.sim_obj.translation = (
                position
                - self.sim_obj.transformation.transform_vector(
                    self.params.base_offset
                )
            )
        else:
            raise NotImplementedError("The base type is not implemented.")

    @property
    def base_rot(self) -> float:
        return self.sim_obj.rotation.angle()

    @base_rot.setter
    def base_rot(self, rotation_y_rad: float):
        if self._base_type == "mobile" or self._base_type == "leg":
            self.sim_obj.rotation = mn.Quaternion.rotation(
                mn.Rad(rotation_y_rad), mn.Vector3(0, 1, 0)
            )
        else:
            raise NotImplementedError("The base type is not implemented.")

    @property
    def leg_motor_pos(self):
        """Get the current target of the leg joint motors."""
        if self._base_type == "leg":
            motor_targets = np.zeros(len(self.params.leg_init_params))
            for i, jidx in enumerate(self.params.leg_joints):
                motor_targets[i] = self._get_motor_pos(jidx)
            return motor_targets
        else:
            raise NotImplementedError(
                "There are no leg motors other than leg robots"
            )

    @leg_motor_pos.setter
    def leg_motor_pos(self, ctrl: List[float]) -> None:
        """Set the desired target of the leg joint motors."""
        if self._base_type == "leg":
            self._validate_ctrl_input(ctrl, self.params.leg_joints)
            for i, jidx in enumerate(self.params.leg_joints):
                self._set_motor_pos(jidx, ctrl[i])
        else:
            raise NotImplementedError(
                "There are no leg motors other than leg robots"
            )

    @property
    def leg_joint_pos(self):
        """Get the current arm joint positions."""
        if self._base_type == "leg":
            joint_pos_indices = self.joint_pos_indices
            leg_joints = self.params.leg_joints
            leg_pos_indices = [joint_pos_indices[x] for x in leg_joints]
            return [self.sim_obj.joint_positions[i] for i in leg_pos_indices]
        else:
            raise NotImplementedError(
                "There are no leg motors other than leg robots"
            )

    @leg_joint_pos.setter
    def leg_joint_pos(self, ctrl: List[float]):
        """Kinematically sets the arm joints and sets the motors to target."""
        if self._base_type == "leg":
            self._validate_ctrl_input(ctrl, self.params.leg_joints)

            joint_positions = self.sim_obj.joint_positions

            for i, jidx in enumerate(self.params.leg_joints):
                self._set_motor_pos(jidx, ctrl[i])
                joint_positions[self.joint_pos_indices[jidx]] = ctrl[i]
            self.sim_obj.joint_positions = joint_positions
        else:
            raise NotImplementedError(
                "There are no leg motors other than leg robots"
            )

    @property
    def base_transformation(self):
        return self.sim_obj.transformation

    def is_base_link(self, link_id: int) -> bool:
        return (
            self.sim_obj.get_link_name(link_id) in self.params.base_link_names
        )

    def update_base(self, rigid_state, target_rigid_state):
        end_pos = self._sim.step_filter(
            rigid_state.translation, target_rigid_state.translation
        )
        # Offset the end position
        end_pos -= self.params.base_offset
        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(), end_pos
        )
        self.sim_obj.transformation = target_trans

        if self._base_type == "leg":
            # Fix the leg joints
            self.leg_joint_pos = [0.0, 0.7, -1.5] * 4

    def _validate_ctrl_input(self, ctrl: List[float], joints: List[int]):
        """
        Raises an exception if the control input is NaN or does not match the
        joint dimensions.
        """
        if len(ctrl) != len(joints):
            raise ValueError(
                f"Control dimension does not match joint dimension: {len(ctrl)} vs {len(joints)}"
            )
        if np.any(np.isnan(ctrl)):
            raise ValueError("Control is NaN")
