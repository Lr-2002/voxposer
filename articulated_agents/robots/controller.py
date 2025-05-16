import copy
import os

import magnum as mn
import numpy as np
import pink
import pinocchio as pin
import qpsolvers
import scipy
from pink import solve_ik
from pink.utils import custom_configuration_vector
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2axangle

###
# Correct for all SE(3) transformation:
#   T_{B to A} = T_{A to B}^-1
#   T_{B to A} = T_{W to A} @ T_{B to W} (order matters!)
# Correct when T involves translation only (no rotation):
#   T_{B to A} * T {A to C} = T {A to C} * T_{B to A} = T_{B to C} (vector adding is commutative)
#   T_{D to A} = T_{B to A} * T_{C to A}, where AD = AB + AC (also vector adding)
###

class Controller(object):
    def __init__(self, robot, rotate_axis, pin_robot=None, end_effector_task=None, list_joint_names=None, ee_link_name=None, ee_offset=None):
        self.robot = robot
        self.rotate_axis = rotate_axis
        self.dof = len(self.rotate_axis)
        self._get_arm_transformation()
        self.gripper_state = 1 # 1 for open, 0 for close

        # For pink
        self.pin_robot = pin_robot
        self.end_effector_task = end_effector_task
        self.list_joint_names = list_joint_names
        self.ee_link_name = ee_link_name
        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"

        # Note: for ee_offset, it will be added to ee, assumed to be in canonical
        # coordinates, and:
        # 1) in ee_transformation, we apply ee_offset to obtain the offset-ed ee pose
        # 2) in pink and customized IK, we modify the target transformation by
        #    adding ee_offset, so the true target become offset-ed ee pose
        self.ee_offset = ee_offset

    @property
    def robot_joint_pos(self):
        # Note:
        # `arm_joint_pos` stores the readings of joint angles from the robot,
        # `arm_motor_pos` stores the target of joint motors.
        #
        # Ideally, aftering setting `arm_motor_pos` and calling `step_physics()`,
        # these two should become identical if there is no obstacle. However, I
        # have observed a small discrepancy between them. In teleoperation, this
        # will cause drifting of the robot arm even if there is no input (the
        # robot simply keeps chasing the "error").
        #
        # To remedy this, I choose to assume open-loop control -- suppose these
        # are always identical and compute the EE pose based off `arm_motor_pos`
        # instead of `arm_joint_pos`.
        return self.robot.arm_motor_pos
        # return self.robot.arm_joint_pos

    @property
    def base_transformation(self):
        # Note:
        # these two represents canonical and habitat coordinates respectively.
        # -**canonical**: `robot.base_transformation` is aligned with `sim.get_agent(0).state.sensor_state`
        # -**habitat**: robot.sim_obj.transformation` is aligned with `sim.get_link_scene_node().transformation`
        #
        # Since our customized IK/FK relies on `sim.get_link_scene_node().transformation`,
        # we have to stick on `robot.sim_obj.transformation` for consistency.
        #
        # While for pink IK/FK, both are fine, but in the `ee_tranformation`
        # function, we may need extra conversion when returning ee transformation
        # w.r.t. world if using habitat coordinate (see below).

        return self.robot.base_transformation
        # return self.robot.sim_obj.transformation

    def gripper_control(self, action):
        """
        0 -- close
        1 -- open
        """
        if action == 1:
            self.robot.open_gripper()
            self.gripper_state = 1
        elif action == 0:
            self.robot.close_gripper()
            self.gripper_state = 0
        else:
            raise NotImplementedError

    def ee_transformation(self, frame='base', impl='pink', base_coordinate='canonical'):
        # frame: which frame to use
        # impl: customized FK or pink
        # base_coordinate: only used when frame=='base':
        #    canonical: x -- forward, y -- left, z -- up
        #    habitat: x -- forward, y -- up, z -- right
        ret = None
        if impl == 'pink':
            list_joint_names = self.list_joint_names
            joint_targets = {k:v for k, v in zip(list_joint_names, self.robot_joint_pos)}
            qpos = custom_configuration_vector(
                self.pin_robot,
                **joint_targets
            )
            configuration = pink.Configuration(self.pin_robot.model, self.pin_robot.data, qpos)
            ee_transformation = mn.Matrix4(np.array(configuration.get_transform_frame_to_world(self.ee_link_name)))
            if frame == 'base':
                if base_coordinate == 'habitat':
                    # convert pink canonical coordinate to habitat
                    # (y_hab = z_pink, z_hab = -y_pink)
                    tmp = np.array([[1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, 0, 1]])
                    ee_transformation = mn.Matrix4(tmp.astype(np.float32)) @ ee_transformation
                ret = ee_transformation
            elif frame == 'world':
                # base to world
                # since now by default, `self.base_transformation` uses `robot.base_transformation`,
                # which is under canonical coordinate already, there is no need to
                # first conver the ee transformation w.r.t. base under pink canonical
                # to under habitat before converting to world.
                ee_transformation = self.base_transformation @ ee_transformation
                ret = ee_transformation
        elif impl == 'customized':
            if frame == 'base':
                ee_transformation = self.gripper_to_X('base')
                if base_coordinate == 'canonical':
                    # convert habitat coordinate to pink canonical
                    # (y_pink = -z_hab, z_pink = z_hab)
                    tmp = np.array([[1, 0, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])
                    ee_transformation = mn.Matrix4(tmp.astype(np.float32)) @ ee_transformation
                ret = ee_transformation
            elif frame == 'world':
                ret =  self.gripper_to_world()

        # apply ee offset
        if self.ee_offset is not None:
            offset_transformation = np.eye(4)
            offset_transformation[:3, 3] = self.ee_offset
            ret = ret @ mn.Matrix4(offset_transformation)
        return ret

    def _delta_to_target(self, delta, frame='intuitive', impl='pink'):
        ee_transformation = self.ee_transformation(frame='base', impl=impl, base_coordinate='canonical')

        delta_transformation = np.eye(4)
        delta_transformation[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix()
        delta_transformation[:3, 3] = delta[:3]
        delta_transformation = mn.Matrix4(delta_transformation)

        if frame == 'intuitive':
            target_transformation = delta_transformation @ ee_transformation
            target_transformation.translation = ee_transformation.translation + delta_transformation.translation
        elif frame == 'base':
            target_transformation = delta_transformation @ ee_transformation
        elif frame == 'self':
            target_transformation = ee_transformation @ delta_transformation
        else:
            raise NotImplementedError

        return target_transformation

    # aka, inverse kinematics
    def _target_to_joint(self, target_transformation, local=True, impl='pink', ignore_rot=False, verbose=True):
        def from_pin_joints(pin_robot, pin_joint, list_joint_names):
            ret = []
            for name in list_joint_names:
                joint_id = pin_robot.model.getJointId(name)
                ret.append(pin_joint[joint_id-1])
            return np.array(ret)

        if self.ee_offset is not None:
            # offset the target transformation to reflect the ee offset
            offset_transformation = np.eye(4)
            offset_transformation[:3, 3] = self.ee_offset
            target_transformation = target_transformation @ mn.Matrix4(offset_transformation).inverted()

        if impl == 'pink':
            if local:
                joint_targets = {k:v for k, v in zip(self.list_joint_names, self.robot_joint_pos)}
            else:
                joint_targets = {k:0 for k in self.list_joint_names}
            qpos = custom_configuration_vector(
                self.pin_robot,
                **joint_targets
            )
            configuration = pink.Configuration(self.pin_robot.model, self.pin_robot.data, qpos)

            pin_target = pin.SE3()
            pin_target.translation = np.array(target_transformation.translation)
            pin_target.rotation = np.array(target_transformation.rotation())
            self.end_effector_task.set_target(pin_target)

            cost_bk = copy.deepcopy(self.end_effector_task.cost)
            if ignore_rot:
                self.end_effector_task.set_orientation_cost([0,0,0])
            dt = 0.1
            ret = solve_ik(configuration, [self.end_effector_task], dt, solver=self.solver)
            self.end_effector_task.cost = cost_bk

            if len(ret) == 2:
                delta_v, result = ret
                error = result.obj
            else:
                delta_v = ret
                error = 0
            delta_q = delta_v * dt
            target_qpos = from_pin_joints(self.pin_robot, delta_q, self.list_joint_names) + self.robot_joint_pos
        elif impl == 'customized':
            def loss(theta):
                ee_transformation = self.gripper_to_X('base', theta)
                ee_translation = np.array(ee_transformation.translation)
                ee_rotmat = Rotation.from_matrix(ee_transformation.rotation())
                target_translation = np.array(target_transformation.translation)
                target_rotmat = Rotation.from_matrix(target_transformation.rotation())

                diff_translation = np.linalg.norm(target_translation - ee_translation) / 1.0
                if ignore_rot:
                    diff_rot = 0
                else:
                    diff_rot = np.linalg.norm(Rotation.from_matrix(ee_rotmat.as_matrix().T @ target_rotmat.as_matrix()).as_rotvec()) / np.pi

                return diff_translation + diff_rot

            if local:
                init = list(self.robot_joint_pos)
            else:
                init = [0, 0, 0, 0, 0, 0, 0]
            limit = [(i[0], i[1]) for i in np.vstack(self.robot.arm_joint_limits).T]
            res = scipy.optimize.minimize(loss, init, method="Powell",
                                            bounds=limit, options={"maxiter": 10000})
            error = res.fun
            target_qpos = res.x
        else:
            raise NotImplementedError

        if verbose:
            print(f"Error: {error}")
        return target_qpos

    def _openx_raw_action_process(self, raw_action, impl='customized', action_scale=1.0):
        """
        **Unormalized** openx raw action (from RT/Octo/OpenVLA) processing
        For action de-normaliztion in open-x, see https://github.com/openvla/openvla/blob/main/prismatic/models/vlas/openvla.py#L36-L38
        ref:
        step 1: https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/policies/octo/octo_model.py#L130-L242
        step 2: https://github.com/simpler-env/ManiSkill2_real2sim/blob/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37/mani_skill2_real2sim/envs/sapien_env.py#L547-L573
        step 3: https://github.com/simpler-env/ManiSkill2_real2sim/blob/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37/mani_skill2_real2sim/agents/controllers/pd_ee_pose.py#L97-L121

        Input: open-x raw action (delta translation, delta euler angles, gripper (1 -- open, 0 -- close)) in canonical coordinates
        Output: delta translation, delta rotvec, and gripper (axis-angle rotation)
        """
        action_translation = raw_action[:3] * action_scale
        roll, pitch, yaw = np.asarray(raw_action[3:6], dtype=np.float64)

        # pink coordinate is canonical as open-x, therefore no need to rotate
        if impl == 'customized':
            # hab3 y == open-x z
            # hab3 z == open-x -y
            action_translation[1], action_translation[2] = action_translation[2], -action_translation[1]
            pitch, yaw = yaw, -pitch

        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_rotvec = action_rotation_ax * action_rotation_angle * action_scale

        return np.concatenate([action_translation, action_rotation_rotvec, [raw_action[6]]])

    def manipulate_by_6dof_delta(self, delta, local=True, frame='base', impl='customized', ignore_rot=False, verbose=True):
        # Note on coordinate system:
        #   - canonical: x-forward, y-left, z-up
        #   - habitat: x-forward, y-up, z-right
        #   `pink` assumes everything in canonical coordinates (open-x, issacsim)
        #   `customized` assumes everything in habitat coordinates

        # Input: (delta) [x, y, z, rotvec_x, rotvec_y, rotvec_z]
        # Output: qpos (joint angles)
        if impl == 'pink':
            assert self.pin_robot is not None
        target_transformation = self._delta_to_target(delta, frame, impl)
        return self._target_to_joint(target_transformation, local, impl, ignore_rot, verbose)

    def manipulate_by_6dof_target(self, target, local=True, impl='customized', ignore_rot=False, verbose=True):
        # Note on coordinate system:
        #   - canonical: x-forward, y-left, z-up
        #   - habitat: x-forward, y-up, z-right
        #   `pink` assumes everything in canonical coordinates (open-x, issacsim)
        #   `customized` assumes everything in habitat coordinates

        # Input: (target) [x, y, z, rotvec_x, rotvec_y, rotvec_z] relative to the robot base
        # Output: qpos (joint angles)
        target_transformation = np.eye(4)
        target_transformation[:3, :3] = Rotation.from_rotvec(target[3:]).as_matrix()
        target_transformation[:3, 3] = target[:3]
        target_transformation = mn.Matrix4(target_transformation)
        return self.manipulate_by_6dof_target_transformation(target_transformation, local, impl, ignore_rot, verbose)

    def manipulate_by_6dof_target_transformation(self, target_transformation, local=True, impl='customized', ignore_rot=False, verbose=True):
        # Note on coordinate system:
        #   - canonical: x-forward, y-left, z-up
        #   - habitat: x-forward, y-up, z-right
        #   `pink` assumes everything in canonical coordinates (open-x, issacsim)
        #   `customized` assumes everything in habitat coordinates

        # Input: 4x4 target transformation matrix relative to the robot base
        # Output: qpos (joint angles)
        if impl == 'pink':
            assert self.pin_robot is not None
        return self._target_to_joint(target_transformation, local, impl, ignore_rot, verbose)

    # Below are customized FK implementation
    def _get_arm_transformation(self):
        """
        Calculate the transformation matrix for each joint on robot arm.
        """
        # this the transformation relative to the world
        # TODO: in customized FK/IK, we have to use sim_obj.transformation as it
        # uses habitat coordinate and aligns with sim_obj.get_link_scene_node().transformation,
        # which is widely used in customized FK/IK
        robot_trans = self.robot.sim_obj.transformation
        parent_trans = [robot_trans.inverted()]
        self.base_trans = []
        for i, arm_id in enumerate(self.robot.params.arm_joints):
            rad = self.robot._get_motor_pos(arm_id)
            # the transformation is relative to the world
            link_trans = self.robot.sim_obj.get_link_scene_node(arm_id).transformation
            # offset the initial rotation of each joint
            base_matrix = mn.Matrix4.rotation(mn.Rad(-rad), self.rotate_axis[i]) @ parent_trans[i] @ link_trans

            parent_trans.append(link_trans.inverted())
            self.base_trans.append(base_matrix)

        gripper_trans = self.robot.sim_obj.get_link_scene_node(self.robot.params.ee_links[0]).transformation
        self.gripper2link = parent_trans[-1] @ gripper_trans

    def gripper_to_X(self, X, thetas=None):
        # Note: only to use this in customized FK/IK!!!
        cur_ee2w = self.gripper_to_world(thetas)
        if X == 'base':
            # base2w
            # TODO: in customized FK/IK, we have to use sim_obj.transformation as it
            # uses habitat coordinate and aligns with sim_obj.get_link_scene_node().transformation,
            # which is widely used in customized FK/IK
            x_pose = self.robot.sim_obj.transformation
        elif X == 'self':
            # ee2w
            x_pose = cur_ee2w
        elif X == 'world':
            return cur_ee2w

        return x_pose.inverted() @ cur_ee2w

    def gripper_to_world(self, thetas=None):
        """
        Get the gripper-to-world transformation matrix by setting rotation angles of each joint.
        Thetas should be rad angles and have a length of 7.
        """
        if thetas is None:
            thetas = self.robot_joint_pos
        # TODO: in customized FK/IK, we have to use sim_obj.transformation as it
        # uses habitat coordinate and aligns with sim_obj.get_link_scene_node().transformation,
        # which is widely used in customized FK/IK
        trans = self.robot.sim_obj.transformation
        for i in range(len(thetas)):
            trans = trans @ mn.Matrix4.rotation(mn.Rad(thetas[i]), self.rotate_axis[i]) @ self.base_trans[i]
        trans = trans @ self.gripper2link
        return trans