from omegaconf import OmegaConf
import habitat_sim
import magnum as mn
import os
import sys
import numpy as np
import open3d as o3d
import cv2
from os.path import join

sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(), 'voxposer'))
from envs.base_env import *

class Habitat_ENV(BASE_ENV):
    def __init__(self, habitat_env, agent, controller, frame='base', 
                 fps=60, margin_array=np.array([1.2, 2.0, 1.2])):
        super().__init__()
        self._env = habitat_env
        self._agent = agent
        self._controller = controller
        self.frame = frame
        self.dt = 1/fps
        self.camera_names = list(self._env._simulator.get_agent(0).state.sensor_states.keys())
        self.get_camera_triplets_mats()
        self.workspace_margin = margin_array
        self.interpolate_seconds = 1                    # arm controller interpolate time
        self.detach_before_open = True                  # fake release gripper
        self.attach_grasped_objects = False             # fake grasp
        self.impl = 'pink'
        
    def set_env_resolution(self, set_env_resolution):
        pass
    def get_camera_triplets_mats(self):
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        print(f'camera names are {self.camera_names}')
        # aggregate all semantic and depth cameras
        self.camera_triplets_mats = []
        for cam_name in self.camera_names:
            if 'semantic' in cam_name:
                rgb = cam_name.split('+')[0]
                semantic = cam_name
                depth = cam_name.replace('semantic', 'depth')
                assert rgb in self.camera_names and depth in self.camera_names
                cam = self._env._simulator.get_agent(0)._sensors[rgb]
                H, W = self._env._simulator.get_sensor_observations()[rgb].shape[:2]
                hfov = float(cam.hfov) * np.pi / 180
                aspect_ratio = W / H
                cam_K = np.array([
                    [1 / np.tan(hfov / 2.), 0., 0., 0.],
                    [0., 1 / np.tan(hfov / 2.) * aspect_ratio, 0., 0.],
                    [0., 0.,  1, 0],
                    [0., 0., 0, 1]])
                ret = self._env._simulator.get_agent(0).state.sensor_states[rgb]
                pos = ret.position
                rot = np.array([ret.rotation.x, ret.rotation.y,
                                ret.rotation.z, ret.rotation.w])
                cam_W = self._to_transformation_matrix(pos, rot)
                lookat = cam_W[:3, :3] @ forward_vector
                self.lookat_vectors[rgb] = self.normalize_vector(lookat)
                self.camera_triplets_mats.append(
                    (rgb, depth, semantic, cam_K, cam_W))
        self.rgb_camera_names = self.lookat_vectors.keys()

    def load_task(self):
        self._reset_task_variables()
        self.reset()

    def reset(self):
        self.reset_to_default_pose()
        self._reset_task_variables()
        self.init_obs = self._env._simulator.get_sensor_observations()
        self.latest_obs = self.init_obs
        self.update_cameras_view()
    
    def _reset_task_variables(self):
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        self.grasped_obj_ids = None
        # scene-specific helper variables
        self.arm_mask_ids = None  # UNUSED
        self.gripper_mask_ids = None  # UNUSED
        self.robot_mask_ids = 0
        self.obj_mask_ids = None  # UNUSED
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {}  # any node id -> first_generation name

    def reset_to_default_pose(self):
        target_qpos = self._agent.params.arm_init_params
        curs = np.array(self._agent.arm_motor_pos)
        diff = target_qpos - curs

        T = int(self.interpolate_seconds / self.dt)
        for i in range(T):
            self._agent.arm_motor_pos = diff * (i + 1) / T + curs
            self._env._simulator.step_physics(self.dt)
            self._agent.update()
            self.update_cameras_view()
        self.update_cameras_view()

    def apply_action(self, action, ignore_arm=False, ignore_ee=False, T=None):
        arm_action = action[:7]
        ee_action = action[7:]
        self.latest_action = action
        
        # arm control
        if not ignore_arm:
            assert self.workspace_bounds_min[0] < arm_action[0] < self.workspace_bounds_max[0]
            assert self.workspace_bounds_min[1] < arm_action[1] < self.workspace_bounds_max[1]
            assert self.workspace_bounds_min[2] < arm_action[2] < self.workspace_bounds_max[2]

            if self.frame == 'world':
                target_transformation = self._to_transformation_matrix(arm_action[:3], arm_action[3:])
                target_transformation_at_base = self.base_transformation.inverted() @ target_transformation
            elif self.frame == 'base':
                target_transformation_at_base = mn.Matrix4(self._to_transformation_matrix(arm_action[:3], arm_action[3:]))
            else:
                raise ValueError(f"Unsupported frame: {self.frame}")
            last_qpos = None
            while True:
                target_qpos = self._controller.manipulate_by_6dof_target_transformation(target_transformation_at_base, impl='pink', ignore_rot=False)
                if last_qpos is not None and np.sum((target_qpos - last_qpos) * (target_qpos - last_qpos)) < 5e-3:
                    break
                last_qpos = target_qpos
                curs = np.array(self._agent.arm_motor_pos)
                diff = target_qpos - curs

                if T is None:
                    cur_T = 15
                else:
                    cur_T = int(T / self.dt)
                print(f'time steps is {T}')
                for i in range(cur_T):
                    self._agent.arm_motor_pos = diff * (i + 1) / cur_T + curs
                    self.update_cameras_view()

        # gripper control
        if not ignore_ee:
            if 0.0 > ee_action[0] > 1.0:
                raise ValueError(
                    'Gripper action expected to be within 0 and 1.')
            ee_action = float(ee_action[0] > 0.5)

            if self._controller.gripper_state != ee_action:      
                if self.detach_before_open and ee_action == 1.0:         # fake release gripper
                    self._agent.desnap()
                self._controller.gripper_control(ee_action)
                for _ in range(10):
                    # self._env._simulator.step_physics(self.dt)
                    # self._agent.update()
                    self.update_cameras_view()
                if ee_action == 0.0 and self.attach_grasped_objects:             # fake grasp
                    # attach object that made contact with the gripper
                    grasp_object_id = None
                    for coll in self._env._simulator.get_physics_contact_points():
                        if coll.object_id_a == self._agent.sim_obj.object_id and coll.object_id_b not in [0, -1]:
                            grasp_object_id = coll.object_id_b
                        elif coll.object_id_b == self._agent.sim_obj.object_id and coll.object_id_a not in [0, -1]:
                            grasp_object_id = coll.object_id_a
                    if grasp_object_id is not None:
                        self._agent.snap_by_id(grasp_object_id)

        self.update_cameras_view()
    
    def move_to_pose(self, pose):
        if self.latest_action is None:
            action = np.concatenate([pose, [1.0]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action, ignore_ee=True)

    def open_gripper(self):
        action = np.concatenate([[0.0] * 7, [1.0]])
        self.latest_action = action
        return self.apply_action(action, ignore_arm=True)
    
    def close_gripper(self):
        action = np.concatenate([[0.0] * 7, [0.0]])
        self.latest_action = action
        return self.apply_action(action, ignore_arm=True)

    def set_gripper_state(self, gripper_state):
        action = np.concatenate(
            [[0.0] * 7, [gripper_state]])
        return self.apply_action(action, ignore_arm=True)
    
    def update_cameras_view(self):
        self._agent.update()
        self._env._simulator.step_physics(self.dt)
        obs = self._env._simulator.get_sensor_observations()
        obs_show = []
        for rgb_camera in self.rgb_camera_names:
            obs_show.append(obs[rgb_camera])
        obs_show = np.concatenate(obs_show, axis=1)
        bgr_obs_show = cv2.cvtColor(obs_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("obs", bgr_obs_show)
        cv2.waitKey(int(self.dt * 1000))

    def depth_to_pc(self, cam_K, cam_W, depth, color=None, downsample_rate=1, frame=None):
        # depth: (H, W)
        # color: (H, W, 3)
        # Note: W and H has to be strict here.
        #
        # return: (N, 3) or (N, 6)
        H, W = depth.shape
        xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
        depth = depth[None, ...]
        if color is not None:
            color = color.transpose(2, 0, 1)
        xs = xs[None, ...]
        ys = ys[None, ...]
        # unproject
        xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        if color is not None:
            color = color.reshape(3, -1)
        xy_c = np.matmul(np.linalg.inv(cam_K), xys)

        if frame is None:
            frame = self.frame
        if frame == 'world':
            pcd = np.matmul(cam_W, xy_c).T[:, :3]
        elif frame == 'base':
            pcd = np.matmul(self.base_transformation.inverted(), np.matmul(cam_W, xy_c)).T[:, :3]
        else:
            raise ValueError(f"Unsupported frame: {frame}")
        if color is not None:
            pcd = np.hstack((pcd, color.T))
        pcd = pcd[::downsample_rate]
        return pcd
    
    '''
    Description: 
        without all filters, get raw point cloud & color matrix and mask matrix
    '''
    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        obs = self._env._simulator.get_sensor_observations()
        points, colors, masks = [], [], []
        for tmp in self.camera_triplets_mats:
            cam_rgb, cam_depth, cam_semantic, cam_K, cam_W = tmp
            points.append(self.depth_to_pc(cam_K, cam_W, obs[cam_depth], frame='world'))
            colors.append(obs[cam_rgb])
            masks.append(obs[cam_semantic].reshape(-1))

        if self.frame == 'base':
            points_base = []
            for point in points:
                temp_point = np.concatenate([point, np.ones((len(point), 1))], axis=-1).T
                point_base = np.matmul(self.base_transformation.inverted(), temp_point).T[:, :3]
                points_base.append(point_base)
            points = points_base

        return points, colors, masks
    
    def get_ee_pose(self):
        ee_transformation = self._controller.ee_transformation(frame=self.frame, impl=self.impl, base_coordinate='canonical')
        pos = np.array(ee_transformation.translation)
        rot_xyzw = Rotation.from_matrix(ee_transformation.rotation()).as_quat()
        rot_wxyz = np.concatenate([rot_xyzw[-1:], rot_xyzw[:-1]])
        return np.concatenate([pos, rot_wxyz])
    
    def get_ee_pos(self):
        return self.get_ee_pose()[:3]
    
    def get_ee_quat(self):
        return self.get_ee_pose()[3:]
    
    def get_last_gripper_action(self):
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return [1.0]
    def o3d_to_grasp(self, rot):
        return rot
    @property
    def workspace_bounds_min(self):
        # bounding box centered at the robot origin
        if self.frame == 'world':
            agent_pos = np.array(self._agent.base_pos)
        elif self.frame == 'base':
            agent_pos = np.zeros(3)
        else:
            raise ValueError(f"Unsupported frame: {self.frame}")
        return agent_pos - self.workspace_margin
    
    @property
    def workspace_bounds_max(self):
        # bounding box centered at the robot origin
        if self.frame == 'world':
            agent_pos = np.array(self._agent.base_pos)
        elif self.frame == 'base':
            agent_pos = np.zeros(3)
        else:
            raise ValueError(f"Unsupported frame: {self.frame}")
        return agent_pos + self.workspace_margin
    
    @property
    def base_transformation(self):
        return self._agent.base_transformation
        
        
