from mani_skill.utils import sapien_utils
import gymnasium as gym
import cv2
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
import sapien.core as sapien
from mani_skill.utils import sapien_utils
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import time
import matplotlib.pyplot as plt
import mshab.envs
from mshab.envs.planner import plan_data_from_file
import sapien.core as sapien
import sapien.physx as physx
import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d
import os
from os.path import join
import sys
sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(), 'voxposer'))
from voxposer.utils_main import *
from envs.base_env import *



class MSHAB_ENV(BASE_ENV):
    def __init__(self, dt=1/60):
        super().__init__()
        task = "prepare_groceries" # "tidy_house", "", or "set_table"prepare_groceries
        subtask = "sequential"    # "sequential", "pick", "place", "open", "close"
                            # NOTE: sequential loads the full task, e.g pick -> place -> ...
                            #     while pick, place, etc only simulate a single subtask each episode
        split = "val"     # "train", "val"

        self.task = task
        REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
        print(REARRANGE_DIR)

        plan_data = plan_data_from_file(
            REARRANGE_DIR / "task_plans" / task / subtask / split / "cracker.json"
        )
        self.plan_data = plan_data
        spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

        self.env = gym.make(
            "SequentialTask-v0", #f"{subtask.capitalize()}SubtaskTrain-v0",
            # Simulation args
            num_envs=1,  # RCAD has 63 train scenes, so 252 envs -> 4 parallel envs reserved for each scene
            obs_mode="rgbd",
            sim_backend="gpu",
            robot_uids="fetch",
            control_mode="pd_joint_pos",
            # Rendering args
            reward_mode="normalized_dense",
            render_mode="rgb_array",
            shader_dir="minimal",
            # TimeLimit args
            max_episode_steps=200,
            # SequentialTask args
            task_plans=plan_data.plans,
            scene_builder_cls=plan_data.dataset,
            # SubtaskTrain args
            # spawn_data_fp=spawn_data_fp,
            # optional: additional env_kwargs
        )

        debug = False
        vis = False

        self.planner = PandaArmMotionPlanningSolver(
            self.env,
            debug=debug,
            vis=vis,
            base_pose=self.env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )


        segmentation_id_map = self.env.unwrapped.segmentation_id_map
        self.id2name = {}
        self.name2pose = {}
        for key in segmentation_id_map.keys():
            self.id2name[key] = segmentation_id_map[key].name
            self.name2pose[segmentation_id_map[key].name] = segmentation_id_map[key].pose
        self.name2id = {v: k for k, v in self.id2name.items()}

        self.dt = dt

        self.record_position = True
        self.frame = 'base'
    
    @property
    def base_transformation(self):
        pose = self.env.unwrapped.agent.robot.pose
        p = np.squeeze(pose.p.cpu().numpy())
        q = np.squeeze(pose.q.cpu().numpy())
        trans_matrix = self._to_transformation_matrix(p, quat2mat(q))
        return trans_matrix

    def update_cameras_view(self):
        self.update_scene()
        obs = self.env.unwrapped.render_all()
        obs = np.squeeze(obs.numpy())
        bgr_obs_show = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow('obs', bgr_obs_show)
        cv2.waitKey(int(self.dt * 1000))
    
    def get_ee_pose(self):
        obs = self.env.unwrapped.get_obs()
        tcp_pose = np.squeeze(obs['extra']['tcp_pose_wrt_base'].numpy())
        position, quat = tcp_pose[:3], tcp_pose[3:]
        return position, quat
    
    def get_ee_pos(self):
        position, quat = self.get_ee_pose()
        return position

    def get_ee_quat(self):
        position, quat = self.get_ee_pose()
        return quat
    
    def get_3d_point_cloud(self):
        obs = self.env.unwrapped.get_obs()
        rgbs, depths, masks, points = [], [], [], []
        for camera in obs['sensor_data'].keys():
            depth = np.squeeze(obs['sensor_data'][camera]['depth'].numpy()) / 1000.0
            rgb = np.squeeze(obs['sensor_data'][camera]['rgb'].numpy())
            param = obs['sensor_param'][camera]
            mask = np.squeeze(obs['sensor_data'][camera]['segmentation'].numpy())

            points_colors = self.get_color_points(rgb, depth, param, mask)

            point = points_colors[:, :3]
            point_mask = mask.reshape((-1, 1))
            point_colors = points_colors[:, 3:]
            rgbs.append(point_colors)
            points.append(point)
            masks.append(point_mask)
        rgbs = np.concatenate(rgbs, axis=0)
        masks = np.concatenate(masks, axis=0)
        points = np.concatenate(points, axis=0)

        if self.frame == 'base':
            temp = np.concatenate([points, np.ones((len(points), 1))], axis=1)
            points = (self.base_transformation.T @ temp.T).T[:, :3]

        return points, rgbs, masks
    
    def depth_to_pc(self, K, depth, rgb=None, mask=None):
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        x = (i - cx) * depth / fx
        y = (j - cy) * depth / fy
        z = depth

        
        points = np.vstack((x.flatten(), y.flatten(), z.flatten(), mask.flatten())).T

        flatten_rgb = rgb.reshape((-1, 3))
        points = np.hstack((points, flatten_rgb))
        return points

    def get_color_points(self, rgb, depth, cam_param, mask=None):
        intrinsic_matrix = np.squeeze(cam_param['intrinsic_cv'].numpy())

        color_points = self.depth_to_point_cloud(intrinsic_matrix, depth, rgb, mask)
        cam2world_gl = np.squeeze(cam_param['cam2world_gl'].numpy())
        color_points[:, :3] = self.transform_camera_to_world(color_points[:, :3], cam2world_gl)
        return color_points

    def depth_to_point_cloud(self, K, depth, rgb=None, mask=None):
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        
        x = (i - cx) * depth / fx
        y = (j - cy) * depth / fy
        z = depth
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        flatten_rgb = rgb.reshape((H*W, 3))
        points = np.hstack((points, flatten_rgb))
        return points
    '''
    Description: 
        transform point cloud from camera coordinicate to world coordinate
    '''
    def transform_camera_to_world(self, points, extri_mat):
        R = extri_mat[:3, :3]
        t = extri_mat[:3, 3]
        pcd_world = (R @ points.T).T - t
        rotation_y_180 = np.array([[-1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]])
        pcd_world = (rotation_y_180 @ pcd_world.T).T
        return pcd_world

    def apply_action(self, action, ignore_arm=False, ignore_ee=False):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply (xyz + wxyz)

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if not ignore_ee:
            ee_action = action[7]
            ee_action = ee_action > 0.5
            qpos = np.squeeze(self.env.agent.robot.get_qpos().numpy())[:7]
            if ee_action == 0:
                # step_action = np.concatenate([qpos, np.array([0])])
                _, reward, _, _, info = self.planner.close_gripper()
                # self.env.step(step_action)
                self.latest_action[-1] = 0
            else:
                # step_action = np.concatenate([qpos, np.array([1])])
                _, reward, _, _, info = self.planner.open_gripper()
                # self.env.step(step_action)
                self.latest_action[-1] = 1
            self.update_cameras_view()
        else:
            ee_action = self.latest_action[-1]

        if not ignore_arm:
            arm_quat = action[3:7]
            arm_pos = action[:3]
            pose = sapien.Pose(p=arm_pos, q=arm_quat)
            result = self.planner.move_to_pose_with_screw((pose), dry_run=True)
            if result == -1:
                print(f'result = -1, cannot move to target pose by IK planner')
            else:
                for pos in result['position']:
                    cur_pos = np.squeeze(self.env.agent.robot.get_qpos().numpy())[:7]
                    end_pos = pos
                    diff_pos = end_pos - cur_pos
                    for i in range(self.interp_num):
                        pos = cur_pos + diff_pos * (i + 1) / self.interp_num
                        pos = pos.tolist()
                        pos.append(ee_action)
                        pos = np.array(pos)
                        self.env.step(pos)
                        self.latest_action = pos
                        self.update_cameras_view()






















    def update_scene(self):
        if physx.is_gpu_enabled():
            self.env.scene._gpu_apply_all()
            self.env.scene.px.gpu_update_articulation_kinematics()
            self.env.scene.step()
            self.env.scene._gpu_fetch_all()
    
    def set_agent_pose(self, pose):
        self.env.agent.robot.set_pose(pose)
        self.update_scene()
        self.update_cameras_view()

    def compute_quaternion(self, xy1, xy2):
        yaw = np.arctan2(xy2[1] - xy1[1], xy2[0] - xy1[0])  # Compute yaw angle
        quaternion = transforms3d.euler.euler2quat(0, 0, yaw)  # Convert to quaternion (roll=0, pitch=0, yaw)
        return quaternion  # (w, x, y, z)
    
    def get_navigable_points(self, object_center, thresold=1):
        navigable_position = self.env.scene_builder.navigable_positions[0].vertices
        position_wrt_center = navigable_position - object_center[:2]
        dists = np.linalg.norm(position_wrt_center, ord=2, axis=1, keepdims=True)
        dists = dists.reshape(-1)
        criterion = dists < thresold
        env_navigable_positions = navigable_position[criterion]
        return env_navigable_positions
    
    def set_agent_navigable_point(self, object_center, thresold=1):
        env_navigable_positions = self.get_navigable_points(object_center, thresold)
        if len(env_navigable_positions) == 0:
            print(f'I cannot find navigable point in thresold {thresold}')
            return
        if not self.record_position:
            random_int = np.random.randint(0, len(env_navigable_positions))
            position = env_navigable_positions[random_int]
            quat = self.compute_quaternion(position, object_center[:2])
            pose = sapien.Pose(p=[
                                    position[0],
                                    position[1],
                                    0
                                ],
                                q=quat)
            self.set_agent_pose(pose)
            self.update_scene()
            self.update_cameras_view()
            return None, None
        else:
            for i in range(len(env_navigable_positions)):
                position = env_navigable_positions[i]
                quat = self.compute_quaternion(position, object_center[:2])
                pose = sapien.Pose(p=[
                                        position[0],
                                        position[1],
                                        0
                                    ],
                                    q=quat)
                self.set_agent_pose(pose)
                self.update_scene()
                self.update_cameras_view()
                flag = input('Do you want to save this position(t/f):')
                if flag == 't':
                    return pose.p.tolist(), pose.q.tolist()
            return None, None