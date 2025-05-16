from mani_skill.utils import sapien_utils
import pdb
from mani_skill.utils.wrappers import VLARecorderWrapper
import tqdm
import gymnasium as gym
import cv2
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
import sapien.core as sapien
from mani_skill.utils import sapien_utils
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

from scipy.spatial.transform import Rotation as R
import os
from os.path import join
import sys
from gymnasium.wrappers import TimeLimit

sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(), "voxposer"))
from voxposer.utils_main import *
from envs.base_env import *

from memory_profiler import profile


class TqdmWrapper(gym.Wrapper):
    """
    为 ManiSkill 环境添加 tqdm 进度条的包装器。

    Args:
        env: 要包装的 Gymnasium 环境。
        max_episode_steps: 每个 episode 的最大步数。
        desc: 进度条的描述。
    """

    def __init__(self, env, max_episode_steps, desc="Episode Progress"):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.desc = desc
        self.pbar = None

    def reset(self, **kwargs):
        # 重置环境
        obs, info = self.env.reset(**kwargs)

        # 如果有旧的进度条，先关闭
        if self.pbar is not None:
            self.pbar.close()

        # 初始化新的 tqdm 进度条
        self.pbar = tqdm.tqdm(total=self.max_episode_steps, desc=self.desc)

        return obs, info

    def step(self, action):
        # 执行 step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 更新进度条
        if self.pbar is not None:
            self.pbar.update(1)

        # 如果 episode 结束，关闭进度条
        if terminated or truncated:
            # pdb.set_trace()
            self.close()

        return obs, reward, terminated, truncated, info

    def close(self):
        # 关闭进度条（如果存在）
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        # 关闭环境
        super().close()


class ManiSkill_Env(BASE_ENV):
    def __init__(
        self,
        task_name="PickCube-v1",
        obs_mode="rgb+depth+segmentation",
        dt=1 / 60,
        output_dir="./evaluation/voxposer_evaluation_0423/",
        model_class="Voxposer",
    ):
        super().__init__()
        self.env = gym.make(
            task_name,  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            num_envs=1,
            obs_mode=obs_mode,  # there is also "state_dict", "rgbd", ...
            control_mode="pd_joint_pos",  # there is also "", ...
            render_mode="rgb_array",
            marker_collision=True,
        )
        term_steps = 2000
        self.env = TimeLimit(self.env, max_episode_steps=term_steps)
        self.env = VLARecorderWrapper(
            self.env,
            output_dir=output_dir,
            model_class=model_class,
            model_path="None",
            save_trajectory=False,
        )

        self.env = TqdmWrapper(self.env, max_episode_steps=term_steps)
        self.done = False
        self.obs = None
        # sele
        self.debug = False
        self.vis = False
        self.limit = 1
        self.planner = PandaArmMotionPlanningSolver(
            self.env,
            debug=self.debug,
            vis=self.vis,
            base_pose=self.env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=True,
            print_env_info=self.debug,
            joint_acc_limits=self.limit,
            joint_vel_limits=self.limit,
        )
        self.dt = dt
        self.interp_num = 2

        # get semantic id2name and name2id maps, get robot id on the same time
        segmentation_id_map = self.env.unwrapped.segmentation_id_map
        # breakpoint()
        self.id2name = {}
        for key in segmentation_id_map.keys():
            self.id2name[key] = segmentation_id_map[key].name
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.robot_mask_ids = []
        for key in self.name2id.keys():
            if key.split("_")[0] == "panda":
                self.robot_mask_ids.append(self.name2id[key])

        self.base_position = np.squeeze(self.env.unwrapped.agent.robot.pose.p.numpy())
        self.bbox = np.array([1.0, 0.8, 0.5])
        self.frame = "world"

    @property
    def workspace_bounds_max(self):
        return self.base_position + self.bbox

    @property
    def workspace_bounds_min(self):
        return self.base_position - self.bbox

    def update_cameras_view(self):
        obs = self.env.unwrapped.render_sensors()
        # obs = self.obs
        obs = np.squeeze(obs.numpy())
        # print(f"\033[91m end-effector position is {self.get_ee_pos()}  \033[0m")
        bgr_obs_show = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow("obs", bgr_obs_show)
        cv2.waitKey(int(self.dt * 1000))

    # def load_task(self, task_name, obs_mode="rgb+depth+segmentation"):
    #     self.env = gym.make(
    #         task_name,  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    #         num_envs=1,
    #         obs_mode=obs_mode,  # there is also "state_dict", "rgbd", ...
    #         control_mode="pd_joint_delta_pos",  # there is also "pd_joint_delta_pos", ...
    #         render_mode="rgb_array",
    #     )
    #     import time
    #
    #     for i in range(20):
    #         self.env.step()
    #         print("----")
    #         time.sleep(0.01)
    #
    def set_env_resolution(self, set_env_resolution):
        pass

    def draw_pc(self, pc_arr):
        draw_pc(pc_arr)

    def reset(self):
        obs, info = self.env.reset()
        self.planner = PandaArmMotionPlanningSolver(
            self.env,
            debug=self.debug,
            vis=self.vis,
            base_pose=self.env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=self.vis,
            print_env_info=self.debug,
            joint_acc_limits=self.limit,
            joint_vel_limits=self.limit,
        )

        self.done = False
        #
        # breakpoint()
        init_pose = obs["agent"]["qpos"].cpu().numpy()[:, :-1]
        for i in range(10):
            # print(obs)
            # step_action = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
            # print("maniskill reset ")
            step_action = init_pose
            obs, _, _, _, _ = self.env.step(step_action)
        # if 'description' in info.keys():
        #     self.description = info["description"]
        # else:
        #     self.description =
        self.obs = obs
        pc, pc1, pc2 = self.get_3d_point_cloud()
        self.draw_pc(pc)
        self.init_qpos = np.squeeze(self.env.agent.robot.get_qpos().numpy())[
            :7
        ]  # get each motor pos of initialize pose
        self.init_qvel = np.squeeze(self.env.agent.robot.get_qvel().numpy())[:7]
        self.init_pose, self.init_quat = self.get_ee_pose()
        obs = self.env.unwrapped.get_obs()
        self.latest_obs = obs
        self.latest_action = np.concatenate(
            [self.init_pose, self.init_quat, np.array([1])]
        )
        self.update_cameras_view()

    def reset_to_default_pose(self):
        ee_action = self.latest_action[-1:]
        action = np.concatenate([self.init_pose, self.init_quat, ee_action])
        self.apply_action(action)

    def _reset_task_variables(self):
        self.init_qpos = np.squeeze(self.env.agent.robot.get_qpos().numpy())[
            :7
        ]  # get each motor pos of initialize pose, contain finger pos
        self.init_qvel = np.squeeze(self.env.agent.robot.get_qvel().numpy())[:7]
        self.init_pose, self.init_quat = self.get_ee_pose()
        obs = self.env.unwrapped.get_obs()
        self.latest_obs = obs
        self.latest_action = np.concatenate(
            [self.init_pose, self.init_quat, np.array([1])]
        )
        self.update_cameras_view()

    @profile
    def apply_action_ik(self, action, ignore_arm=False, ignore_ee=False):
        done = False
        trunc = False
        if not ignore_ee:
            ee_action = action[7]
            ee_action = ee_action > 0.5
            step_action = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
            if ee_action == 0:
                step_action[-1] = -1
            else:
                step_action[-1] = 1
            obs, reward, terminated, truncated, info = self.env.step(step_action)
            self.obs = obs
            self.update_cameras_view()
            self.latest_action[-1] = action[-1]
            done = max(done, terminated)
            trunc = max(trunc, truncated)

        if not ignore_arm:
            arm_quat = action[3:7]
            arm_pos = action[:3]
            target_rpy = R.from_matrix(quat2mat(arm_quat)).as_euler(
                "xyz", degrees=False
            )
            target_pos = arm_pos
            step_action = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
            while True:
                step_action[:3] = target_pos
                step_action[3:6] = target_rpy
                cur_pos = self.get_ee_pos()
                cur_quat = self.get_ee_quat()
                cur_rpy = R.from_matrix(quat2mat(cur_quat)).as_euler(
                    "xyz", degrees=False
                )
                differ_pos = target_pos - cur_pos
                differ_rpy = target_rpy - cur_rpy
                for i in range(len(differ_rpy)):
                    if differ_rpy[i] > np.pi:
                        differ_rpy[i] -= 2 * np.pi
                    if differ_rpy[i] < -np.pi:
                        differ_rpy[i] += 2 * np.pi

                if (
                    np.sum(np.abs(differ_pos) < 2e-2) == 3
                    and np.sum(np.abs(differ_rpy) < 5e-2) == 3
                ):
                    break

                step_action[:3] = differ_pos
                step_action[3:6] = -differ_rpy
                obs, reward, terminated, truncated, info = self.env.step(step_action)
                self.obs = obs
                done = max(done, terminated)
                trunc = max(trunc, truncated)

                self.update_cameras_view()
                break
            self.update_cameras_view()
            self.latest_action[:7] = action[:7]

    def apply_action(self, action, ignore_arm=False, ignore_ee=False):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply (xyz + wxyz)

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.done:
            return True
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
        print("the arm quat and pose in apply_action", action)
        if not ignore_arm:
            arm_quat = action[3:7]
            arm_pos = action[:3]
            pose = sapien.Pose(p=arm_pos, q=arm_quat)
            result = self.planner.move_to_pose_with_screw((pose), dry_run=True)
            if result == -1:
                print(f"result = -1, cannot move to target pose by IK planner")
            else:
                for pos in result["position"]:
                    if self.done:
                        break
                    cur_pos = np.squeeze(self.env.agent.robot.get_qpos().numpy())[:7]
                    end_pos = pos
                    diff_pos = end_pos - cur_pos
                    for i in range(self.interp_num):
                        pos = cur_pos + diff_pos * (i + 1) / self.interp_num
                        pos = pos.tolist()
                        pos.append(ee_action)
                        pos = np.array(pos)
                        obs, reward, terminated, truncated, info = self.env.step(pos)
                        self.obs = obs
                        self.latest_action = pos
                        self.update_cameras_view()
                        # self.done = self.done or (terminated or truncated)
                        # if self.done:
                        #     self.env.close()

    def open_gripper(self):
        action = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        self.apply_action(action, ignore_arm=True)

    def close_gripper(self):
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.apply_action(action, ignore_arm=True)

    def move_to_pose(self, pose):
        action = pose
        self.apply_action(action, ignore_ee=True)

    def set_gripper_state(self, state):
        action = np.array([0, 0, 0, 0, 0, 0, 0, state])
        self.apply_action(action, ignore_arm=True)

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        points, colors, masks = [], [], []
        point, color, mask = self.get_3d_point_cloud()
        # breakpoint()

        points.append(point)
        colors.append(color)
        masks.append(mask)
        return points, colors, masks

    def get_ee_pose(self):
        obs = self.env.unwrapped.get_obs()
        # print(obs.keys(), obs["agent"], obs["extra"])
        # print(obs["sensor_data"].keys())
        tcp_pose = np.squeeze(obs["extra"]["tcp_pose"].numpy())
        position, quat = tcp_pose[:3], tcp_pose[3:]
        return position, quat

    def get_ee_pos(self):
        position, quat = self.get_ee_pose()
        return position

    def get_ee_quat(self):
        position, quat = self.get_ee_pose()
        return quat

    def get_last_gripper_action(self):
        return self.latest_action[-1]

    """
    Description: 
        Get 3D point cloud from RGBD cameras from simulator

    Returns:
        points:
            3D point cloud
        points_colors:
            3D point cloud's color
        points_mask:
            3D point cloud's semantic id
    """

    def get_3d_point_cloud_with_exp(self):
        came_name = ["base_camera", "right_camera"]
        # came_name = ["right_camera"]
        came_name = ["base_camera"]
        came_name = ["left_camera"]
        came_name = ["right_camera"]
        # came_name = ["base_front_camera", 'base_camera', 'hand_camera']
        # came_name = ["base_front_camera", "base_camera", "left_camera"]
        came_name = ["left_camera"]
        came_name = [
            "base_front_camera",
            "base_camera",
        ]
        came_name = ["base_front_camera", "base_camera", "base_up_front_camera"]
        points, points_colors, points_mask = self.get_3d_point_cloud(
            came_name=came_name
        )
        obs = self.env.unwrapped.get_obs()
        cam_list = []

        for came in came_name:
            info = obs["sensor_param"][came]["cam2world_gl"][0]

            # breakpoint()
            cam_list.append(info)

        return points, cam_list

    def get_3d_point_cloud(
        self,
        came_name=[
            "base_front_camera",
            "base_camera",
            "base_up_front_camera",
            "front_camera",
        ],
    ):
        obs = self.env.unwrapped.get_obs()
        point_list = []
        point_color_list = []
        point_mask_list = []
        # print(obs["sensor_data"].keys())
        # input()
        for came in came_name:
            depth = np.squeeze(obs["sensor_data"][came]["depth"].cpu().numpy()) / 1000.0
            rgb = np.squeeze(obs["sensor_data"][came]["rgb"].cpu().numpy())
            param = obs["sensor_param"][came]
            mask = np.squeeze(obs["sensor_data"][came]["segmentation"].cpu().numpy())

            # points_mask_colors = self.depth_to_pc(intrinsic_matrix, depth, rgb, mask)
            points_colors = self.get_color_points(rgb, depth, param, mask)

            points = points_colors[:, :3]
            points_mask = mask.reshape((-1, 1))
            points_colors = points_colors[:, 3:]
            point_list.append(points)
            point_mask_list.append(points_mask)
            point_color_list.append(points_colors)

        points = np.concatenate(point_list, axis=0)
        points_colors = np.concatenate(point_color_list, axis=0)
        points_mask = np.concatenate(point_mask_list, axis=0)
        return points, points_colors, points_mask

    """
    Description:
        Use camera intrisic matrix and depth map to project 2D image into 3D point cloud
    Args:
        rgb: 2D RGB image. If rgb is not None, return colored point cloud.
        mask: 2D mask. If mask is not None, return masked point cloud.
    Returns:
        points: 1-3 cols are 3D point cloud, 4 col is mask, 5-7 cols are rgb.
    """

    def depth_to_pc(self, K, depth, rgb=None, mask=None):
        H, W = depth.shape
        fx, fy = K[-1, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

        x = (i - cx) * depth / fx
        y = (j - cy) * depth / fy
        z = depth

        points = np.vstack((x.flatten(), y.flatten(), z.flatten(), mask.flatten())).T

        flatten_rgb = rgb.reshape((-1, 3))
        points = np.hstack((points, flatten_rgb))
        return points

    def get_color_points(self, rgb, depth, cam_param, mask=None):
        intrinsic_matrix = np.squeeze(cam_param["intrinsic_cv"].numpy())

        color_points = self.depth_to_point_cloud(intrinsic_matrix, depth, rgb, mask)
        if self.frame == "base":
            return color_points
        cam2world_gl = np.squeeze(cam_param["cam2world_gl"].numpy())
        print("cam2world_gl", cam2world_gl)
        # input()
        #
        color_points[:, :3] = self.transform_camera_to_world(
            color_points[:, :3], cam2world_gl
        )
        return color_points

    def depth_to_point_cloud(self, K, depth, rgb=None, mask=None):
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

        x = (i - cx) * depth / fx
        y = (j - cy) * depth / fy
        z = depth
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        flatten_rgb = rgb.reshape((H * W, 3))
        points = np.hstack((points, flatten_rgb))
        return points

    """
    Description: 
        transform point cloud from camera coordinicate to world coordinate
    """

    def transform_camera_to_world(self, points, extri_mat):
        R = extri_mat[:3, :3]
        # R = np.identity(3)
        t = extri_mat[:3, 3]
        # t = t[[1, 0, 2]]
        # t = np.array([0, 0, 0])
        pcd_world = (R @ points.T).T - t
        rotation_y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        pcd_world = (rotation_y_180 @ pcd_world.T).T
        return pcd_world

    def o3d_to_grasp(self, rot):
        theta = np.pi / 2
        rotate_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        rotate_y = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        rotate_z = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        unit_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        maniskill_down = rotate_x @ rotate_x @ unit_mat
        o3d_down = rotate_z @ rotate_y @ unit_mat
        transfer_mat = maniskill_down @ o3d_down.T

        return maniskill_down @ rot

    def close(self):
        self.env.close()
