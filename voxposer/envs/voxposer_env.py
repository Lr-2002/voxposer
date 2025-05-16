import os
import sys
import cv2
import numpy as np
import torch
import re
import ast
from os.path import join
import open3d as o3d

sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(), "voxposer"))

from server_client import *
from visualizers import *
from utils import *
from GPT.call_LM import GPT
from arguments import get_config
from voxposer.interfaces_hab import setup_LMP
from voxposer.utils_main import set_lmp_objects, draw_pc


class VoxPoser_ENV:
    """
    Args:
        env: simulator env class, contain perception, executor and other interfaces that provided to voxposer
    """

    def __init__(self, env, visulizer_config=None):
        self._env = env  # env for different simulators
        self.perception_gt = (
            True  # if true, get ground truth information from simulator
        )
        if visulizer_config is not None:
            self.visualizer = ValueMapVisualizer(visulizer_config)
            self.visualizer.update_bounds(
                self.workspace_bounds_min, self.workspace_bounds_max
            )

        # memory part
        self.object_pd_buffer = {}
        self.grasp_pose_buffer = {}

        # client agent init
        self.client_agent_init()

        self.gt = True
        self._env.reset()

        config = get_config("rlbench")
        _, self.lmp_env = setup_LMP(self, config, debug=False)

    def call_voxposer(self, instruction, main_object_name=["apple"]):
        self.reset()
        config = get_config("rlbench")
        lmps, lmp_env = setup_LMP(self, config, debug=False)
        voxposer_ui = lmps["plan_ui"]
        # main_object_name = self.get_object_names_vlm(instruction)
        # main_object_name = ["apple"]
        set_lmp_objects(lmps, main_object_name)
        voxposer_ui(instruction)

    @property
    def workspace_bounds_max(self):
        return self._env.workspace_bounds_max

    @property
    def workspace_bounds_min(self):
        return self._env.workspace_bounds_min

    @property
    def name2id(self):
        return self._env.name2id

    def reset(self):
        self.object_pd_buffer = {}
        self.grasp_pose_buffer = {}
        self._env.reset()

    def set_env_resolution(self, resolution):
        self._env.set_env_resolution(resolution)

    def apply_action(self, action, ignore_arm=False, ignore_ee=False):
        self._env.apply_action(action, ignore_arm, ignore_ee)

    def reset_to_default_pose(self):
        self._env.reset_to_default_pose()

    def get_ee_pose(self):
        return self._env.get_ee_pose()

    def get_ee_pos(self):
        return self._env.get_ee_pos()

    def get_ee_quat(self):
        return self._env.get_ee_quat()

    def get_last_gripper_action(self):
        return self._env.get_last_gripper_action()

    def get_3d_obs_by_name(
        self,
        query_name,
    ):
        print(self.name2id)
        if query_name in self.object_pd_buffer.keys():
            return self.object_pd_buffer[query_name]["points"], self.object_pd_buffer[
                query_name
            ]["normals"]

        points, colors, masks = self._env.get_scene_3d_obs()
        print("all name ", self.name2id, query_name)
        if self.gt == True:
            print("---- query_name", query_name)
            # breakpoint()
            if "scene-0-" in query_name:
                query_name = query_name.replace("scene-0-", "")
            if query_name not in self.name2id.keys():
                # equal_list = self.query_name_equal(query_name, self.name2id.keys())
                # breakpoint()
                raise NotImplementedError("your input query_name is not in the list ")
                equal_list = []
                if len(equal_list) != 0:
                    equal_name = equal_list[0]
                    print(f"\033[91mI find {equal_name} to replace {query_name}\033[0m")
                    obj_ids = self.name2id[equal_name]
                else:
                    assert f"cannot find equivalent object of {query_name} in object list {self.object_pd_buffer.keys()}. cannot get 3d obs with gt format"
            else:
                obj_ids = self.name2id[query_name]
        else:
            obj_ids = 1
        obj_points = []
        obj_colors = []

        print(f"object id is {obj_ids}")
        # breakpoint()
        for i in range(len(points)):
            if self.gt == True:
                point = points[i]
                mask = masks[i]
                indices = np.where(mask == obj_ids)[0]

                obj_point = point[indices, :]
                color = colors[i][..., :3].reshape(-1, 3)
                obj_color = color[indices, :]
                obj_points.append(obj_point)
                # print("obj size", obj_point.shape)
                obj_colors.append(obj_color)
            else:
                self.perception_client.send((query_name, colors[i]))
                obj_mask = self.perception_client.receive()

                point = points[i]
                color = colors[i][..., :3].reshape(-1, 3)
                obj_mask = obj_mask.reshape(-1)
                indices = np.where(obj_mask == obj_ids)[0]
                obj_point = point[indices, :]
                obj_color = color[indices, :]
                obj_points.append(obj_point)
                obj_colors.append(obj_color)
        obj_points = np.concatenate(obj_points, axis=0)
        print("obj size", obj_points.shape)
        obj_colors = np.concatenate(obj_colors, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        o3d.io.write_point_cloud("/tmp/habitat.ply", pcd)
        draw_pc(obj_points)
        obj_normals = np.asarray(pcd.normals)
        self.object_pd_buffer[query_name] = {
            "points": obj_points,
            "colors": obj_colors,
            "normals": obj_normals,
        }
        return obj_points, obj_normals

    def get_grasp_pose_graspnet(self, object_pd, obj_normal=None, query_name=None):
        if query_name is not None and query_name in self.grasp_pose_buffer.keys():
            return self.grasp_pose_buffer[query_name]["tran"], self.grasp_pose_buffer[
                query_name
            ]["rot"]

        if self.gpd_client is None:
            print("Grasp client is None, please open grasp pose detection server")
            return None, None
        scene_pd, scene_color = self.get_scene_3d_obs()

        self.gpd_client.send((object_pd, scene_pd))
        tran, rot = self.gpd_client.receive()
        if len(tran) == 0:
            tran = np.mean(object_pd, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(object_pd)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            normals = pcd.normals
            rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            direct_normal = np.mean(normals, axis=0)
            rot = align_rotation_to_vector(rotation, direct_normal)
        rot = self._env.o3d_to_grasp(rot)
        return tran, rot

    def clear_memory_buffer(self):
        self.object_pd_buffer = {}
        self.grasp_pose_buffer = {}

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        points, colors, masks = self._env.get_scene_3d_obs(
            ignore_robot, ignore_grasped_obj
        )
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        return points, colors

    """
    Description:
        Update 3D voxel map
    """

    def update_visualizer(self):
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(
                ignore_robot=False, ignore_graspd_obj=False
            )
            self.visualizer.update_scene_points(points, colors)

    def visualize_cameras_view(self):
        self._env.update_camera_view()

    # LLM tools part

    def client_agent_init(self):
        try:
            self.perception_client = SocketClient(parent_port=12360)
        except:
            self.perception_client = None
            print("cannot connect to perception client, please check it is started")
        try:
            self.gpd_client = SocketClient(parent_port=12377)
        except:
            self.gpd_client = None
            print(
                "cannot connect to grasp pose detection client, please check it is started"
            )

        self.chat_agent = GPT("tonggpt")

    def get_object_names_vlm(self, task_instruction, image_list=None):
        system_content = f"You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."

        with open(
            join(os.getcwd(), "voxposer/prompts/rlbench/open_vocab_prompt.txt"), "r"
        ) as text_file:
            prompt = text_file.read()
        text_file.close()
        prompt = prompt.replace("TASK_TEMPLATE", task_instruction)

        message = self.chat_agent.create_message(
            prompt, image_list, system_content=system_content
        )
        response = self.chat_agent._call_lm(message, "gpt-4o-2024-08-06")

        main_object_list = self.parse_list(response)

        with open(
            join(os.getcwd(), "voxposer/prompts/rlbench/open_world_detect_prompt.txt"),
            "r",
        ) as text_file:
            prompt = text_file.read()
        text_file.close()
        prompt = prompt.replace("TASK_TEMPLATE", task_instruction)
        prompt = prompt.replace("OBJECT_TEMPLATE", str(main_object_list))

        message = self.chat_agent.create_message(prompt, system_content=system_content)
        response = self.chat_agent._call_lm(message, "gpt-4o-2024-08-06")

        fine_grained_objects = self.parse_list(response)
        objects = main_object_list
        for object in fine_grained_objects:
            if object in objects:
                continue
            objects.append(object)

        return objects

    def move_to_pose(self, action, speed=None):
        self._env.move_to_pose(action)

    def open_gripper(self):
        self._env.open_gripper()

    def close_gripper(self):
        self._env.close_gripper()

    def set_gripper_state(self, state):
        self._env.set_gripper_state(state)

    def query_name_equal(self, query, name_list):
        with open(
            join(os.getcwd(), "voxposer/prompts/rlbench/query_equal_name.txt"), "r"
        ) as file:
            prompt = file.read()
        file.close()
        prompt = prompt.replace("QUERY_TEMPLATE", query)
        prompt = prompt.replace("CANDIDATES_TEMPLATE", str(name_list))
        message = self.chat_agent.create_message(prompt)
        response = self.chat_agent._call_lm(message, "gpt-4o-2024-08-06")
        equal_list = self.parse_list(response)
        return equal_list

    def parse_list(self, res_str):
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, res_str, re.DOTALL)
        if len(matches) != 0:
            actual_list = ast.literal_eval("[" + matches[0] + "]")
            return actual_list
        else:
            return []
