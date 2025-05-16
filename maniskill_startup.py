import os
import datetime
import traceback
import torch
import sys
from os.path import join
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
import mani_skill
import os


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# Example usage:
# clear_screen()
sys.path.append(os.getcwd())


theta = np.pi / 2
rotate_x = np.array(
    [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
)
rotate_y = np.array(
    [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
)
rotate_z = np.array(
    [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
)

from voxposer.utils_main import *
from voxposer.envs.maniskill_env import *
from voxposer.envs.voxposer_env import *
from scipy.spatial.transform import Rotation as R

# obs = mani_env.reset()
# print(obs.keys(), obs["sensor_data"].keys())
# print(mani_env.workspace_bounds_min)
# cube = voxposer_env.lmp_env.detect('cube')
# pos = cube.position
# pos_world = voxposer_env.lmp_env._voxel_to_world(pos)
# print(voxposer_env.lmp_env._voxel_to_world(pos))
# quat = mani_env.get_ee_quat()
# action = np.concatenate([pos_world, quat, np.array([1])])
# mani_env.apply_action(action)

# def get_affordance_map():
#     affordance_map = voxposer_env.lmp_env.get_empty_affordance_map()
#     pos = cube.position
#     print(pos)
#     affordance_map[pos[0], pos[1], pos[2]] = 1
#     return affordance_map
# def move_func():
#     gripper = voxposer_env.lmp_env.detect('gripper')
#     return gripper


# voxposer_env.lmp_env.execute(move_func, get_affordance_map)
# quat = np.array([0, 1, 0, 0])

# pos[2] -= 0.015
# # tran = mani_env.get_ee_pos()
# # quat = mani_env.get_ee_quat()
# action = np.concatenate([pos, quat, np.array([1])])
# mani_env.apply_action(action)
# mani_env.close_gripper()
# mani_env.reset_to_default_pose()

# tran = tran + np.array([0.1, 0.1, 0.1])
# action[:3] = tran
# mani_env.apply_action(action)
# mani_env.reset_to_default_pose()


# cube = voxposer_env.lmp_env.estimate_grasp_pose('cube')
# print(cube.translation)
# print(cube.rotation)
def process_objects(obj):
    return [x.replace("scene-0_", "").replace("scene-0-", "") for x in obj]


experiment_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = "./evaluation/" + experiment_date
pkl_path = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/interactive_instruction_objects.pkl"
import pickle as pkl

error_list = []
# processed_count = 0
total_count = 0
NUM_EPISODE = 10

with open(pkl_path, "rb") as f:
    interactive_dict = pkl.load(f)
    total_count = len(interactive_dict)
    # breakpoint()
primitive_list_pkl = "/home/lr-2002/project/reasoning_manipulation/ManiSkill/primitive_instruction_objects.pkl"

with open(primitive_list_pkl, "rb") as f:
    primitive_dict = pkl.load(f)
# breakpoint()


# try:
def process_with_try(
    env_name=None, primitive_only=False, num_episodes=NUM_EPISODE, filter=None
):
    if primitive_only:
        env_dict = primitive_dict
    else:
        env_dict = {**primitive_dict, **interactive_dict}
    # List of environments to skip due to low success rates

    import pickle as pkl

    skip_environments = []
    if os.path.exists("skip_task.pkl"):
        with open("skip_task.pkl", "rb") as f:
            skip_environments = pkl.load(f)
            print("skip environments", skip_environments)
    try:
        for name, v in env_dict.items():
            # breakpoint()
            if name in skip_environments:
                print(f"Skipping environment {name}")
                continue
            print(f"\n----- Processing environment: {name} -----")
            if env_name and env_name.lower() not in name.lower():
                continue
            if filter is not None:
                if filter not in name.lower():
                    continue
            instruction, object_list = v["ins"], v["objects"]

            print(f"Instruction: {instruction}")
            print(f"Objects: {object_list}")
            for test_iter in range(num_episodes):
                mani_env = ManiSkill_Env(
                    task_name=name,
                    output_dir=output_dir,
                    model_class="Voxposer_normal",
                )

                try:
                    voxposer_env = VoxPoser_ENV(mani_env)
                    object_list = process_objects(object_list)
                    print(f"Calling VoxPoser with processed objects: {object_list}")
                    voxposer_env.call_voxposer(
                        instruction, object_list
                    )  # this is manigen

                    print(f"Successfully processed {name}")
                    mani_env.close()
                # except KeyboardInterrupt:
                #     print("\nProcess interrupted by user (Ctrl+C)")
                #     # Clean up resources for current environment
                #     try:
                #         if "mani_env" in locals():
                #             mani_env.close()
                #     except:
                #         pass
                #     break
                except Exception as e:
                    error_list.append((name, str(e)))
                    print(f"Error processing {name}: {str(e)}")
                    traceback.print_exc()
                    # Clean up resources even if there was an error
                    # breakpoint()
                    try:
                        if "mani_env" in locals():
                            mani_env.close()
                        del mani_env
                    except:
                        pass
            # clear_screen()

    # except KeyboardInterrupt:
    #     print(
    #         f"\nProcess interrupted by user (Ctrl+C) during initialization, {error_list}"
    # )
    except Exception as e:
        print(f"Error loading environments: {str(e)}")


def process_for_one(env_name=None, task_n=0):
    cnt = 0

    if not env_name:
        for name, v in env_dict.items():
            if cnt == task_n:
                name = name
                v = v
            else:
                cnt += 1

            # if "Apple" in name:
            #     # v=  v
            #     # name = env_name
            #     break
    # print(name, v )
    # input()
    # name = "PickCube-v1"
    if not name:
        name = env_name
        v = {"ins": "pick up the cube", "objects": ["cube", "gripper"]}
    # v={'ins': 'pick the cube', 'objects': ['cube', 'gripper']}
    print(f"\n----- Processing environment: {name} -----")
    instruction, object_list = v["ins"], v["objects"]
    print(f"Instruction: {instruction}")
    print(f"Objects: {object_list}")

    mani_env = ManiSkill_Env(task_name=name)
    voxposer_env = VoxPoser_ENV(mani_env)
    object_list = process_objects(object_list)

    print(f"Calling VoxPoser with processed objects: {object_list}")
    voxposer_env.call_voxposer(instruction, object_list)  # this is manigen

    # processed_count += 1
    print(f"Successfully processed {name}")

    # Print summary
    print("\n----- Execution Summary -----")
    print(f"Total environments: {total_count}")
    # print(f"Successfully processed: {processed_count}")
    print(f"Failed environments: {len(error_list)}")

    if error_list:
        print("\n----- Error Details -----")
        for name, error in error_list:
            print(f"{name}: {error}")

    # print(mani_env.name2id)
    #


# mani_env = ManiSkill_Env(task_name="Tabletop-Pick-Apple-v1")
# points, cam_list = mani_env.get_3d_point_cloud_with_exp()
#
#
# def draw_pc(pc_arr):
#     import open3d as o3d
#
#     # 创建一个坐标系
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=1.0,  # 坐标轴的长度
#         origin=[0, 0, 0],  # 坐标系的原点
#     )
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc_arr)
#
#     # Visualize the point cloud
#     info_list = [pcd, coordinate_frame]
#     for cam in cam_list:
#         cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#             size=1.0,  # 坐标轴的长度
#             # origin=cam[:3, 3],  # 坐标系的原点
#             origin=[0, 0, 0],  # 坐标系的原点
#         )
#         cam[:3, :3] = torch.tensor(np.identity(3))
#         cam_frame.transform(cam)
#         info_list.append(cam_frame)
#
#     o3d.visualization.draw_geometries(info_list)
#     o3d.io.write_point_cloud("/tmp/habitat.ply", pcd)
#
#
# draw_pc(points)

# process_for_one("PickCube-v1")
process_with_try()
# process_with_try('Cabinet')
