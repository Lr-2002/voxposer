import gymnasium as gym

from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mshab.envs
from mshab.envs.planner import plan_data_from_file
import cv2
import numpy as np
import sapien
from utils.functions import transform_rgb_bgr
import sapien.physx as physx
import transforms3d
import os
import json
from os.path import join
import sys
sys.path.append(os.getcwd())


from voxposer.envs.mshab_env import *

mshab_env = MSHAB_ENV()
obs = mshab_env.env.get_obs()
print(obs.keys())
print(obs['agent'].keys())
print(obs['extra'].keys())
print(obs['sensor_data'].keys())
print(obs['extra']['tcp_pose_wrt_base'])
print()
# print(mshab_env.get_ee_pose())


# plan_datas = mshab_env.plan_data
# plans = plan_datas.plans

# logger = []
# for plan in plans:
#     for subtask in plan.subtasks:
#         if subtask.type == "navigate":
#             continue
#         elif subtask.type == "pick":
#             obj_id = subtask.obj_id
#             obj_id = f'env-0_{obj_id}'
#             print(obj_id)
#             obj_pose = mshab_env.name2pose[obj_id]
#             obj_center = obj_pose.p.cpu().numpy().reshape(-1)
#             p, q = mshab_env.set_agent_navigable_point(obj_center, 1.8)
#             if p is not None:
#                 logger.append({'position': p, 'quat': q})

#         elif subtask.type == "place":
#             goal_rectangle_corner = subtask.goal_rectangle_corners
#             goal_pos = subtask.goal_pos
#             p, q = mshab_env.set_agent_navigable_point(goal_pos, 1.8)
#             if p is not None:
#                 logger.append({'position': p, 'quat': q})

# with open(f'./{mshab_env.task}_teleport.json', 'w') as json_file:
#     json.dump({'teleport_point': logger}, json_file, indent=4)
# json_file.close()


