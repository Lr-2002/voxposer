import os
import random
import sys

import habitat_sim
import magnum as mn
import numpy as np
import pinocchio as pin
from habitat_sim.gfx import LightInfo, LightPositionModel
from omegaconf import OmegaConf
from pink.tasks import FrameTask
import os
import sys
sys.path.append(os.getcwd())

from articulated_agents.robots.controller import Controller
from articulated_agents.robots.fetch_robot import FetchRobot
from articulated_agents.robots.fetch_suction import FetchSuctionRobot
from articulated_agents.robots.google_robot import GoogleRobot
from articulated_agents.robots.franka_robot import FrankaRobot
from articulated_agents.robots.stretch_robot import StretchRobot
from control import fetchbot_keyboard_control
from dataset import Dataset
from simulator import Simulator
from voxposer.envs.voxposer_env import *
from voxposer.envs.habitat_env import *

# pinochio installation (from source): https://stack-of-tasks.github.io/pinocchio/download.html
# pink installation: https://github.com/stephane-caron/pink
os.environ['PYTHONPATH'] = ':'.join(
    [os.environ['PYTHONPATH'], '/home/diligent/Desktop/third_part_packages/pinocchio/pinocchio_installation/lib/python3.9/site-packages'])
os.environ['LD_LIBRARY_PATH'] = ':'.join([os.environ['LD_LIBRARY_PATH'], '/usr/local/lib', '/home/diligent/anaconda3/envs/bigai-eai/lib',
                                         '/home/diligent/Desktop/third_part_packages/pinocchio/pinocchio_installation/lib', '/home/diligent/Desktop/third_part_packages/pinocchio/pinocchio_installation/lib/pkgconfig'])


def load_episode_objects(sim, episode, transformations):
    # potential paths for object config json file
    paths = episode.additional_obj_config_paths
    objects = episode.rigid_objs

    for object in objects:
        idx = object[1]
        array = np.vstack([transformations[idx], [0, 0, 0, 1]])
        transformation = mn.Matrix4(
            [[array[j][i] for j in range(4)] for i in range(4)]
        )
        name = object[0]
        object_config_path = None
        for path in paths:
            if os.path.exists(os.path.join(path, name)):
                object_config_path = os.path.join(path, name)
                break
        print(object_config_path)
        sim.load_object(object_config_path=object_config_path,
                        transformation=transformation,
                        semantic_id=29)


if __name__ == '__main__':
    # default simulator settings
    sim_settings = OmegaConf.load('config/default_sim_config.yaml')
    robot_type = 'google_robot' # google_robot or franka_robot
    dataset = 'home-robot-remake' #-remake
    # dataset = 'simplerenv'
    # fill in "scene_dataset_config_file" and "scene" ind sim_settings
    if dataset == 'hssd':
        sim_settings["scene_dataset_config_file"] = "data/scenes/hssd-hab/hssd-hab.scene_dataset_config.json"
        sim_settings["scene"] = "03997643_171030747"
    elif dataset == "mp3d":
        sim_settings["scene_dataset_config_file"] = "data/scenes/mp3d/mp3d.scene_dataset_config.json"
        sim_settings["scene"] = "1LXtFkjw3qL"
    elif dataset == "replicaCAD":
        sim_settings["scene_dataset_config_file"] = "data/scenes/replica_cad_dataset/replicaCAD.scene_dataset_config.json"
        sim_settings["scene"] = "apt_2"
    elif dataset == "ai2thor":
        sim_settings["scene_dataset_config_file"] = "data/scenes/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json"
        sim_settings["scene"] = "FloorPlan_Val3_5"
    elif dataset == "home-robot":
        sim_settings["scene_dataset_config_file"] = "BIGAI-EAI/data/scenes/hssd-hab-home-robot/hssd-hab-uncluttered.scene_dataset_config.json"
        sim_settings["scene"] = "BIGAI-EAI/data/scenes/hssd-hab-home-robot/scenes-uncluttered-final-extend/103997643_171030747_demo.scene_instance.json"
    elif dataset == "home-robot-remake":
        sim_settings["scene_dataset_config_file"] = "data/scenes/home-robot-remake/hssd-hab-uncluttered.scene_dataset_config.json"
        sim_settings["scene"] = "data/scenes/home-robot-remake/scenes-uncluttered/104862660_172226844_new.scene_instance.json"
    elif dataset == 'simplerenv':
        sim_settings["scene_dataset_config_file"] = "jxma_data/open_drawer_google/google_pick_coke_can_1_v4.scene_dataset_config.json"
        sim_settings["scene"] = "jxma_data/open_drawer_google/configs/scenes/apt_0.scene_instance.json"

    

    fetchbot_settings = OmegaConf.load('config/fetchbot_config.yaml')
    agents_settings = [fetchbot_settings]

    # lighting
    # sim_settings["scene_light_setup"] = habitat_sim.gfx.NO_LIGHT_KEY
    # sim_settings["override_scene_light_defaults"] = True

    sim = Simulator(sim_settings, agents_settings)

    if robot_type == 'google_robot':
        robot_path = "robots/googlerobot_description/google_robot_meta_sim_fix_fingertip.urdf"
        agent = GoogleRobot(robot_path, sim._simulator)
    elif robot_type == 'franka_robot':
        robot_path = "robots/hab_franka/panda_arm_umi_hand.urdf"
        agent = FrankaRobot(robot_path, sim._simulator)
    elif robot_type == 'fetch_robot':
        robot_path = 'robots/hab_fetch/robots/fetch_arm.urdf'
        agent = FetchSuctionRobot(robot_path, sim._simulator)
    
    agent.reconfigure()
    agent.update()
    ############ Robot related settings ############
    rotation_axis = [
        mn.Vector3(0, 1, 0),
        mn.Vector3(0, 1, 0),
        mn.Vector3(1, 0, 0),
        mn.Vector3(0, 1, 0),
        mn.Vector3(1, 0, 0),
        mn.Vector3(0, 1, 0),
        mn.Vector3(1, 0, 0),
    ]
    IK_IMPL = 'pink'  # or 'pink'
    if IK_IMPL == 'pink':
        if robot_type == 'google_robot':
            pin_robot = pin.RobotWrapper.BuildFromURDF(
                'robots/googlerobot_description/google_robot_meta_sim_fix_fingertip.urdf', package_dirs='./robots/googlerobot_description/')
            list_joint_names = ['joint_torso', 'joint_shoulder', 'joint_bicep',
                                'joint_elbow', 'joint_forearm', 'joint_wrist', 'joint_gripper']
            ee_link_name = 'link_gripper_tcp'
        elif robot_type == 'franka_robot':
            pin_robot = pin.RobotWrapper.BuildFromURDF('robots/hab_franka/panda_arm_umi_hand.urdf', package_dirs='./robots/hab_franka/')
            list_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
            ee_link_name = 'panda_hand'
        elif robot_type == 'fetch_robot':
            pin_robot = pin.RobotWrapper.BuildFromURDF('robots/hab_fetch/robots/hab_suction.urdf', package_dirs='robots/hab_fetch/robots')
            list_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
            ee_link_name = 'gripper_link'

        # Note: I found the following cost settings to work well with:
        # openvla: position_cost=1.0, orientation_cost=1.0
        # teleopration: position_cost=0.2, orientation_cost=10.0
        end_effector_task = FrameTask(
            ee_link_name,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            # position_cost=0.2,  # [cost] / [m]
            # orientation_cost=10.0,  # [cost] / [rad]
            lm_damping=1.0,  # tuned for this setup
        )
    else:
        pin_robot = None
        end_effector_task = None
        list_joint_names = None
        ee_link_name = None
    agent_controller = Controller(
        agent, rotation_axis, pin_robot, end_effector_task, list_joint_names,
        ee_link_name, ee_offset=[0,0,-0.05])
    #################################################

    # load a dyamic object into the scene
    sim.load_object(object_config_path="jxma_data/objects/baked_opened_pepsi_can_v2.object_config.json",
                    translation=[-10.6, 1.3, -5.1], 
                    motion="DYNAMIC",
                    name='pepsi_can',
                    mass=0.5,
                    friction_coefficient=0.8,
                    light_setup_key="lol")

    sim.load_object(object_config_path="jxma_data/objects/baked_sponge_v2.object_config.json",
                    translation=[-10.2, 1.0, -4.9], 
                    rotation=[ -0.5, 0.5, 0.5, 0.5],
                    motion="DYNAMIC",
                    name='sponge',
                    mass=0.5,
                    friction_coefficient=0.8,
                    light_setup_key="lol",
                    scale=0.6)

    agent.base_pos = mn.Vector3([-10.3, 0.23, -4.3]) #* baseline
    agent.base_rot = mn.Rad(np.pi * 0.6)# * baseline


    # Note: MUST call the following before doing anything, otherwise the
    # perception will be a total failure due to misplacement of robots
    dt = 1/60
    sim._simulator.step_physics(dt)
    agent.update()

    vla, processor = None, None
    
    # fetchbot_keyboard_control(sim, agent, fps=sim_settings['fps'], crosshair=True, controller=agent_controller, openvla=vla, openvla_processor=processor)

    habitat_env = Habitat_ENV(sim, agent, agent_controller, 'base', 60)
    voxposer_env = VoxPoser_ENV(habitat_env)
    for i in range(100):
        voxposer_env._env.update_cameras_view()
    # voxposer_env.call_voxposer('pick up pepsi can')
    # action = np.array([0.8363638,  0.02020192, 1.030303, 0, 1, 0, 0])
    # for i in range(3):
    #     voxposer_env.move_to_pose(action)
    # voxposer_env.reset_to_default_pose()
    # voxel = np.array([84., 50., 92.])
    # voxel_world = voxposer_env.lmp_env._voxel_to_world(voxel)
    # print(f'voxel world is {voxel_world}')    

