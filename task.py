import random
import time
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from simulator import Simulator
from reward import DistanceToGoalReward
from dataset import Dataset, Episode
from measures import DistanceToGoal, Success, SPL, SoftSPL, FindObjSuccess, FindRecSuccess, PickObjSuccess, PlaceSuccess
from control import continuous_keyboard_control, discrete_keyboard_control
from utils.functions import transform_rgb_bgr
from agent import Instruction
import os



class Task:
    r"""Fundamental environment class for :ref:`habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied task with simulator
    is abstracted inside :ref:`Env`. Acts as a base for other derived
    environment classes. :ref:`Env` consists of three major components:
    ``dataset`` (`episodes`), ``simulator`` (:ref:`sim`) and :ref:`task` and
    connects all the three components together.
    """
    task_type: str
    observation_space: List[str]
    action_space: List[str]
    _dataset: Dataset
    _sim_config: Dict
    _agent_config: Dict
    _sim: Simulator
    number_of_episodes: Optional[int]
    num_of_agents: int
    current_episode_id: int
    episode_over: bool
    reward: DistanceToGoalReward

    def __init__(self, config) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """
        self.task_type = config['task_type']
        self.base_dir = config['base_dir']
        self.action_space = config['agent_config']['action_space']
        self.observation_space = config['agent_config']['sensors']

        self._dataset = Dataset(self.task_type, config['dataset_path'])
        self.number_of_episodes = len(self._dataset)
        assert(self.number_of_episodes > 0)
        self.current_episode_id = -1
        self.current_episode = None

        self._sim_config = OmegaConf.load('config/default_sim_config.yaml')
        self._sim_config.update(config['sim_config'])
        self._agent_config = OmegaConf.load('config/default_agent_config.yaml')
        self._agent_config.update(config['agent_config'])
        self._sim_config["scene_dataset_config_file"] = self.base_dir+self._dataset[0].scene_dataset_config
        self._sim_config["scene"] = self.base_dir+self._dataset[0].scene_id
        self._sim = Simulator(self._sim_config, [self._agent_config], config.get("semantic_id_mapping", None))
        self.num_of_agents = 1
        
    
    def load_episode_objects(self, dir, episode):
        #load the essential objects in the OVMM episode
        paths = episode.additional_obj_config_paths #potential paths for object config json file
        print('paths', paths)
        objects = episode.candidate_objects_hard
        for object in objects:
            position = object['position']
            name = object['object_name']
            object_config_path = None
            for path in paths:
                if os.path.exists(dir+path+name):
                    object_config_path = dir+path+name
                    break
            #print(object_config_path)
            self._sim.load_object(object_config_path=object_config_path,
                        translation=position, 
                        motion="DYNAMIC")


    def reset(self): 
        #move to the next episode
        next_episode_id = self.current_episode_id + 1
        if next_episode_id >= self.number_of_episodes:
            print("No more epsiodes. Task ended.")
            return False
        
        previous_episode = self.current_episode
        self.current_episode = self._dataset[next_episode_id]
        self.current_episode_id = next_episode_id
        
        if previous_episode == None or previous_episode.scene_id != self.current_episode.scene_id: #different scene, need to reconfigure Sim
            self._sim_config["scene_dataset_config_file"] = self.base_dir+self.current_episode.scene_dataset_config
            self._sim_config["scene"] = self.base_dir+self.current_episode.scene_id
            #self._sim = Sim(self._sim_config, [self._agent_config])
            self._sim.reconfigure(self._sim_config, [self._agent_config])
        if self.task_type == 'OVMM':
            # for every episode in OVMM, we need to delete the inserted dynamic objects
            self._sim.remove_all_dynamic_objects()
            # and then we load the objects in the new episode
            self.load_episode_objects(self.base_dir, self.current_episode)


        self._sim.initialize_agent(agent_id=0, 
                                       position=self.current_episode.start_position,
                                       rotation=self.current_episode.start_rotation)
        self.metrics_dict = {}
        if self.task_type == "navigation":
            self.reward = DistanceToGoalReward(self._sim, self.current_episode, agent_id=0)
            self.metrics = [
                DistanceToGoal(self._sim, self.metrics_dict, self.current_episode, 0),
                Success(self._sim, self.metrics_dict, self.current_episode, 0),
                SPL(self._sim, self.metrics_dict, self.current_episode, 0),
                SoftSPL(self._sim, self.metrics_dict, self.current_episode, 0),
            ]
        elif self.task_type == "OVMM":
            self.metrics = [
                FindObjSuccess(self._sim, self.metrics_dict, self.current_episode, 0),
                PickObjSuccess(self._sim, self.metrics_dict, self.current_episode, 0),
                FindRecSuccess(self._sim, self.metrics_dict, self.current_episode, 0),
                PlaceSuccess(self._sim, self.metrics_dict, self.current_episode, 0),
            ]
        return True


    def different_scene_reset(self): 
        #move to the next episode
        while True:
            next_episode_id = self.current_episode_id + 1
            if next_episode_id >= self.number_of_episodes:
                print("No more epsiodes. Task ended.")
                return False
            
            previous_episode = self.current_episode
            self.current_episode = self._dataset[next_episode_id]
            self.current_episode_id = next_episode_id
            
            if previous_episode == None or previous_episode.scene_id != self.current_episode.scene_id: #different scene, need to reconfigure Sim
                self._sim_config["scene_dataset_config_file"] = self.base_dir+self.current_episode.scene_dataset_config
                self._sim_config["scene"] = self.base_dir+self.current_episode.scene_id
                #self._sim = Sim(self._sim_config, [self._agent_config])
                self._sim.reconfigure(self._sim_config, [self._agent_config])
            else:
                continue
            
            if self.task_type == 'OVMM':
                # for every episode in OVMM, we need to delete the inserted dynamic objects
                self._sim.remove_all_dynamic_objects()
                # and then we load the objects in the new episode
                self.load_episode_objects(self.base_dir, self.current_episode)


            self._sim.initialize_agent(agent_id=0, 
                                        position=self.current_episode.start_position,
                                        rotation=self.current_episode.start_rotation)
            self.metrics_dict = {}
            if self.task_type == "navigation":
                self.reward = DistanceToGoalReward(self._sim, self.current_episode, agent_id=0)
                self.metrics = [
                    DistanceToGoal(self._sim, self.metrics_dict, self.current_episode, 0),
                    Success(self._sim, self.metrics_dict, self.current_episode, 0),
                    SPL(self._sim, self.metrics_dict, self.current_episode, 0),
                    SoftSPL(self._sim, self.metrics_dict, self.current_episode, 0),
                ]
            elif self.task_type == "OVMM":
                self.metrics = [
                    FindObjSuccess(self._sim, self.metrics_dict, self.current_episode, 0),
                    PickObjSuccess(self._sim, self.metrics_dict, self.current_episode, 0),
                    FindRecSuccess(self._sim, self.metrics_dict, self.current_episode, 0),
                    PlaceSuccess(self._sim, self.metrics_dict, self.current_episode, 0),
                ]
            return True

    def step(self, action):
        observations = self._sim.step(action)[0]
        #reward = self.reward.get_reward()
        return observations
    

    def run(self, model):
        while self.reset(): #when there is a valid episode
            print(self.current_episode_id)
            observations = self._sim.get_observations() #get the default agent's initial observations
            reward = 0
            model.reset()
            while True:
                rgb_1st = transform_rgb_bgr(observations["rgb_1st"], True)
                cv2.imshow("RGB_1st", rgb_1st)
                cv2.waitKey(int(1/60*1000))
                action = model(observations, reward)
                if action == 'stop':
                    break
                #observations, reward = self.step(action)
                observations = self.step(action)
                '''
                for metric in self.metrics:
                    metric.update_metric()
                print(reward, self.metrics_dict)
                '''
            #code for evaluate the episode here
        #code for aggregating the evaluation on all epsiodes here

    def auto_run(self):
        """
        Currently only support for OVMM
        """
        assert self.task_type == "OVMM", "auto_run() currently only support for OVMM"

        while self.reset(): #when there is a valid episode
        #while self.different_scene_reset():
            object_handles = list(self.current_episode.targets.keys())
            receptacle_handles = [e[0] for e in self.current_episode.goal_receptacles]
            obj_pos = self._sim.get_object_position(object_handles[0])
            rec_pos = self._sim.get_object_position(receptacle_handles[0])
            print(self.current_episode_id)
            observations = [self._sim.get_observations()] #get the default agent's initial observations
            # Here we generate higher level actions to support plan
            high_level_actions = [
                Instruction("goto", obj_pos),
                Instruction("look", obj_pos),
                Instruction("grab"),
                Instruction("goto", rec_pos),
                Instruction("look", rec_pos),
                Instruction("release"),
                #Instruction("wait")
            ]
            self._sim.agents[0].high_level_action_buffer += high_level_actions
            sgn = True
            while True:
                rgb_1st = transform_rgb_bgr(observations[0]["rgb_1st"], True)
                cv2.imshow("RGB_1st", rgb_1st)
                cv2.waitKey(int(1/60*1000))
                if sgn:
                    #time.sleep(20)
                    sgn = False
                observations, done = self._sim.auto_step()
                for metric in self.metrics:
                    metric.update_metric()
                print(self.metrics_dict)
                if done:
                    break


navigation_config = {
    'task_type': 'navigation',
    'base_dir': None,
    'dataset_path': "/home/yue/data/habitat-data/datasets/pointnav_mp3d_v1/test/test.json.gz",
    'sim_config':
        {
            'width': 256,
            'height': 256
        },
    'agent_config':
        {
            'action_space': ["no_op", "stop", "move_forward", "turn_left", "turn_right"],
            'sensors': ["rgb_1st", "rgb_3rd", "depth_1st", "semantic_1st"]
        }
}

OVMM_config = {
    'task_type': 'OVMM',
    'base_dir': 'data/home-robot-data/',
    'dataset_path': "data/home-robot-data/data/datasets/ovmm/val/episodes.json.gz",
    'sim_config':
        {
            'width': 1000,
            'height': 1000
        },
    'agent_config':
        {
            'action_space': ["no_op", "stop", "move_forward", "turn_left", "turn_right"],
            'sensors': ["rgb_1st", "rgb_3rd", "depth_1st", "semantic_1st"]
        }
}


class Model:
    #use for testing
    def __init__(self):
        self.cnt = 0

    def __call__(self, observations, reward):
        self.cnt += 1
        if self.cnt == 10:
            return 'stop'
        if self.cnt % 3 == 0:
            return 'move_forward'
        elif self.cnt % 3 == 1:
            return 'turn_left'
        else:
            return 'turn_right'

    def reset(self):
        self.cnt = 0

if __name__ =='__main__':
    base_dir = OVMM_config["base_dir"]
    sim_settings = OmegaConf.load('config/default_sim_config.yaml') #default simulator settings
    sim_settings["scene_dataset_config_file"] = base_dir + 'data/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json'
    sim_settings["scene"] = base_dir + "data/hssd-hab/scenes-uncluttered/103997643_171030747.scene_instance.json"
    default_agent_settings = OmegaConf.load('config/default_agent_config.yaml')
    semantic_mapping_file = 'data/home-robot-data/data/hssd-hab/semantics/object_semantic_id_mapping.json'
    with open(semantic_mapping_file) as f:
        semantic_id_mapping = json.load(f)
        semantic_id_mapping['object'] = 29

    OVMM_config['sim_config'] = sim_settings
    OVMM_config['agent_config'] = default_agent_settings
    OVMM_config['semantic_id_mapping'] = semantic_id_mapping
    task = Task(OVMM_config)
    #task.reset()
    #discrete_keyboard_control(task, type='task')
    # model = Model()
    # task.run(model)
    task.auto_run()

