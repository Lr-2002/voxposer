from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
    Sequence,
    TypeVar,
    Generic
)
import copy
import os
import random
from omegaconf import DictConfig
from itertools import groupby
import numpy as np
from numpy import ndarray
ALL_SCENES_MASK = "*"
from utils.functions import DatasetJSONEncoder, not_none_validator
import json
import gzip


class NavigationGoal:
    r"""Base class for a goal specification hierarchy."""
    position: List[float] = None
    radius: Optional[float] = None

class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius."""
    room_id: str = None
    room_name: Optional[str] = None

class ShortestPathPoint:
    position: List[Any]
    rotation: List[Any]
    action: Optional[int] = None


class Episode:
    """
    Base class for episode specification that includes only the episode_id
    and scene id. This class allows passing the minimum required episode
    information to identify the episode (unique key) to the habitat baseline process, thus saving evaluation time.
    :property episode_id: id of episode in the dataset, usually episode number.
    :property scene_id: id of scene in dataset.
    """
    episode_id: str = None
    scene_id: str = None
    scene_dataset_config: str = None
    additional_obj_config_paths: List[str] = None
    start_position: List[float] = None
    start_rotation: List[float] = None
    info: Optional[Dict[str, Any]] = None
    _shortest_path_cache: Any = None

    def __init__(self, **kwargs):
        self.episode_id = kwargs["episode_id"]
        self.scene_id = kwargs["scene_id"]
        self.scene_dataset_config = kwargs["scene_dataset_config"] if "scene_dataset_config" in kwargs else None
        self.start_position = [float(i) for i in kwargs["start_position"]]

        self.start_rotation = [float(i) for i in kwargs["start_rotation"]]
        self.info = kwargs["info"]

    def _reset_shortest_path_cache_hook(self):
        self._shortest_path_cache = None

    def __getstate__(self):
        return {
            k: v
            for k, v in self.__dict__.items() 
            if k not in {"_shortest_path_cache"}
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["_shortest_path_cache"] = None


class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """
    goals: List[NavigationGoal] = None
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None

    def __init__(self, **kwargs):
        super(NavigationEpisode, self).__init__(**kwargs)
        self.goals = kwargs["goals"]
        self.start_room = kwargs["start_room"]
        self.shortest_paths = kwargs["shortest_paths"]


class OVMMEpisode(Episode):
    # move 'targets' from 'target_receptacles' to 'goal receptacles'
    targets: Dict = None
    target_receptacles: List[Any] = None
    goal_receptacles: List[Any] = None
    candidate_objects_hard: List[Any] = None
    additional_obj_config_paths: List[str]=None
    rigid_objs: List[Any]=None
    def __init__(self, **kwargs):
        super(OVMMEpisode, self).__init__(**kwargs)
        self.targets = kwargs["targets"]
        self.target_receptacles = kwargs["target_receptacles"]
        self.goal_receptacles = kwargs["goal_receptacles"]
        self.candidate_objects = kwargs["candidate_objects"]
        self.candidate_objects_hard = kwargs["candidate_objects_hard"]
        self.additional_obj_config_paths = kwargs["additional_obj_config_paths"]
        self.rigid_objs = kwargs["rigid_objs"]
        self.viewpoints = kwargs["viewpoints"]


class Dataset:
    def __init__(self, task_type, data_path, viewpoints=None):
        self.episodes = []
        with gzip.open(data_path, "rb") as file:
            data = json.loads(file.read())
        if task_type == 'navigation':
            for episode in data["episodes"]:
                episode = NavigationEpisode(**episode)
                self.episodes.append(episode)
        if task_type == 'OVMM':
            for episode in data["episodes"]:
                episode = OVMMEpisode(**episode, viewpoints=viewpoints)
                self.episodes.append(episode)

    def __getitem__(self, index):
        return self.episodes[index]
    
    def __len__(self):
        return len(self.episodes)


if __name__ == '__main__':
    d = Dataset('navigation', "/home/yue/data/habitat-data/datasets/objectnav_hssd-hab_v0.2.3/train/content/102344022.json.gz")
    print(d[0].object_category)