import json
import math
import time
# from new_agent import run_agent
from multiprocessing import Process, Queue
# from new_agent import run_agent
from queue import Queue
# from new_agent import run_agent
from threading import Thread
from typing import (TYPE_CHECKING, Any, Dict, Iterator, List, Optional,
                    Sequence, Tuple, Union, overload)

import cv2
import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim import Simulator as Sim
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum
from omegaconf import DictConfig

from agent import Agent
from utils.functions import *
from utils.make_config import make_agent_cfg, make_sim_cfg


class Simulator:
    """simulator related"""
    _resolution: [int, int] # resolution of all sensors
    _fps: int # each step in the simulator equals to (1/fps) secs in the simulated world
    _config: DictConfig # used to initialize the habitat-sim simulator
    _simulator: Sim # habitat_sim simulator

    """agent related"""
    agents: List[Agent] # store agent states during simulation
    num_of_agents: int # number of agents in the simulator
    _agent_object_ids: List[int] # the object_ids of the rigid objects attached to the agents
    _default_agent_id: int # 0 for the default agent
    holding_object: Dict # a dict(key: agent_id, value: object_id) that records which object the agent is holding
    grab_and_release_distance: float # the maximal distance for all agents to grab or release an object

    """object related"""
    _obj_template_mgr: Any #habitat-sim ObjectAttributesManager
    _articulated_obj_mgr: Any #habitat-sim ArticulatedObjectManager
    _rigid_obj_mgr: Any #habitat-sim RigidObjectManager
    semantic_id_mapping: Dict # a dict (key: str, value: int) that maintains the object categories and their semantic ids
    articulated_object_state: Dict # key: articulated_object_id, value: 'open' or 'closed'
    articulated_object_semantic: Dict # key: articulated_object_id, value: semantic_id
    link_id2art_id: Dict # key: link_id, value: the articulated_object_id of the articulated object that the link belongs to
    semantic2: Dict # key: category name, value: semantic_id
    id2semantic: Dict # key: semantic id, value: category_name

    """threading related"""
    agent_threads: List[Thread] # a list of agent threads
    action_queue: List[Queue] # an agent sends atomic actions to the simulator by its action queue
    observation_queue: List[Queue] # the simulator sends observations to the agent by the observation queue
    cheat_request_queue: List[Queue] # an agent sends a cheat request to the simulator by cheat request queue
    cheat_action_queue: List[Queue] # the simulator handles cheat request and returns a list of atomic actions to the cheat action queue
    world_observation_queue: List[Queue]

    def __init__(self, sim_settings, agents_settings, semantic2id=None) -> None:
        self._resolution = [sim_settings['height'], sim_settings['width']]
        self._fps = sim_settings['fps']
        self.agents = []
        agent_configs = []
        for i in range(len(agents_settings)):
            agent_configs.append(make_agent_cfg(self._resolution, agents_settings[i]))
            self.agents.append(Agent(self, agents_settings[i], i))
        self._config = make_sim_cfg(sim_settings, agent_configs)
        self._simulator = Sim(self._config)
        self.num_of_agents = len(self.agents)
        self._agent_object_ids = [None for _ in range(self.num_of_agents)] #by default agents have no rigid object bodies

        self._obj_template_mgr = self._simulator.get_object_template_manager()
        self._rigid_obj_mgr = self._simulator.get_rigid_object_manager()
        self._articulated_obj_mgr = self._simulator.get_articulated_object_manager()

        self.grab_and_release_distance = sim_settings['grab_and_release_distance']
        self._default_agent_id = sim_settings["default_agent"]
        self.holding_object = dict()
        for agent_id in range(self.num_of_agents):
            agent_object_path = agents_settings[agent_id]['agent_object']
            self.attach_object_to_agent(agent_object_path, agent_id)
            self.holding_object[agent_id] = None
        self.articulated_object_state = dict()
        self.articulated_object_semantic = dict()
        self.link_id2art_id = dict()
        self.semantic2id = dict()
        self.id2semantic = dict()

        if semantic2id is not None:
            self.semantic2id = semantic2id
            for c in self.semantic2id:
                self.id2semantic[self.semantic2id[c]] = c

        self.action_queue = [Queue() for _ in range(self.num_of_agents)]
        self.observation_queue = [Queue() for _ in range(self.num_of_agents)]
        self.cheat_request_queue = [Queue() for _ in range(self.num_of_agents)]
        self.cheat_action_queue = [Queue() for _ in range(self.num_of_agents)]
        self.world_observation_queue = [Queue() for _ in range(self.num_of_agents)]

        #agent threads will not start unless self.simulate() is called
        # self.agent_threads = []
        # for i in range(self.num_of_agents):
        #     action = self.action_queue[i]
        #     observation = self.observation_queue[i]
        #     cheat_request = self.cheat_request_queue[i]
        #     cheat_action = self.cheat_action_queue[i]
        #     world_observation = self.world_observation_queue[i]
        #     self.agent_threads.append(Thread(target=run_agent, args=(i, action, observation, cheat_request, cheat_action, world_observation)))

        # preprocessing the semantic ID
        mapping_files = [
            'data/objects/articulated_receptacles.json',
            'data/objects/pickable_objects.json',
            'data/objects/rigid_receptacles.json',
            'data/objects/uncommon_receptacles.json',
            'data/objects/unpickable_objects.json'
        ]
        self._template_to_name = {}
        for i in mapping_files:
            for k, v in json.load(open(i, 'r')).items():
                if k in self._template_to_name:
                    self._template_to_name[k].append(v)
                else:
                    self._template_to_name[k] = [v]

        self._semantic_id_to_name = {}
        self._name_to_semantic_id = {}
        self._handle_to_name = {}
        self._name_to_handle = {}
        self._loaded_object_handle = []
        self._sem_cnt = 1
        for handle in self.get_all_rigid_objects():
            template = handle.split("_:")[0]
            object = self._rigid_obj_mgr.get_object_by_handle(handle)
            object.semantic_id = self._sem_cnt
            self._sem_cnt += 1
            # we use the first name in the list as the object name
            try:
                name = self._template_to_name[template][0]
            except:
                name = 'unknown'
            self._semantic_id_to_name[object.semantic_id] = name
            self._handle_to_name[handle] = name
            if name in self._name_to_semantic_id:
                self._name_to_semantic_id[name].append(object.semantic_id)
            else:
                self._name_to_semantic_id[name] = [object.semantic_id]
            if name in self._name_to_handle:
                self._name_to_handle[name].append(handle)
            else:
                self._name_to_handle[name] = [handle]

    def __del__(self):
        self._simulator.close()

    def all_object_names(self):
        return list(self._name_to_semantic_id.keys())

    def name_to_semantic_id(self, name):
        return self._name_to_semantic_id[name]

    def semantic_id_to_name(self, semantic_id):
        return self._semantic_id_to_name[semantic_id]

    def print_scene_object_category(self):
        """print the objects' semantic_ids of the current scene."""
        categories = set()
        for object_handle in self._rigid_obj_mgr.get_object_handles():
            object = self._rigid_obj_mgr.get_object_by_handle(object_handle)
            semantic_id = object.semantic_id
            if semantic_id in self.id2semantic:
                categories.add(self.id2semantic[semantic_id])
        for object_handle in self._articulated_obj_mgr.get_object_handles():
            object_id = self._articulated_obj_mgr.get_object_id_by_handle(object_handle)
            if object_id not in self.articulated_object_semantic:
                continue
            semantic_id = self.articulated_object_semantic[object_id]
            if semantic_id in self.id2semantic:
                categories.add(self.id2semantic[semantic_id])
        print('available category: {categories}')


    def update_articulated_id_mapping(self):
        """maintain link_id2art_id, articulated_object_state"""
        for object_handle in self._articulated_obj_mgr.get_object_handles():
            articulated_object = self._articulated_obj_mgr.get_object_by_handle(object_handle)
            articulated_object_id = articulated_object.object_id
            for link_object_id in articulated_object.link_object_ids:
                self.link_id2art_id[link_object_id] = articulated_object_id
            if articulated_object_id not in self.articulated_object_state:
                self.articulated_object_state[articulated_object_id] = 'closed'


    def get_all_rigid_objects(self):
        """return the list of all object handles of the rigid objects"""
        return self._rigid_obj_mgr.get_object_handles()


    def get_all_loaded_objects(self):
        """return the list of all object handles of the manually loaded objects"""
        return self._loaded_object_handle


    def get_object_position(self, object_handle):
        """get the opsition of an object indictated by its handle"""
        obj = self._rigid_obj_mgr.get_object_by_handle(object_handle)
        object_position = obj.translation
        return object_position



    def load_articulated_object(self, urdf_path, semantic_id, position=[0.0, 1.5, 0.0], rotation=mn.Quaternion.rotation(mn.Deg(90.0), [0.0, 1.0, 0.0]), scale=1.0):
        """load an articulated object"""
        obj = self._articulated_obj_mgr.add_articulated_object_from_urdf(urdf_path, fixed_base=True, global_scale=scale)
        obj.translation = position
        obj.rotation = rotation
        obj.motion_type = habitat_sim.physics.MotionType.STATIC
        self.articulated_object_semantic[obj.object_id] = semantic_id
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = self.agents[self._default_agent_id].agent_height
        navmesh_settings.agent_radius = self.agents[self._default_agent_id].agent_radius
        navmesh_settings.include_static_objects = True
        self._simulator.recompute_navmesh(
            self._simulator.pathfinder,
            navmesh_settings,
        )
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC


    def awake_all_objects(self):
        """awake all objects for simulation"""
        rigid_handles = self._rigid_obj_mgr.get_object_handles()
        for handle in rigid_handles:
            obj = self._rigid_obj_mgr.get_object_by_handle(handle)
            obj.awake = True
        articulated_handles = self._articulated_obj_mgr.get_object_handles()
        for handle in articulated_handles:
            obj = self._articulated_obj_mgr.get_object_by_handle(handle)
            obj.awake = True


    def remove_rigid_object(self, object_handle):
        """remove an rigid_object by its handle"""
        self._rigid_obj_mgr.remove_object_by_handle(object_handle,
                                                delete_object_node=True,
                                                delete_visual_node=True)

    def remove_all_dynamic_objects(self):
        """remove all dynamic objects in the scene"""
        object_handles = self.get_all_rigid_objects()
        for handle in object_handles:
            obj = self._rigid_obj_mgr.get_object_by_handle(handle)
            if obj.motion_type == habitat_sim.physics.MotionType.DYNAMIC:
                self.remove_rigid_object(handle)


    def nearest_object_position_with_category(self, agent_id, category):
        """get the nearest object position with the given semantic category"""
        if category not in self.semantic2id:
            print('Invalid category '+category+'!')
            return None
        semantic_id = self.semantic2id[category]

        agent_position = self.get_agent_state(agent_id).position
        agent_island = self.get_island(agent_position)
        distance = None
        closest_handle = None
        closest_position = None

        for art_id in self.articulated_object_semantic:
            if self.articulated_object_semantic[art_id] == semantic_id:
                obj = self._articulated_obj_mgr.get_object_by_id(art_id)
                object_position = obj.translation
                target_pos = self.get_path_finder().snap_point(point=object_position, island_index=agent_island)
                cur_distance = self.geodesic_distance(agent_position, target_pos)
                if distance == None or cur_distance < distance:
                    distance = cur_distance
                    closest_handle = obj.handle
                    closest_position = object_position
        if distance is not None:
            return closest_position

        handles = self.get_all_rigid_objects()
        for handle in handles:
            obj = self._rigid_obj_mgr.get_object_by_handle(handle)
            if obj.semantic_id == semantic_id:
                object_position = obj.translation
                target_pos = self.get_path_finder().snap_point(point=object_position, island_index=agent_island)
                cur_distance = self.geodesic_distance(agent_position, target_pos)
                if distance == None or cur_distance < distance:
                    distance = cur_distance
                    closest_handle = handle
                    closest_position = object_position
        return closest_position


    def get_observations(self):
        """get all observations"""
        obs = self._simulator.get_sensor_observations()
        return obs


    def get_agent_state(self, agent_id=0):
        """get the agent state given agent_id"""
        state = self._simulator.agents[agent_id].get_state()
        return state


    def set_agent_state(self, state, agent_id=0):
        """set the agent state given agent_id"""
        state = self._simulator.agents[agent_id].set_state(state)
        return state


    def get_object_id_in_hand(self, agent_id=None):
        """return the object_id of the object in hand, -1 if no object is in hand"""
        if agent_id is None:
            agent_id = self._default_agent_id
        return self.holding_object[agent_id]

    def set_object_id_in_hand(self, object_id, agent_id=None):
        """grab an object"""
        if agent_id is None:
            agent_id = self._default_agent_id
        self.holding_object[agent_id] = object_id


    def get_path_finder(self):
        """get the habitat-sim pathfinder"""
        return self._simulator.pathfinder


    def get_island(self, position):
        """get the navmesh id of the position"""
        return self.get_path_finder().get_island(position)


    def unproject(self, agent_id=None):
        """get the crosshair ray of the agent"""
        if agent_id is None:
            agent_id = self._default_agent_id
        sensor = self._simulator.get_agent(agent_id)._sensors["rgb_1st"]
        view_point = mn.Vector2i(self._resolution[0]//2, self._resolution[1]//2)
        ray = sensor.render_camera.unproject(view_point, normalized=True)
        return ray


    def get_camera_info(self, camera_name, agent_id=None):
        if agent_id is None:
            agent_id = self._default_agent_id
        sensor = self._simulator.get_agent(agent_id)._sensors[camera_name]
        camera = sensor.render_camera
        print('camera matrix: ', camera.camera_matrix)
        print('node: ', camera.node)
        print('projection matrix: ', camera.projection_matrix)
        return camera.camera_matrix, camera.projection_matrix



    def info_action(self, agent_id=0):
        """get the info of the object the agent is looking at"""
        hit_object_id, hit_point = self.get_nearest_object_under_crosshair(
            self.unproject(agent_id)) #get the first object (or the stage) hit by the viewpoint of the agent
        if hit_object_id == None:
            print("Too far away!")
            return
        if hit_object_id == -1: #the center viewpoint ray is hitting on the stage instead of an object
            print("Hitting on the stage!")
            return
        if hit_object_id in self._agent_object_ids: #hitting on an agent
            print("Hitting on an agent!")
            return
        #the center viewpoint ray is hitting on an object
        print('hit_point: ', hit_point)
        object_in_view = self._rigid_obj_mgr.get_object_by_id(hit_object_id)
        if object_in_view == None:
            print("Cannot get the position of an articulated object!")
            return
        else:
            print('rigid object handle:', object_in_view.handle)
            print('rigid object position:', object_in_view.translation)
            print('rigid object rotation:', object_in_view.rotation)


    def goto_action(self, target_position, agent_id=0):
        """return an action list to goto someplace"""
        agent_state = self.get_agent_state(agent_id)
        agent_position = agent_state.position
        agent_island = self.get_island(agent_position)
        #project the target position to the agent's navmesh island
        path_finder = self.get_path_finder()
        target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
        follower = habitat_sim.GreedyGeodesicFollower(
            path_finder,
            self._simulator.agents[agent_id],
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right")
        action_list = follower.find_path(target_on_navmesh)
        #the final item in the action_list is 'None', remove it and return the action sequence
        return action_list


    def grab_release_action(self, agent_id=None):
        """atomic grab and release action"""
        hit_object_id, hit_point = self.get_nearest_object_under_crosshair(
            self.unproject(agent_id)) #get the first object (or the stage) hit by the viewpoint of the agent
        # print("hit object: ", hit_object_id)
        # print("hit point: ", hit_point)
        if hit_object_id == None: #can neither pick nor place
            return "Too far away!"

        object_id_in_hand = self.get_object_id_in_hand(agent_id)
        if object_id_in_hand == None: #no object in hand, try to pick an object
            if hit_object_id == -1: #the center viewpoint ray is hitting on the stage instead of an object
                return "Hitting on the stage!"
            if hit_object_id in self._agent_object_ids: #hitting on an agent
                return "Cannot grab an agent!"
            #the center viewpoint ray is hitting on an object
            object_in_hand = self._rigid_obj_mgr.get_object_by_id(hit_object_id)
            if object_in_hand == None:
                return "Cannot grab articulated object!"
            if object_in_hand.motion_type != habitat_sim.physics.MotionType.DYNAMIC:
                return "Cannot grab object with non-dynamic motion type!"
            # the object is now legal for grabbing
            object_position = object_in_hand.translation
            new_object_position = mn.Vector3(
                object_position.x,
                object_position.y-10000.0,
                object_position.z
            )
            object_in_hand.translation = new_object_position #change the object altitude to hide it underground
            object_in_hand.motion_type = habitat_sim.physics.MotionType.STATIC #set it to static temporarily to avoid continuous fall
            self.set_object_id_in_hand(hit_object_id, agent_id)
            return "Grab the target sucessfully!"
        else: #agent is grabbing an object, try to place it in the scene
            object_in_hand = self._rigid_obj_mgr.get_object_by_id(object_id_in_hand)
            object_in_hand.motion_type = habitat_sim.physics.MotionType.DYNAMIC #change the grabbed object to DYNAMIC
            can_place = False
            old_object_position = object_in_hand.translation
            for i in range(10): #start from the hit point, try different object altitude
                #print("try: ", i)
                new_object_position = mn.Vector3(
                    hit_point.x,
                    hit_point.y+i*0.1,
                    hit_point.z
                )
                object_in_hand.translation = new_object_position
                if not object_in_hand.contact_test(): #the object will not contact the collision world
                    can_place = True
                    break
            if can_place:
                object_in_hand.translation = new_object_position
                self.set_object_id_in_hand(None, agent_id)
                return "Place the target sucessfully!"
            else: #invalid place to release the object, freeze the object underground again
                object_in_hand.translation = old_object_position
                object_in_hand.motion_type = habitat_sim.physics.MotionType.STATIC
                return "Cannot place object here!"


    def open_close_action(self, agent_id=0):
        """open or close a receptacle"""
        self.update_articulated_id_mapping()
        hit_object_id, hit_point = self.get_nearest_object_under_crosshair(
            self.unproject(agent_id)) #get the first object (or the stage) hit by the viewpoint of the agent
        if hit_object_id == None: #too far
            return "Too far away!"
        if hit_object_id == -1:
            return "Cannot interact with the stage!"
        object = self._rigid_obj_mgr.get_object_by_id(hit_object_id)
        if object is not None: # it is a rigid object not an articulated object
            return "Cannot interact with a rigid object!"
        object = self._articulated_obj_mgr.get_object_by_id(hit_object_id)
        if object is None: #hitting on a non-base link of an articulated object
            articulated_object_id = self.link_id2art_id[hit_object_id]
            object = self._articulated_obj_mgr.get_object_by_id(articulated_object_id)
        object_state = self.articulated_object_state[object.object_id]
        joint_velocities = object.joint_velocities
        if object_state == 'closed':
            new_joint_velocities = [5.0 for _ in range(len(joint_velocities))]
            self.articulated_object_state[object.object_id] = 'open'
        else:
            new_joint_velocities = [-5.0 for _ in range(len(joint_velocities))]
            self.articulated_object_state[object.object_id] = 'closed'
        object.joint_velocities = new_joint_velocities
        print('successfully interacted with the articulated object.')


    def look_action(self, target_position, agent_id=0):
        """return an action list to look at someplace"""
        camera_ray = self.unproject(agent_id)
        #get the camera's center pixel position, ray direction, and target direction
        camera_position = np.array(camera_ray.origin)
        camera_direction = np.array(camera_ray.direction)
        target_direction = np.array(target_position)-camera_position
        target_direction = target_direction / np.linalg.norm(target_direction)
        action_list = []
        #initialize the inner product
        max_product = np.dot(target_direction, camera_direction)
        y_axis = [0.0, 1.0, 0.0]
        #greedy algorithm for maximizing the inner product of the camera ray and the target direction
        #first try to turn left and right
        while True:
            step = None
            current_camera_direction = None
            for action in ['turn_left', 'turn_right']:
                degree = self.agents[agent_id].action_space[action]
                if action == 'turn_left':
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=y_axis, radian=degree/180*np.pi)
                if action == 'turn_right':
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=y_axis, radian=-degree/180*np.pi)
                product = np.dot(new_camera_direction, target_direction)
                if product > max_product:
                    max_product = product
                    current_camera_direction = new_camera_direction
                    step = action
            if step == None:
                break
            camera_direction = current_camera_direction
            action_list.append(step)

        while True:
            step = None
            current_camera_direction = None
            for action in ['look_up', 'look_down']:
                degree = self.agents[agent_id].action_space[action]
                if action == 'look_up':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=-degree/180*np.pi)
                if action == 'look_down':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=degree/180*np.pi)
                product = np.dot(new_camera_direction, target_direction)
                if product > max_product:
                    max_product = product
                    current_camera_direction = new_camera_direction
                    step = action
            camera_direction = current_camera_direction
            action_list.append(step)
            if step == None:
                break
        return action_list


    def perform_discrete_collision_detection(self):
        """perform discrete collision detection for the scene"""
        self._simulator.perform_discrete_collision_detection()


    def get_physics_contact_points(self):
        """return a list of ContactPointData ” “objects describing the contacts from the most recent physics substep"""
        return self._simulator.get_physics_contact_points()



    def is_agent_colliding(self, agent_id, action):
        """ check wether the action will cause collision. Used to avoid border conditions during simulation. """
        if action not in ["move_forward", "move_backward"]: #only move action will cause collision
            return False
        step_size = self.agents[agent_id].step_size
        agent_transform = self._simulator.agents[agent_id].body.object.transformation
        if action == "move_forward":
            position = - agent_transform.backward * step_size
        else:
            position = agent_transform.backward * step_size

        new_position = agent_transform.translation + position
        filtered_position = self.get_path_finder().try_step(
            agent_transform.translation,
            new_position)
        dist_moved_before_filter = (new_position - agent_transform.translation).dot()
        dist_moved_after_filter = (filtered_position - agent_transform.translation).dot()
        # we check to see if the the amount moved after the application of the filter
        # is _less_ than the amount moved before the application of the filter
        EPS = 1e-4
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        return collided


    def get_nearest_object_under_crosshair(self, ray):
        """ get the nearest object hit by the crosshair ray within the grab_release distance """
        ray_cast_results = self._simulator.cast_ray(ray, self.grab_and_release_distance)
        object_id = None
        hit_point = None
        if ray_cast_results.has_hits(): #the ray hit some objects
            first_hit = ray_cast_results.hits[0]
            object_id = first_hit.object_id
            hit_point = first_hit.point
        #object_id: None if no hit, -1 if hit on the stage, non-negative value if hit on an object
        return object_id, hit_point


    def auto_step(self):
        """ automatically simluates the agent's planned actions, return observations and a stop signal """
        self.awake_all_objects()
        actions = dict()
        sgn = False
        for agent_id in range(self.num_of_agents):
            action = self.agents[agent_id].get_next_action()
            if action == "grab_release":
                self.grab_release_action(agent_id)
                actions[agent_id] = "no_op"
            elif action == 'open_close':
                self.open_close_action(agent_id)
                actions[agent_id] = "no_op"
            elif action == 'info':
                self.info_action(agent_id)
                actions[agent_id] = "no_op"
            elif action == "stop":
                sgn = True
                actions[agent_id] = "no_op"
            else:
                actions[agent_id] = action
        observations = self._simulator.step(action=actions, dt=1/self._fps)
        for i in range(self.num_of_agents):
            self.agents[i].set_observations(observations[i])
        return observations, sgn


    def step(self, actions: Union[str, dict, None]):
        """all agents perform actions in the environment and return observations."""
        self.awake_all_objects()
        if actions == None:
            actions = {self._default_agent_id: "no_op"}
        assert type(actions) in [str, dict]
        if type(actions) is str: #a single action for the default agent
            actions = {self._default_agent_id: actions}
        for agent_id in actions:
            action = actions[agent_id]
            # print(action)
            # print(actions[agent_id])
            assert action in self.agents[agent_id].action_space
            if action == "grab_release":
                self.grab_release_action(agent_id)
                actions[agent_id] = "no_op"
            if action == "open_close":
                self.open_close_action(agent_id)
                actions[agent_id] = "no_op"
            if action == "info":
                self.info_action(agent_id)
                actions[agent_id] = "no_op"
            agent_position = self.get_agent_state().position
        observations = self._simulator.step(action=actions, dt=1/self._fps)
        return observations


    def handle_cheat_request(self):
        """handle the cheat request from the agents. return the atomic actions to cheat action queue"""
        for agent_id in range(self.num_of_agents):
            if self.cheat_request_queue[agent_id].empty():
                continue
            cheat_request = self.cheat_request_queue[agent_id].get()
            command, target = cheat_request[0], cheat_request[1]
            target_position = self.nearest_object_position_with_category(agent_id, target)
            if command == 'goto':
                cheat_actions = self.goto_action(target_position=target_position, agent_id=agent_id)
                for a in cheat_actions:
                    self.cheat_action_queue[agent_id].put(a)
            elif command == 'look':
                cheat_actions = self.look_action(target_position=target_position, agent_id=agent_id)
                for a in cheat_actions:
                    self.cheat_action_queue[agent_id].put(a)


    def new_simulate(self, obs_queue):
        """continuously simluates the agents' actions using threads"""
        self.print_scene_object_category()
        for i in range(self.num_of_agents):
            self.agent_threads[i].start()
        no_op_actions = dict()
        for i in range(self.num_of_agents):
            no_op_actions[i] = 'no_op'
        obs = self._simulator.step(action=no_op_actions, dt=1/self._fps)
        for i in range(self.num_of_agents):
            self.observation_queue[i].put(obs[i])
        while True:
            actions = dict()
            sgn = dict()
            for i in range(self.num_of_agents):
                sgn[i] = False
            self.handle_cheat_request()
            for agent_id in range(self.num_of_agents):
                action = None
                if not self.action_queue[agent_id].empty():
                    action = self.action_queue[agent_id].get()
                    sgn[i] = True # the agent performs its action in this step
                else: #non-blocking simulation. If an agent doesn't return the action, just replace it with 'no_op'
                    action = 'no_op'
                if action == "grab_release":
                    self.grab_release_action(agent_id)
                    actions[agent_id] = "no_op"
                elif action == "open_close":
                    self.open_close_action(agent_id)
                    actions[agent_id] = "no_op"
                else:
                    actions[agent_id] = action
            self.awake_all_objects()
            observations = self._simulator.step(action=actions, dt=1/self._fps)
            obs = transform_rgb_bgr(observations[0]["rgb_1st"], crosshair=True)
            for i in range(self.num_of_agents):
                if sgn[i]: # only return observations for the agents that perform their actions in this step
                    self.observation_queue[i].put(observations[i])
                self.world_observation_queue[i].put(observations[i]["rgb_1st"][:, :, :3].astype(np.uint8))
            #cv2.imshow("Observations", obs)
            cv2.waitKey(int(1/60*1000))
            obs_queue.put(obs)

    def simulate(self):
        """continuously simluates the agents' actions using threads"""
        self.print_scene_object_category()
        for i in range(self.num_of_agents):
            self.agent_threads[i].start()
        no_op_actions = dict()
        for i in range(self.num_of_agents):
            no_op_actions[i] = 'no_op'
        obs = self._simulator.step(action=no_op_actions, dt=1/self._fps)
        for i in range(self.num_of_agents):
            self.observation_queue[i].put(obs[i])
        while True:
            actions = dict()
            sgn = dict()
            for i in range(self.num_of_agents):
                sgn[i] = False
            self.handle_cheat_request()
            for agent_id in range(self.num_of_agents):
                action = None
                if not self.action_queue[agent_id].empty():
                    action = self.action_queue[agent_id].get()
                    sgn[i] = True # the agent performs its action in this step
                else: #non-blocking simulation. If an agent doesn't return the action, just replace it with 'no_op'
                    action = 'no_op'
                if action == "grab_release":
                    self.grab_release_action(agent_id)
                    actions[agent_id] = "no_op"
                elif action == "open_close":
                    self.open_close_action(agent_id)
                    actions[agent_id] = "no_op"
                else:
                    actions[agent_id] = action
            self.awake_all_objects()
            observations = self._simulator.step(action=actions, dt=1/self._fps)
            obs = transform_rgb_bgr(observations[0]["rgb_1st"], crosshair=True)
            for i in range(self.num_of_agents):
                if sgn[i]: # only return observations for the agents that perform their actions in this step
                    self.observation_queue[i].put(observations[i])
                self.world_observation_queue[i].put(observations[i]["rgb_1st"][:, :, :3].astype(np.uint8))
            #cv2.imshow("Observations", obs)
            cv2.waitKey(int(1/60*1000))


    def load_object(self, object_config_path,
                    translation=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    transformation=None,
                    semantic_id=0,
                    name='unknown',
                    mass=0.05,
                    friction_coefficient=0.5,
                    motion="DYNAMIC",
                    light_setup_key='',
                    scale=1.0) -> Dict:
        """load a rigid object"""
        # Note: semantic ID is deprecated and will be managed by the simulator
        semantic_id = self._sem_cnt
        self._sem_cnt += 1
        self._semantic_id_to_name[semantic_id] = name
        if name in self._name_to_semantic_id:
            self._name_to_semantic_id[name].append(semantic_id)
        else:
            self._name_to_semantic_id[name] = [semantic_id]

        object_template_id = self._obj_template_mgr.load_configs(object_config_path)[0]
        obj = self._rigid_obj_mgr.add_object_by_template_id(
            object_template_id,
            light_setup_key=light_setup_key)
        self._loaded_object_handle.append(obj.handle)
        # obj.collidable = False
        obj.semantic_id = semantic_id
        obj.translation = translation
        obj.mass = mass
        obj.friction_coefficient = friction_coefficient
        # obj.scale = mn.Vector3([scale, scale, scale])
        rotation = quat_to_magnum(quat_from_coeffs(rotation))
        obj.rotation = rotation
        if transformation is not None:
            obj.transformation = transformation
        assert motion in ["DYNAMIC", "STATIC"]
        if motion == "DYNAMIC":
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        else:
            obj.motion_type = habitat_sim.physics.MotionType.STATIC

    def place_marker(self, translation=[0,0,0]):
        object_template_id = self._obj_template_mgr.load_configs('assets/sphere')[0]
        obj = self._rigid_obj_mgr.add_object_by_template_id(
            object_template_id,
            light_setup_key='')
        obj.collidable = False
        obj.translation = translation
        obj.motion_type = habitat_sim.physics.MotionType.STATIC

    def attach_object_to_agent(self, object_path, agent_id=0) -> Dict:
        """attach an rigid object to an agent"""
        if object_path is None:
            return
        object_template_id = self._obj_template_mgr.load_configs(object_path)[0]
        obj = self._rigid_obj_mgr.add_object_by_template_id(object_template_id, self._simulator.agents[agent_id].scene_node)
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        self._agent_object_ids[agent_id] = obj.object_id


    def initialize_agent(self, agent_id=0, position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0, 1.0]):
        """initialize an agent by its position and rotation"""
        agent_state = habitat_sim.AgentState()
        agent_state.position = position
        agent_state.rotation = rotation
        agent = self._simulator.initialize_agent(agent_id, agent_state)
        return agent.scene_node.transformation_matrix()


    def randomly_initialize_agent(self, agent_id=0):
        """randomly initialize an agent"""
        point = self.get_path_finder().get_random_navigable_point(max_tries=10, island_index=0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = point
        agent_state.rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        agent = self._simulator.initialize_agent(agent_id, agent_state)
        return agent.scene_node.transformation_matrix()


    def reconfigure(self, sim_settings, agents_settings):
        """reconfigure"""
        self._resolution = [sim_settings['height'], sim_settings['width']]
        self._fps = sim_settings['fps']
        self.agents = []

        agent_configs = []
        for i, single_agent_settings in enumerate(agents_settings):
            agent_configs.append(make_agent_cfg(self._resolution, single_agent_settings))
            self.agents.append(Agent(self, single_agent_settings, i))

        self.num_of_agents = len(self.agents)
        self._agent_object_ids = [None for _ in range(self.num_of_agents)] #by default agents have no rigid object bodies
        self._config = make_sim_cfg(sim_settings, agent_configs)
        self._simulator.reconfigure(self._config)

        self.grab_and_release_distance = sim_settings['grab_and_release_distance']
        self._obj_template_mgr = self._simulator.get_object_template_manager()
        self._rigid_obj_mgr = self._simulator.get_rigid_object_manager()
        self._default_agent_id = sim_settings["default_agent"]

        for agent_id in range(self.num_of_agents):
            agent_object_path = agents_settings[agent_id]['agent_object']
            self.attach_object_to_agent(agent_object_path, agent_id)


    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode=None) -> float:
        """shortest distance from a to b"""
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], (Sequence, np.ndarray)): #multiple endpoints
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else: #single endpoints
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)]
                )
        else:
            path = episode._shortest_path_cache
        path.requested_start = np.array(position_a, dtype=np.float32)
        self.get_path_finder().find_path(path) #Finds the shortest path between a start point and the closest of a set of end points (in geodesic distance) on the navigation mesh using MultiGoalShortestPath module. Path variable is filled if successful. Returns boolean success.
        if episode is not None:
            episode._shortest_path_cache = path
        return path.geodesic_distance