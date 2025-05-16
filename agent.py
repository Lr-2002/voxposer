import habitat_sim
import numpy as np
from omegaconf import OmegaConf
import actions
import magnum as mn
import utils.functions as functions
from utils.gpt import ChatBot
from utils.functions import run_utility_module_in_parallel, connect_with_retry
from multiprocessing import Process
import pickle
import cv2
import os


class Instruction:
    def __init__(self, command, target=None):
         # command in ['goto', 'look', 'grab', 'release']
         # target can be either str (object category) or a target position
         self.command = command
         self.target = target


class Planner:
    def __init__(self):
        self.chatbot = ChatBot()
        self.chatbot.reset()
        with open('prompt/gpt_prompt.txt', 'r') as f:
            prompt = f.read()
        reply = self.chatbot.ask(prompt)
        print(reply)

    def plan(self, task_description):
        # using gpt to decompose a task into a list of high_level actions
        high_level_actions = []
        reply = self.chatbot.ask(task_description)
        print(reply)
        reply = reply[0].split('\n')
        for line in reply:
            items = line.split(' ')
            if len(items) >= 2:
                if items[1] in ['goto', 'look']:
                    high_level_actions.append(Instruction(items[1], items[2]))
                if items[1] in ['grab', 'release']:
                    high_level_actions.append(Instruction(items[1]))
        return high_level_actions


class Manipulation:
    def __init__(self, sim, agent_id, high_level=True):
        self.high_level = high_level
        self.agent_id = agent_id
        self._sim = sim

    def grab_release(self):
        return ['grab_release']
    
    def open_close(self):
        return ['open_close']


class Vision:
    def __init__(self, sim, agent_id, action_space, high_level=True, load_model=False):
        self.high_level = high_level
        self.agent_id = agent_id
        self._sim = sim
        self.action_space = action_space
        self.socket_file = 'tmp/vision.sock'
        self.content_file = 'tmp/vision.pkl'
        self.socket = None
        if load_model == True:
            #delete old socket file
            if os.path.exists(self.socket_file):
                os.unlink(self.socket_file)
            with open('modules/vision.txt') as f:
                cmd = f.readline()
            inst = Process(target=run_utility_module_in_parallel, args=('vision', cmd,))
            inst.start()
            self.socket = connect_with_retry(self.socket_file)


    def vqa(self, image, questions):
        content = dict()
        content['questions'] = questions
        content['img'] = image
        with open(self.content_file, 'wb') as f:
            pickle.dump(content, f)
        self.socket.send(b'sent')
        sgn = self.socket.recv(1024).decode('utf-8')
        with open(self.content_file, 'rb') as f:
            ans = pickle.load(f).strip().split('\n')
        print(ans)
        return ans


class Navigation:
    def __init__(self, sim, agent_id, action_space, high_level=True):
        self.high_level = high_level
        self.agent_id = agent_id
        self._sim = sim
        self.action_space = action_space

    
    def goto(self, target_position):
        # atom-action planning using greedy algorithm when knowing the whole navmesh
        # return a list of atom actions to goto the target position
        agent_state = self._sim.get_agent_state(self.agent_id)
        agent_position = agent_state.position
        #get the id of the agent's current nav island
        agent_island = self._sim.get_island(agent_position)
        pathfinder =  self._sim.get_path_finder()
        #project the target position to the agent's nav island 
        target_on_navmesh = pathfinder.snap_point(point=target_position, island_index=agent_island)
        follower = habitat_sim.GreedyGeodesicFollower(
            pathfinder,
            self._sim._simulator.agents[self.agent_id],
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right")
        action_list = follower.find_path(target_on_navmesh)
        if len(action_list) == 1:
            return ['no_op']
        else:
        #the final item in the action_list is 'None', remove it and return the action sequence
            return action_list

    
    def look(self, target_position):
        #atom-action planning of 'look' using greedy algorithm
        #return a list of atom actions (rotate) to look at the target position

        #get the agent's center pixel ray of the rgb_1st camera
        camera_ray = self._sim.unproject(self.agent_id)
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
                degree = self.action_space[action]
                if action == 'turn_left':
                    new_camera_direction = functions.rotate_vector_along_axis(vector=camera_direction, axis=y_axis, radian=degree/180*np.pi)
                if action == 'turn_right':
                    new_camera_direction = functions.rotate_vector_along_axis(vector=camera_direction, axis=y_axis, radian=-degree/180*np.pi)
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
                degree = self.action_space[action]
                if action == 'look_up':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = functions.rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=-degree/180*np.pi)
                if action == 'look_down':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = functions.rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=degree/180*np.pi)
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



class Agent:
    def __init__(self, sim, agent_settings, agent_id) -> None:
        self._sim = sim
        self.agent_id = agent_id
        self.articulated = agent_settings['articulated']
        self.agent_height = agent_settings['agent_height']
        self.agent_radius = agent_settings['agent_radius']

        self.action_space = dict()
        for action in agent_settings['action_space']:
            self.action_space[action] = None
            if action == 'move_forward' or 'move_backward':
                self.action_space[action] = agent_settings['step_size']
            if action == 'turn_left' or 'turn_right':
                self.action_space[action] = agent_settings['turn_angle']
            if action == 'look_up' or 'look_down':
                self.action_space[action] = agent_settings['tilt_angle']
        self.atom_action_buffer = []
        self.high_level_action_buffer = []
        
        allow_gpt = False
        if 'gpt' in agent_settings:
            allow_gpt = agent_settings['gpt']
        
        if allow_gpt == False:
            self.planner = None
        else:
            self.planner = Planner()

        self.navigation = Navigation(self._sim, self.agent_id, self.action_space)
        load_vision_model = agent_settings['vision'] if 'vision' in agent_settings else False
        self.vision = Vision(sim=self._sim, agent_id=self.agent_id, action_space=self.action_space, load_model=load_vision_model)
        self.manipulation = Manipulation(self._sim, self.agent_id)
        self.observations = None


    def translate(self):
        # translate a high level action into a sequence of atom actions, then add the actions to the buffer
        # note: will be only executed when the atom action buffer is empty.  
        if len(self.atom_action_buffer) == 0 and len(self.high_level_action_buffer) > 0:
            instruction = self.high_level_action_buffer.pop(0)
            command, target = instruction.command, instruction.target
            target_position = None
            if instruction.target != None:
                if type(instruction.target) is str:
                    target_position = self._sim.nearest_object_position_with_category(self.agent_id, instruction.target)
                    if command == 'goto' and target == 'cabinet':
                        target_position = [-0.3959591, 0.10927765, -6.63411]
                    if command == 'goto' and target == 'fridge':
                        target_position = [-8.027445, 0.10927765, -3.0552301]
                    if command == 'look' and target == 'kitchen_counter':
                        target_position = [-6.58399, 0.92129, -3.41037]
                    if command == 'look' and target == 'counter':
                        target_position = [-6.58399, 0.92129, -3.41037]
                else:
                    target_position = instruction.target
            print(target_position)
            if command == 'goto':
                action_list = self.navigation.goto(target_position)
            if command == 'look':
                action_list = self.navigation.look(target_position)
            if command == 'grab' or command == 'release':
                action_list = self.manipulation.grab_release()
            if command == 'open' or command == 'close':
                action_list = self.manipulation.open_close()
            if command == 'wait':
                action_list = ['no_op' for i in range(60)]
            self.atom_action_buffer += action_list


    def plan(self, task_description):
        # use the planner to decompose the task into high-level actions
        # add these high-level actions into buffer
        high_level_actions = self.planner.plan(task_description)
        self.high_level_action_buffer += high_level_actions


    def add_high_level_action(self, command, target):
        # add an instruction into the buffer directly
        inst = Instruction(command, target)
        self.high_level_action_buffer.append(inst)


    def add_atom_action(self, action):
        # add an atom action into the buffer directly
        self.atom_action_buffer.append(action)


    def get_next_action(self):
        # get the next atom action
        # automatically translate the pending high-level action
        self.translate()
        if len(self.atom_action_buffer) > 0:
            action = self.atom_action_buffer.pop(0)
            if action is None:
                action = 'no_op'
        else:
            # ne atomic action or high-level action left
            action = 'stop'
        return action


    def print_high_level_action_buffer(self):
        print('buffer:')
        for inst in self.high_level_action_buffer:
            print('    '+inst.command, inst.target)
    
    
    def set_observations(self, observations):
        # set the agent's observations
        self.observations = observations