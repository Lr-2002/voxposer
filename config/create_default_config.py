import habitat_sim
import agent
from habitat_sim.agent.controls.controls import ActuationSpec
from typing import Dict, List, Any
from omegaconf import OmegaConf


sim_settings = {
    "width": 1000,  # Spatial resolution of the observations for all sensors
    "height": 1000,
    "default_agent": 0,
    "allow_sliding": True,
    "seed": 1,
    "scene_dataset_config_file": None,
    "scene": None,  # Scene path
    "default_agent_navmesh": True, # Give the agent the navmesh with the size of its radius and its sensor height 
    "navmesh_include_static_objects": True, # consider static objects for navmesh
    "grab_and_release_distance": 2.0, #The maximum distance for grab_and_release action
    "fps": 60 #how many simulation steps in 1 sec
}

agent_settings = {
    # Sensor settings.
    "articulated": False,
    "agent_object": None,
    "agent_radius": 0.1, # radius of the agent cylinder approximation for navmesh
    "agent_height": 1.5,  # Height of sensors in meters, the same as the agent height for navmesh
    "sensor_front_bias": 0.0,   #bias 
    "hfov": 90, # horizontal field of view (FoV)
    "sensors": ["rgb_1st", "rgb_3rd"], 
    "action_space": ["no_op", "move_forward", "move_backward", "turn_left", "turn_right", "look_up", "look_down"], # Action settings
    "step_size": 0.15, # the distance of move forward every step
    "turn_angle": 5, # the angle of turn left and right every step
    "tilt_angle": 5, # the angle of look up and down every step
    }

locobot_settings ={
    # Sensor settings.
    "articulated": False,
    "agent_object": "assets/locobot_merged",
    "agent_radius": 0.1, # radius of the agent cylinder approximation for navmesh
    "agent_height": 0.55,  # Height of sensors in meters, the same as the agent height for navmesh
    "sensor_front_bias": 0.1,   #bias 
    "hfov": 90, # horizontal field of view (FoV)
    "sensors": ["rgb_1st", "rgb_3rd"], 
    "action_space": ["no_op", "move_forward", "move_backward", "turn_left", "turn_right", "look_up", "look_down"], # Action settings
    "step_size": 0.15, # the distance of move forward every step
    "turn_angle": 5, # the angle of turn left and right every step
    "tilt_angle": 5, # the angle of look up and down every step
    }


fetchbot_settings ={
    # Sensor settings.
    "articulated": True,
    "agent_object": None,
    "agent_radius": 0.3, # radius of the agent cylinder approximation for navmesh
    "agent_height": 1.5,  # Height of sensors in meters, the same as the agent height for navmesh
    "sensor_front_bias": 0.0,   #bias 
    "hfov": 90, # horizontal field of view (FoV)
    "sensors": ["articulated_agent_arm", "head", "third"],
    "action_space": [], # Action settings
    "step_size": 0.15, # the distance of move forward every step
    "turn_angle": 5, # the angle of turn left and right every step
    "tilt_angle": 5, # the angle of look up and down every step
    }

config = sim_settings
conf = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
OmegaConf.save(config=conf, f="default_sim_config.yaml")

config = agent_settings
conf = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
OmegaConf.save(config=conf, f="default_agent_config.yaml")

config= locobot_settings
conf = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
OmegaConf.save(config=conf, f="locobot_config.yaml")

config= fetchbot_settings
conf = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
OmegaConf.save(config=conf, f="config/fetchbot_config.yaml")