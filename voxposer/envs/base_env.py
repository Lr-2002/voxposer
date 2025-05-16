from scipy.spatial.transform import Rotation
import numpy as np
import magnum as mn
import sys
import os
from os.path import join
sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(), 'voxposer'))

class BASE_ENV():
    def __init__(self):
        pass
    '''
    Description: 
        Load objects and scene into simulator. Initialize simulator for completing tasks.
    '''
    def load_task(self):
        assert "You need to implement the load_task function in your env" 
    '''
    Description: 
        Reset simulator and reload all object and update visualizer
    '''
    def reset(self):
        assert "You need to implement the reset function in your env"
    '''
    Description:
        Set simulator resolution of cameras
    '''
    def set_env_resolution(self, set_env_resolution):
        assert "You need to implement the set_env_resolution function in your env"
    '''
    Description:
        Reset all task related variables
    '''
    def _reset_task_variables(self):
        assert "You need to implement the _reset_task_variables function in your env"
    '''
    Description: 
        Reset robotic arm to default position. Need to record initial pose first.
    '''
    def reset_to_default_pose(self):
        assert "You need to implement the reset_to_default_pose function in your env"
    '''
    Description:
        Step action to let robot arm and gripper go to target position.
    '''
    def apply_action(self):
        assert "You need to implement the apply_action function in your env"
    '''
    Description:
        Move end-effector to target position without rotation
    '''
    def move_to_pose(self, pose):
        assert "You need to implement the move_to_pose function in your env"
    '''
    Description:
        Open gripper
    '''
    def open_gripper(self):
        assert "You need to implement the open_gripper function in your env"
    '''
    Description:
        Close gripper
    '''
    def close_gripper(self):
        assert "You need to implement the close_gripper function in your env"
    '''
    Description:
        Set gripper state
    '''
    def set_gripper_state(self):
        assert "You need to implement the set_gripper_state function in your env"
    '''
    Description: 
        Get point cloud of scene from simulator's sensors. From rgbd projects to 3D space or get point cloud directly.
    '''
    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        assert "You need to implement the get_scene_3d_obs function in your env"
    '''
    Description:
        From depth image gets 3D point cloud
    '''
    def depth_to_pc(self):
        assert "You need to implement the depth_to_pc function in your env"
    '''
    Description: 
        Get object names in current scene from simulator. (object handles)
    '''
    def get_object_names(self):
        assert "You need to implement the get_object_names function in your env"
    '''
    Description:
        Get agent end-effector position and rotation
    '''
    def get_ee_pose(self):
        assert "You need to implement the get_ee_pose function in your env"
    '''
    Description:
        Get agent end-effector position
    '''
    def get_ee_pos(self):
        assert "You need to implement the get_ee_pos function in your env"
    '''
    Description:
        Get agent end-effector quat
    '''
    def get_ee_quat(self):
        assert "You need to implement the get_ee_quat function in your env"
    '''
    Description:
        Get all cameras view from simulator and visualize them.
    '''
    def update_cameras_view(self):
        assert "You need to implement the update_camera_view function in your env"

    def o3d_to_grasp(self):
        assert "You need to implement the name2id function in your env"
    '''
    Desctiption:
        Get rotation matrix from position and rotation array
    '''
    def _to_transformation_matrix(self, pos, rot):
        trans = np.eye(4)
        trans[:3, :3] = Rotation.from_quat(rot).as_matrix()
        trans[:3, -1] = pos
        return trans
    '''
    Description:
        normalize a vector to unit vector
    '''
    def normalize_vector(x, eps=1e-6):
        """normalize a vector to unit length"""
        x = np.asarray(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            return np.zeros_like(x) if norm < eps else (x / norm)
        elif x.ndim == 2:
            norm = np.linalg.norm(x, axis=1)  # (N,)
            normalized = np.zeros_like(x)
            normalized[norm > eps] = x[norm > eps] / norm[norm > eps][:, None]
            return normalized
    


    

