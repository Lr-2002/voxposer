import os
import numpy as np
import plotly.graph_objects as go
import datetime
from transforms3d.quaternions import mat2quat
import sys
import open3d as o3d
import transforms3d as t3d
import cv2
sys.path.append(os.getcwd())
from utils.project_pc import (compute_mapping, filter_mask_with_point,
                              filter_pc_color_with_mask, filter_pc_with_mask)

def draw_pc(pc_arr):
    import os
    disable = os.environ.get("DISABLE_PC", "").lower()
    if disable in ("1", "true", "yes"):  # If disabled, just return
        return
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_arr)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def set_lmp_objects(lmps, objects):
    if isinstance(lmps, dict):
        lmps = lmps.values()
    for lmp in lmps:
        lmp._context = f'objects = {objects}'

def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}'
    else:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_prompt(prompt_fname):
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # get full path to file
    if '/' in prompt_fname:
        prompt_fname = prompt_fname.split('/')
        full_path = os.path.join(curr_dir, 'prompts', *prompt_fname)
    else:
        full_path = os.path.join(curr_dir, 'prompts', prompt_fname)
    # read file
    with open(full_path, 'r') as f:
        contents = f.read().strip()
    return contents

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

def normalize_map(map):
    """normalization voxel maps to [0, 1] without producing nan"""
    denom = map.max() - map.min()
    if denom == 0:
        return map
    return (map - map.min()) / denom

def quaternion_to_direction(quaternion, reference=np.array([0, 0, 1])):
    """
    Convert a quaternion into a rotated direction vector.
    
    Parameters:
    - quaternion: [x, y, z, w], assumed to be normalized.
    - reference: The reference direction vector to rotate (default: [0, 0, 1]).
    
    Returns:
    - A 3D direction vector after applying the quaternion rotation.
    """
    # Ensure inputs are numpy arrays
    q = np.array(quaternion)
    v = np.array(reference)
    
    # Extract quaternion components
    x, y, z, w = q
    
    # Compute the quaternion conjugate
    q_conjugate = np.array([-x, -y, -z, w])
    
    # Convert the vector to a quaternion with w=0
    v_quat = np.array([v[0], v[1], v[2], 0])
    
    # Perform the rotation: q * v_quat * q_conjugate
    def quaternion_multiply(q1, q2):
        # Quaternion multiplication formula
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ])
    
    rotated_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conjugate)
    
    # Extract the rotated vector (x, y, z part of the quaternion)
    rotated_vector = rotated_quat[:3]
    
    return rotated_vector

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [x, y, z, w].
    """
    # Ensure the matrix is 3x3
    assert R.shape == (3, 3), "Input rotation matrix must be 3x3"
    
    # Compute the trace of the matrix
    trace = np.trace(R)
    
    if trace > 0:
        w = np.sqrt(1.0 + trace) / 2.0
        x = (R[2, 1] - R[1, 2]) / (4.0 * w)
        y = (R[0, 2] - R[2, 0]) / (4.0 * w)
        z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    else:
        # Find the largest diagonal element
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            x = np.sqrt(1.0 + 2 * R[0, 0] - trace) / 2.0
            w = (R[2, 1] - R[1, 2]) / (4.0 * x)
            y = (R[0, 1] + R[1, 0]) / (4.0 * x)
            z = (R[0, 2] + R[2, 0]) / (4.0 * x)
        elif R[1, 1] > R[2, 2]:
            y = np.sqrt(1.0 + 2 * R[1, 1] - trace) / 2.0
            w = (R[0, 2] - R[2, 0]) / (4.0 * y)
            x = (R[0, 1] + R[1, 0]) / (4.0 * y)
            z = (R[1, 2] + R[2, 1]) / (4.0 * y)
        else:
            z = np.sqrt(1.0 + 2 * R[2, 2] - trace) / 2.0
            w = (R[1, 0] - R[0, 1]) / (4.0 * z)
            x = (R[0, 2] + R[2, 0]) / (4.0 * z)
            y = (R[1, 2] + R[2, 1]) / (4.0 * z)
    
    # Return quaternion as [x, y, z, w]
    return np.array([w, x, y, z])

def calc_curvature(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    dz = np.gradient(path[:, 2])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)
    curvature = np.sqrt((ddy * dx - ddx * dy)**2 + (ddz * dx - ddx * dz)**2 + (ddz * dy - ddy * dz)**2) / np.power(dx**2 + dy**2 + dz**2, 3/2)
    # convert any nan to 0
    curvature[np.isnan(curvature)] = 0
    return curvature

class IterableDynamicObservation:
    """acts like a list of DynamicObservation objects, initialized with a function that evaluates to a list"""
    def __init__(self, func):
        assert callable(func), 'func must be callable'
        self.func = func
        self._validate_func_output()

    def _validate_func_output(self):
        evaluated = self.func()
        assert isinstance(evaluated, list), 'func must evaluate to a list'

    def __getitem__(self, index):
        def helper():
            evaluated = self.func()
            item = evaluated[index]
            # assert isinstance(item, Observation), f'got type {type(item)} instead of Observation'
            return item
        return helper

    def __len__(self):
        return len(self.func())

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __call__(self):
        static_list = self.func()
        return static_list

class DynamicObservation:
    """acts like dict observation but initialized with a function such that it uses the latest info"""
    def __init__(self, func):
        try:
            assert callable(func) and not isinstance(func, dict), 'func must be callable or cannot be a dict'
        except AssertionError as e:
            print(e)
            import pdb; pdb.set_trace()
        self.func = func
    
    def __get__(self, key):
        evaluated = self.func()
        if isinstance(evaluated[key], np.ndarray):
            return evaluated[key].copy()
        return evaluated[key]
    
    def __getattr__(self, key):
        return self.__get__(key)
    
    def __getitem__(self, key):
        return self.__get__(key)

    def __call__(self):
        static_obs = self.func()
        if not isinstance(static_obs, Observation):
            static_obs = Observation(static_obs)
        return static_obs

class Observation(dict):
    def __init__(self, obs_dict):
        super().__init__(obs_dict)
        self.obs_dict = obs_dict
    
    def __getattr__(self, key):
        return self.obs_dict[key]
    
    def __getitem__(self, key):
        return self.obs_dict[key]

    def __getstate__(self):
        return self.obs_dict
    
    def __setstate__(self, state):
        self.obs_dict = state

def pointat2quat(pointat):
    """
    calculate quaternion from pointat vector
    """
    up = np.array(pointat, dtype=np.float32)
    up = normalize_vector(up)
    rand_vec = np.array([1, 0, 0], dtype=np.float32)
    rand_vec = normalize_vector(rand_vec)
    # make sure that the random vector is close to desired direction
    if np.abs(np.dot(rand_vec, up)) > 0.99:
        rand_vec = np.array([0, 1, 0], dtype=np.float32)
        rand_vec = normalize_vector(rand_vec)
    left = np.cross(up, rand_vec)
    left = normalize_vector(left)
    forward = np.cross(left, up)
    forward = normalize_vector(forward)
    rotmat = np.eye(3).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    quat_wxyz = mat2quat(rotmat)
    return quat_wxyz

def rotation_matrix_to_direction_vectors(rotation_matrix):
    """
    Converts a rotation matrix into normalized direction vectors.

    Args:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        numpy.ndarray: 3x3 array where each column is a normalized direction vector.
    """
    # Validate the rotation matrix
    if rotation_matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix.")
    
    # Extract and normalize each column (already normalized if the matrix is orthogonal)
    direction_vectors = rotation_matrix / np.linalg.norm(rotation_matrix, axis=0)
    
    return direction_vectors

def get_3d_point_cloud(hab_env):
    # points, _, normals, rgbs = hab_env.get_3d_pc(frame='world')
    obs = hab_env.env._simulator.get_sensor_observations()

    detect_object_name = 'sponge' # pepsi_can or sponge
    # points_n_normals = np.hstack((points, normals))
    # points, rgbs, normals = [], [], []
    ret = []
    rgbs_ret = []
    # print(hab_env.env.all_object_names())
    H, W = 0,0
    for tmp in hab_env.camera_triplets_mats:
        cam_rgb, cam_depth, cam_semantic, cam_K, cam_W = tmp
        object_id = hab_env.env.name_to_semantic_id(detect_object_name)
        print(f'{detect_object_name} id is:{object_id}')

        points_rgb = hab_env.depth_to_pc(cam_K, cam_W, obs[cam_depth], obs[cam_rgb][:,:,:3], frame='world')
        points = points_rgb[:,:3]
        rgbs = points_rgb[:,3:]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        cam_normals = np.asarray(pcd.normals)
        # use lookat vector to adjust normal vectors
        flip_indices = np.dot(cam_normals, hab_env.lookat_vectors[cam_rgb]) < 0
        cam_normals[flip_indices] *= -1
    
        normals = cam_normals

        points_n_normals = np.hstack((points, normals))

        rgb, depth, semantic = obs[cam_rgb], obs[cam_depth], obs[cam_semantic]
        # print(semantic.shape)
        mask = semantic == object_id
        # mask_list.append(mask)
        H, W = depth.shape
        mapping, _ = compute_mapping(cam_W, cam_K, points, depth, (W, H))
        # if top_mask.sum() > 0:
        cropped_points_n_normals, cropped_rgb = filter_pc_color_with_mask(points_n_normals, rgbs, mapping, mask)
        # print(f'cropped points shape is {cropped_points_n_normals.shape}')
        ret.append(cropped_points_n_normals)
        rgbs_ret.append(cropped_rgb)
    ret = np.concatenate(ret, axis=0)
    # print(f"ret.shape is {ret.shape}")
    rgbs_ret = np.concatenate(rgbs_ret, axis=0)
    if hab_env.frame == 'base':
        tmp = np.concatenate([ret[:, :3], np.ones((len(ret), 1))], axis=-1).T
        ret[:, :3] = np.matmul(hab_env.base_transformation.inverted(), tmp).T[:, :3]

    # # Example numpy arrays: points and colors
    # points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)  # Shape (N, 3)
    # colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)  # Shape (N, 3), RGB normalized to [0, 1]

    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()

    # Assign points and colors
    point_cloud.points = o3d.utility.Vector3dVector(ret[:, :3])   # Points as (N, 3)
    point_cloud.colors = o3d.utility.Vector3dVector(ret[:, 3:] / 255.0)   # Colors as (N, 3), RGB

    # Save the point cloud to a .pcd file
    o3d.io.write_point_cloud(f"{detect_object_name}.pcd", point_cloud)

    print("Point cloud saved to output.pcd")

def o3d_translate_habitat(tran, rot):
    if isinstance(tran, list):
        tran = np.array(tran)
    if isinstance(rot, list):
        rot = np.array(rot)
    tmp = tran[1]
    tran[1] = -tran[2]
    tran[2] = tmp

    corr_mat = t3d.axangles.axangle2mat([1, 0, 0], np.radians(90))

    R_habitat = np.dot(corr_mat, rot)

    return tran.tolist(), R_habitat.tolist()
def o3d_to_grasp(rot):
    theta = np.pi / 2
    rotation_matrix_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    if isinstance(rot, list):
        rot = np.array(rot)
    return rotation_matrix_x @ rotation_matrix_x @ rotation_matrix_y @ rot

def convert_xyz_to_xzy_rotation_matrix_raw(o3d_rot):
    """
    Converts a rotation matrix from the `xyz` coordinate system to the `xzy` coordinate system.

    Args:
        rotation_matrix_xyz (numpy.ndarray): 3x3 rotation matrix in the `xyz` coordinate system.

    Returns:
        numpy.ndarray: 3x3 rotation matrix in the `xzy` coordinate system.
    """
    # rotation_matrix_xyz[:, 1:3] = rotation_temp[:, :2]
    # rotation_matrix_xyz[:, 0] = rotation_temp[:, 2]
    # rotation_temp[:, 1] = rotation_matrix_xyz[:, 2]
    # rotation_temp[:, 2] = -rotation_matrix_xyz[:, 1]

    # rotation_temp = -rotation_temp

    Rx = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]])
    Ry = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [-1, 0, 0]])
    Rz = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
    fix_matrix = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
                    )
    P = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    fix_matrix = np.dot(Ry, fix_matrix)
    fix_matrix = Rx @ fix_matrix
    # fix_matrix = np.dot(Rz, fix_matrix)
    # fix_matrix = np.dot(Rz, fix_matrix)
    # fix_matrix = np.dot(Rx, fix_matrix)
    habitat_rot = o3d_rot.copy()
    # habitat_rot = P @ habitat_rot @ P.T
    rot_matrix = habitat_rot @ fix_matrix
    # rot_matrix = np.dot(o3d_rot, fix_matrix)
    return rot_matrix

def convert_xyz_to_xzy_rotation_matrix(o3d_rot):
    """
    Converts a rotation matrix from the `xyz` coordinate system to the `xzy` coordinate system.

    Args:
        rotation_matrix_xyz (numpy.ndarray): 3x3 rotation matrix in the `xyz` coordinate system.

    Returns:
        numpy.ndarray: 3x3 rotation matrix in the `xzy` coordinate system.
    """
    # rotation_matrix_xyz[:, 1:3] = rotation_temp[:, :2]
    # rotation_matrix_xyz[:, 0] = rotation_temp[:, 2]
    # rotation_temp[:, 1] = rotation_matrix_xyz[:, 2]
    # rotation_temp[:, 2] = -rotation_matrix_xyz[:, 1]

    # rotation_temp = -rotation_temp

    Rx = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]])
    Ry = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [-1, 0, 0]])
    Rz = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
    fix_matrix = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
                    )
    P = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    fix_matrix = np.dot(Rz.T, fix_matrix)
    # fix_matrix = np.dot(Ry, fix_matrix)
    habitat_rot = o3d_rot.copy()
    habitat_rot = P @ habitat_rot @ P.T
    rot_matrix = np.dot(habitat_rot, fix_matrix)
    rot_matrix = np.dot(o3d_rot, fix_matrix)
    return rot_matrix

def convert_xyz_to_xzy_rotation_matrix_invert(o3d_rot):
    """
    Converts a rotation matrix from the `xyz` coordinate system to the `xzy` coordinate system.

    Args:
        rotation_matrix_xyz (numpy.ndarray): 3x3 rotation matrix in the `xyz` coordinate system.

    Returns:
        numpy.ndarray: 3x3 rotation matrix in the `xzy` coordinate system.
    """
    # rotation_matrix_xyz[:, 1:3] = rotation_temp[:, :2]
    # rotation_matrix_xyz[:, 0] = rotation_temp[:, 2]
    # rotation_temp[:, 1] = rotation_matrix_xyz[:, 2]
    # rotation_temp[:, 2] = -rotation_matrix_xyz[:, 1]

    # rotation_temp = -rotation_temp

    Rx = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]])
    Ry = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [-1, 0, 0]])
    Rz = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
    fix_matrix = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
                    )
    P = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    fix_matrix = np.dot(Ry, fix_matrix)
    # fix_matrix = np.dot(Rx, fix_matrix)
    habitat_rot = o3d_rot.copy()
    habitat_rot = np.dot(habitat_rot, fix_matrix.T)
    # habitat_rot = P @ habitat_rot @ P.T
    # rot_matrix = np.dot(habitat_rot, fix_matrix)
    # rot_matrix = np.dot(o3d_rot, fix_matrix)
    return habitat_rot

def visualize_points(point_cloud, point_colors=None, show=True):
    """visualize point clouds using plotly"""
    if point_colors is None:
        point_colors = point_cloud[:, 2]
    fig = go.Figure(data=[go.Scatter3d(x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
                                    mode='markers', marker=dict(size=3, color=point_colors, opacity=1.0))])
    if show:
        fig.show()
    else:
        # save to html
        fig.write_html('temp_pc.html')
        print(f'Point cloud saved to temp_pc.html')

def _process_llm_index(indices, array_shape):
    """
    processing function for returned voxel maps (which are to be manipulated by LLMs)
    handles non-integer indexing
    handles negative indexing with manually designed special cases
    """
    if isinstance(indices, int) or isinstance(indices, np.int64) or isinstance(indices, np.int32) or isinstance(indices, np.int16) or isinstance(indices, np.int8):
        processed = indices if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, float) or isinstance(indices, np.float64) or isinstance(indices, np.float32) or isinstance(indices, np.float16):
        processed = np.round(indices).astype(int) if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, slice):
        start, stop, step = indices.start, indices.stop, indices.step
        if start is not None:
            start = np.round(start).astype(int)
        if stop is not None:
            stop = np.round(stop).astype(int)
        if step is not None:
            step = np.round(step).astype(int)
        # only convert the case where the start is negative and the stop is positive/negative
        if (start is not None and start < 0) and (stop is not None):
            if stop >= 0:
                processed = slice(0, stop, step)
            else:
                processed = slice(0, 0, step)
        else:
            processed = slice(start, stop, step)
    elif isinstance(indices, tuple) or isinstance(indices, list):
        processed = tuple(
            _process_llm_index(idx, (array_shape[i],)) for i, idx in enumerate(indices)
        )
    elif isinstance(indices, np.ndarray):
        print("[IndexingWrapper] Warning: numpy array indexing was converted to list")
        processed = _process_llm_index(indices.tolist(), array_shape)
    else:
        print(f"[IndexingWrapper] {indices} (type: {type(indices)}) not supported")
        raise TypeError("Indexing type not supported")
    # give warning if index was negative
    if processed != indices:
        print(f"[IndexingWrapper] Warning: index was changed from {indices} to {processed}")
    # print(f"[IndexingWrapper] {idx} -> {processed}")
    return processed

class VoxelIndexingWrapper:
    """
    LLM indexing wrapper that uses _process_llm_index to process indexing
    behaves like a numpy array
    """
    def __init__(self, array):
        self.array = array

    def __getitem__(self, idx):
        return self.array[_process_llm_index(idx, tuple(self.array.shape))]
    
    def __setitem__(self, idx, value):
        self.array[_process_llm_index(idx, tuple(self.array.shape))] = value
    
    def __repr__(self) -> str:
        return self.array.__repr__()
    
    def __str__(self) -> str:
        return self.array.__str__()
    
    def __eq__(self, other):
        return self.array == other
    
    def __ne__(self, other):
        return self.array != other
    
    def __lt__(self, other):
        return self.array < other
    
    def __le__(self, other):
        return self.array <= other
    
    def __gt__(self, other):
        return self.array > other
    
    def __ge__(self, other):
        return self.array >= other
    
    def __add__(self, other):
        return self.array + other
    
    def __sub__(self, other):
        return self.array - other
    
    def __mul__(self, other):
        return self.array * other
    
    def __truediv__(self, other):
        return self.array / other
    
    def __floordiv__(self, other):
        return self.array // other
    
    def __mod__(self, other):
        return self.array % other
    
    def __divmod__(self, other):
        return self.array.__divmod__(other)
    
    def __pow__(self, other):
        return self.array ** other
    
    def __lshift__(self, other):
        return self.array << other
    
    def __rshift__(self, other):
        return self.array >> other
    
    def __and__(self, other):
        return self.array & other
    
    def __xor__(self, other):
        return self.array ^ other
    
    def __or__(self, other):
        return self.array | other
    
    def __radd__(self, other):
        return other + self.array
    
    def __rsub__(self, other):
        return other - self.array
    
    def __rmul__(self, other):
        return other * self.array
    
    def __rtruediv__(self, other):
        return other / self.array
    
    def __rfloordiv__(self, other):
        return other // self.array
    
    def __rmod__(self, other):
        return other % self.array
    
    def __rdivmod__(self, other):
        return other.__divmod__(self.array)
    
    def __rpow__(self, other):
        return other ** self.array
    
    def __rlshift__(self, other):
        return other << self.array
    
    def __rrshift__(self, other):
        return other >> self.array
    
    def __rand__(self, other):
        return other & self.array
    
    def __rxor__(self, other):
        return other ^ self.array
    
    def __ror__(self, other):
        return other | self.array
    
    def __getattribute__(self, name):
        if name == "array":
            return super().__getattribute__(name)
        elif name == "__getitem__":
            return super().__getitem__
        elif name == "__setitem__":
            return super().__setitem__
        else:
            # print(name)
            return super().array.__getattribute__(name)
    
    def __getattr__(self, name):
        return self.array.__getattribute__(name)


#   ********** visualization of grasp pose tools **********
import re
import ast
import base64
import numpy as np
import open3d as o3d

def parse_list(string):
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, string, re.DOTALL)
    if len(matches) != 0:
        actual_list = ast.literal_eval('[' + matches[0] + ']')
        return actual_list
    else:
        return []
    
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
  ''' Author: chenxi-wang
  Create box instance with mesh representation.
  '''
  box = o3d.geometry.TriangleMesh()
  vertices = np.array([[0,0,0],
                        [width,0,0],
                        [0,0,depth],
                        [width,0,depth],
                        [0,height,0],
                        [width,height,0],
                        [0,height,depth],
                        [width,height,depth]])
  vertices[:,0] += dx
  vertices[:,1] += dy
  vertices[:,2] += dz
  triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                        [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                        [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
  box.vertices = o3d.utility.Vector3dVector(vertices)
  box.triangles = o3d.utility.Vector3iVector(triangles)
  return box

def plot_gripper_pro_max(center, R, width, depth, score=1, color=None):
  '''
  Author: chenxi-wang
  
  **Input:**

  - center: numpy array of (3,), target point as gripper center

  - R: numpy array of (3,3), rotation matrix of gripper

  - width: float, gripper width

  - score: float, grasp quality score

  **Output:**

  - open3d.geometry.TriangleMesh
  '''
  x, y, z = center
  height=0.004
  finger_width = 0.004
  tail_length = 0.04
  depth_base = 0.02
  
  if color is not None:
      color_r, color_g, color_b = color
  else:
      color_r = score # red for high score
      color_g = 0
      color_b = 1 - score # blue for low score
  
  left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
  right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
  bottom = create_mesh_box(finger_width, width, height)
  tail = create_mesh_box(tail_length, finger_width, height)

  left_points = np.array(left.vertices)
  left_triangles = np.array(left.triangles)
  left_points[:,0] -= depth_base + finger_width
  left_points[:,1] -= width/2 + finger_width
  left_points[:,2] -= height/2

  right_points = np.array(right.vertices)
  right_triangles = np.array(right.triangles) + 8
  right_points[:,0] -= depth_base + finger_width
  right_points[:,1] += width/2
  right_points[:,2] -= height/2

  bottom_points = np.array(bottom.vertices)
  bottom_triangles = np.array(bottom.triangles) + 16
  bottom_points[:,0] -= finger_width + depth_base
  bottom_points[:,1] -= width/2
  bottom_points[:,2] -= height/2

  tail_points = np.array(tail.vertices)
  tail_triangles = np.array(tail.triangles) + 24
  tail_points[:,0] -= tail_length + finger_width + depth_base
  tail_points[:,1] -= finger_width / 2
  tail_points[:,2] -= height/2

  vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
  vertices = np.dot(R, vertices.T).T + center
  triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
  colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])

  gripper = o3d.geometry.TriangleMesh()
  gripper.vertices = o3d.utility.Vector3dVector(vertices)
  gripper.triangles = o3d.utility.Vector3iVector(triangles)
  gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
  return gripper


def depth_to_point_cloud(K, depth, rgb=None, mask=None):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    x = (i - cx) * depth / fx
    y = (j - cy) * depth / fy
    z = depth
    if mask is not None:
        indices = np.squeeze(np.where(mask == 1))
        x = x[indices[0], indices[1]]
        y = y[indices[0], indices[1]]
        z = z[indices[0], indices[1]]
        points = np.vstack((x, y, z)).T
    else:
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    if rgb is not None:
        if mask is not None:
            indices = np.squeeze(np.where(mask == 1))
            flatten_rgb = rgb[indices[0], indices[1], :]
        else:
            flatten_rgb = rgb.reshape((H*W, 3))
        points = np.hstack((points, flatten_rgb))
    return points
def transform_to_world(points, extri_mat):
    R = extri_mat[:3, :3]
    t = extri_mat[:3, 3]
    pcd_world = (R @ points.T).T - t
    rotation_y_180 = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -1]])
    pcd_world = (rotation_y_180 @ pcd_world.T).T
    return pcd_world
def get_color_points(rgb, depth, cam_param, mask=None, frame='base'):
    intrinsic_matrix = np.squeeze(cam_param['intrinsic_cv'].numpy())

    color_points = depth_to_point_cloud(intrinsic_matrix, depth, rgb, mask)
    if frame == 'base':
        return color_points
    cam2world_gl = np.squeeze(cam_param['cam2world_gl'].numpy())
    color_points[:, :3] = transform_to_world(color_points[:, :3], cam2world_gl)
    return color_points

def cal_object_bbox(points):
    bb_x_min, bb_x_max = np.min(points[:, 0]), np.max(points[:, 0])
    bb_y_min, bb_y_max = np.min(points[:, 1]), np.max(points[:, 1])
    bb_z_min, bb_z_max = np.min(points[:, 2]), np.max(points[:, 2])

    return [bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max]


def transform_rgb_bgr(image, crosshair=False):
    bgr = image[:, :, [2, 1, 0]].copy()
    if crosshair:
        w, h = bgr.shape[:2]
        cx, cy = w // 2, h // 2
        l = max(w * h // 100000, 1)
        thickness = max(w * h // 300000, 1)
        cv2.line(bgr, (cy, cx-l), (cy, cx+l), color=(0,255,0), thickness=thickness)
        cv2.line(bgr, (cy-l, cx), (cy+l, cx), color=(0,255,0), thickness=thickness)
    return bgr

def convert_wxyz2xyzw(quat_wxyz):
        return np.concatenate((quat_wxyz[1:], quat_wxyz[:1]))

def convert_xyzw2wxyz(quat_xyzw):
    return np.concatenate((quat_xyzw[-1:], quat_xyzw[:-1]))

def align_rotation_to_vector(R, normal):
    """
    Adjusts rotation matrix R so that its forward direction aligns with 'normal'.
    Assumes R is a 3x3 rotation matrix.
    """
    # Extract current forward direction (assuming it's the third column of R)
    current_forward = -R[:, 0]  # Z-axis (assuming a typical convention)

    # Compute the rotation axis (cross product)
    axis = np.cross(current_forward, normal)
    axis_norm = np.linalg.norm(axis)
    
    # If already aligned, return the same rotation
    if axis_norm < 1e-6:
        return R  

    # Compute rotation angle (dot product)
    angle = np.arccos(np.clip(np.dot(current_forward, normal) / 
                              (np.linalg.norm(current_forward) * np.linalg.norm(normal)), -1.0, 1.0))

    # Compute rotation matrix using Rodrigues' formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])  # Skew-symmetric cross-product matrix
    I = np.eye(3)
    R_delta = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)  # Rodrigues' formula

    # Apply the rotation to the original matrix
    R_new = R_delta @ R  

    return R_new
