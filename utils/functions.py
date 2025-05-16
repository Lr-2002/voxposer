import cmath
import dataclasses
import json
import math
import os
import socket
import time
from typing import Any, Dict, List, Optional

import attr
import cv2
import magnum as mn
import numpy as np
import quaternion  # noqa: F401
import quaternion as qt
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R


def quaternion_to_list(q: quaternion.quaternion):
    return q.imag.tolist() + [q.real]


# def quat_from_magnum(quat: mn.Quaternion) -> qt.quaternion:
#     a = qt.quaternion(1, 0, 0, 0)
#     a.real = quat.scalar
#     a.imag = quat.vector
#     return a


def not_none_validator(
    self: Any, attribute: attr.Attribute, value: Optional[Any]
) -> None:
    if value is None:
        raise ValueError(f"Argument '{attribute.name}' must be set")


def rotate_vector_along_axis(vector, axis, radian):
    v = np.asarray(vector)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    r = R.from_rotvec(radian * axis)
    m = r.as_matrix()
    a = np.matmul(m, v.T)
    return a.T

class DatasetJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, quaternion.quaternion):
            return quaternion_to_list(obj)
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        return (
            obj.__getstate__()
            if hasattr(obj, "__getstate__")
            else obj.__dict__
        )


def run_utility_module_in_parallel(component, cmd):
    print('Connecting to '+component+' module...')
    os.system(cmd)


def connect_with_retry(socket_file, max_retries=1000, retry_interval=2):
    retries = 0
    connected = False
    while retries < max_retries and not connected:
        try:
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect(socket_file)
            connected = True
        except FileNotFoundError or ConnectionRefusedError:
            retries += 1
            print(f"Waiting for the connection. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    if connected:
        print("Connection successful!")
        return client_socket
    else:
        print("Connection failed after max retries.")
        return None


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


def save_image_frames(frame, path):
    if not os.path.exists(path):
        os.makedirs(path)
    idx = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    frame_name = f"frame_{idx}"
    np.save(os.path.join(path, frame_name), frame)


def load_image_frames(path):
    num = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    frames = []
    for i in range(num):
        frame = np.load(os.path.join(path, f"frame_{i}.npy"))
        frames.append(frame)
    return frames


if __name__ == '__main__':
    res = rotate_vector_along_axis([3,5,0], [4,4,1], 1.2)
    print(np.linalg.norm(res)**2)