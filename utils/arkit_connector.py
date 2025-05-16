import asyncio
import json
import os
import platform
import signal
import socket
import subprocess
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import websockets
from filterpy.kalman import KalmanFilter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


class KalmanFilterWrapper:
    def __init__(self, n_dim_state, n_dim_obs):
        # Initialize the Kalman filter
        self.kf = KalmanFilter(dim_x=n_dim_obs, dim_z=n_dim_state)

        # State transition matrix (6x6), assuming constant velocity model
        self.kf.F = np.eye(6)

        # Measurement function (6x6)
        self.kf.H = np.eye(6)

        # Initial state estimate (6D pose)
        self.kf.x = np.zeros(6)

        # Initial covariance matrix
        self.kf.P = np.eye(6) * 1000  # high uncertainty in initial state

        # Process noise covariance matrix
        self.kf.Q = np.eye(6) * 0.1

        # Measurement noise covariance matrix
        self.kf.R = np.eye(6) * 1.0

    def apply_online(self, observation):
        self.kf.predict()
        self.kf.update(observation)
        return self.kf.x

class ARKitConnector:
    """
    A connector to receive position and rotation data from a connected application.

    ARKit coordinate system (assuming normal phone position)
    -x: left
    -y: backward
    -z: up
    """

    def __init__(self, port=8888, debug=False):
        """
        Initialize the connector with the given port and other parameters.

        Args:
            port (int): The port on which the connector listens.
            debug (bool): Enable debug mode for verbose output.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        self.ip_address = s.getsockname()[0]
        self.port = port
        self.latest_data = {
            "rotation": None,
            "position": None,
            "button": None,
            "toggle": None,
        }
        self.server = None
        self.ping_interval = 20000000000
        self.ping_timeout = 10000000000
        self.connected_clients = set()
        self.debug = debug
        self.reset_position_values = np.array([0.0, 0.0, 0.0])
        self.get_updates = True

    async def _stop_server(self):
        """
        Stop the WebSocket server and close all active connections.
        """
        if self.server is not None:
            # Closing all connected clients
            for websocket in self.connected_clients:
                await websocket.close()
            # Stopping the server
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            print("[INFO] ARKitConnector Stopped")

    def stop(self):
        """
        Stop the ARKitConnector.
        """
        self.loop.call_soon_threadsafe(asyncio.create_task, self._stop_server())


    def reset_position(self):
        """
        Reset the position to the current position, treating it as (0,0,0).
        """
        if self.latest_data["position"] is not None:
            self.reset_position_values += self.latest_data["position"]
            if self.debug:
                print(f"[INFO] Position reset to: {self.reset_position_values}")

    def pause_updates(self):
        """
        Pauses getting updates of data from the connected device.
        """
        self.get_updates = False

    def resume_updates(self):
        """
        Resumes getting updates of data from the connected device.
        """
        self.get_updates = True

    def _kill_process_using_port(self, port):
        """
        Kill the process using the given port on Unix-based systems.

        Args:
            port (int): The port to check for existing processes.
        """
        if platform.system() != "Windows":
            try:
                # Use lsof to find the process using the port
                command = f"lsof -t -i:{port}"
                pid = subprocess.check_output(command, shell=True).strip().decode()
                if pid:
                    os.kill(int(pid), signal.SIGKILL)
                    if self.debug:
                        print(f"[INFO] Killed process with PID {pid} using port {port}")
                else:
                    if self.debug:
                        print(f"[INFO] No process found using port {port}")
            except subprocess.CalledProcessError as e:
                if self.debug:
                    print(f"[ERROR] Failed to kill process using port {port}: {e}")
        else:
            try:
                command = f"netstat -ano | findstr :{port}"
                output = subprocess.check_output(command, shell=True).decode()
                lines = output.strip().splitlines()
                if lines:
                    pid = lines[0].strip().split()[-1]
                    os.system(f'taskkill /PID {pid} /F')
                    if self.debug:
                        print(f"[INFO] Killed process with PID {pid} using port {port}")
                else:
                    if self.debug:
                        print(f"[INFO] No process found using port {port}")
            except subprocess.CalledProcessError as e:
                if self.debug:
                    print(f"[ERROR] Failed to kill process using port {port}: {e}")

    async def _handle_connection(self, websocket, path):
        """
        Handle incoming connections and messages from the application.

        Args:
            websocket: The WebSocket connection.
            path: The URL path of the WebSocket connection.
        """
        print("[INFO] Application connected")
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                if self.get_updates:
                    data = json.loads(message)
                    if 'rotation' in data and 'position' in data:
                        rotation = np.array(data['rotation'])
                        position = np.array(data['position'])
                        self.latest_data["rotation"] = rotation
                        self.latest_data["position"] = np.array([position[0],position[1],position[2]]).astype(float)
                        if self.reset_position_values is not None and self.latest_data["position"].dtype == self.reset_position_values.dtype:
                            self.latest_data["position"] -= self.reset_position_values
                        self.latest_data["button"] = data.get('button', False)
                        self.latest_data["toggle"] = data.get('toggle', False)

                        if self.debug:
                            print(f"[DATA] Rotation: {self.latest_data['rotation']}, Position: {self.latest_data['position']}, Button: {self.latest_data['button']}, Toggle: {self.latest_data['toggle']}")
        except websockets.ConnectionClosed as e:
            print(f"[INFO] Application disconnected: {e}")
        finally:
            self.connected_clients.remove(websocket)

    async def _start(self):
        """
        Start the connector.
        """
        attempt = 0
        max_attempts = 10
        while attempt < max_attempts:
            self._kill_process_using_port(self.port)
            await asyncio.sleep(0.1)
            try:
                print(f"[INFO] ARKitConnector Starting on port {self.port}...")
                self.server = await websockets.serve(self._handle_connection, "0.0.0.0", self.port,ping_interval=self.ping_interval,ping_timeout=self.ping_timeout)
                print(f"[INFO] ARKitConnector Started. Details: \n[INFO] IP Address: {self.ip_address}\n[INFO] Port: {self.port}")
                break  # Exit the loop if server starts successfully
            except OSError as e:
                print(f"[WARNING] Port {self.port} is in use. Trying next port.")
                self.port += 1  # Increment the port number
                attempt += 1
        else:
            raise RuntimeError("Failed to start server on any port. Exceeded maximum attempts.")

        # Start the camera and control loops
        await self.server.wait_closed()

    def start(self):
        """
        Start the ARKitConnector.
        """
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_event_loop).start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start())


    def get_latest_data(self):
        """
        Get the latest received data.

        Returns:
            dict: The latest rotation, position, and grasp data.
        """
        return self.latest_data


class PoseViz:

    def __init__(self):

        # Initial setup for the plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Setting the axes properties
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])

        faces = self.get_box_faces()
        self.box = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
        self.ax.add_collection3d(self.box)

    # Function to define box vertices
    def get_box_faces(self):
        vertices = np.array([
            [0.4, 1, -0.1], [0.4, -1, -0.1], [-0.4, -1, -0.1], [-0.4, 1, -0.1],
            [0.4, 1, 0.1], [0.4, -1, 0.1], [-0.4, -1, 0.1], [-0.4, 1, 0.1]
        ])
        # Define the vertices that compose each of the 6 faces of the box
        faces = [[vertices[i] for i in face] for face in [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 7, 4],
            [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6]
        ]]
        return faces

    # Example function to simulate streaming 6D pose data
    def viz(self, pose):
        # x, y, z, rot_x, rot_y, rot_z
        rotation_matrix = Rotation.from_rotvec(pose[3:]).as_matrix()
        translation_vector = pose[:3]

        # Get box vertices
        faces = self.get_box_faces()

        transformed_faces = []
        for face in faces:
            transformed_face = np.dot(face, rotation_matrix) + translation_vector
            transformed_faces.append(transformed_face)

        self.box.set_verts(transformed_faces)
        plt.draw()
        plt.pause(0.001)  # Pause to allow the plot to update
