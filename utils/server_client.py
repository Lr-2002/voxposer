import pickle
import socket
import struct
import json

class SocketServer(object):
    def __init__(self, host='127.0.0.1', parent_port=12345):
        self.host = host
        self.parent_port = parent_port
        self.client_socket = None
        self.server_socket = None
        self.initiate()

    def initiate(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.parent_port))
        self.server_socket.listen(1)
        print("Parent is waiting for a connection...")
        tmp, addr = self.server_socket.accept()
        print("Connected by", addr)
        self.client_socket = tmp

    def send(self, data):
        serialized_array = pickle.dumps(data)
        serialized_length = struct.pack('>I', len(serialized_array))  # pack the length of data
        self.client_socket.sendall(serialized_length + serialized_array)  # send length + data

    def receive(self):
        # print('parent recv')
        data_length = self.client_socket.recv(4)  # receive the length of data first
        # if data_length == b'':
        #     return None
        data_length = struct.unpack('>I', data_length)[0]
        data = b''
        # print(f'parent about to recv {data_length}')
        while len(data) < data_length:
            packet = self.client_socket.recv(data_length - len(data))
            if not packet:
                break
            data += packet
        data = pickle.loads(data)
        return data

class SocketServer_Cpp(object):
    def __init__(self, host='127.0.0.1', parent_port=12345):
        self.host = host
        self.parent_port = parent_port
        self.client_socket = None
        self.server_socket = None
        self.initiate()

    def initiate(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.parent_port))
        self.server_socket.listen(1)
        print("Parent is waiting for a connection...")
        tmp, addr = self.server_socket.accept()
        print("Connected by", addr)
        self.client_socket = tmp

    def send(self, data):
        serialized_array = pickle.dumps(data)
        serialized_length = struct.pack('>I', len(serialized_array))  # pack the length of data
        self.client_socket.sendall(serialized_length + serialized_array)  # send length + data

    def receive(self):
        # print('parent recv')
        data_length = self.client_socket.recv(4)  # receive the length of data first
        data_length = struct.unpack('>I', data_length)[0]
        data = b''
        # print(f'parent about to recv {data_length}')
        while len(data) < data_length:
            packet = self.client_socket.recv(data_length - len(data))
            if not packet:
                break
            data += packet
        json_data = json.loads(data.decode('utf-8'))
        # print("Received JSON data:", json_data)

        # Convert JSON to pickle-compatible object
        pickle_data = pickle.dumps(json_data)
        restored_data = pickle.loads(pickle_data)
        data = pickle.loads(data)
        return data
    
class SocketClient(object):
    def __init__(self, host='127.0.0.1', parent_port=12345):
        self.host = host
        self.parent_port = parent_port
        self.client_socket = None
        self.server_socket = None
        self.initiate()

    def initiate(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.parent_port))
        self.client_socket = s

    def send(self, data):
        # print('child send')
        serialized_array = pickle.dumps(data)
        serialized_length = struct.pack('>I', len(serialized_array))  # pack the length of data
        self.client_socket.sendall(serialized_length + serialized_array)  # send length + data

    def receive(self):
        # print('child recv')
        data_length = self.client_socket.recv(4)  # receive the length of data first
        data_length = struct.unpack('>I', data_length)[0]
        # print(f'data length is {data_length}')
        data = b''
        # print(f'child about to recv {data_length}')
        while len(data) < data_length:
            packet = self.client_socket.recv(data_length - len(data))
            if not packet:
                break
            data += packet
        data = pickle.loads(data)
        return data
    
class SocketClient_Cpp(object):
    def __init__(self, host='127.0.0.1', parent_port=12345):
        self.host = host
        self.parent_port = parent_port
        self.client_socket = None
        self.server_socket = None
        self.initiate()

    def initiate(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.parent_port))
        self.client_socket = s

    def send(self, data):
        # print('child send')
        # serialized_array = pickle.dumps(data)
        # serialized_length = struct.pack('>I', len(serialized_array))  # pack the length of data
        # self.client_socket.sendall(serialized_length + serialized_array)  # send length + data
        self.client_socket.sendall(data)

    def receive(self):
        # print('child recv')
        data_length = self.client_socket.recv(4)  # receive the length of data first
        data_length = struct.unpack('>I', data_length)[0]
        # print(f'data length is {data_length}')
        data = b''
        # print(f'child about to recv {data_length}')
        while len(data) < data_length:
            packet = self.client_socket.recv(data_length - len(data))
            if not packet:
                break
            data += packet
        json_data = json.loads(data.decode('utf-8'))
        # print("Received JSON data:", json_data)

        # Convert JSON to pickle-compatible object
        pickle_data = pickle.dumps(json_data)
        data = pickle.loads(pickle_data)
        return data