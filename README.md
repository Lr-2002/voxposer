### 1. Pinocchio and pink installation guidance

#### 1.1 pink installation

follow the guidance of url below to finish pink installation

pink installation: https://github.com/stephane-caron/pink

#### 1.2 pinocchio installation

pinocchio installation (from source): https://stack-of-tasks.github.io/pinocchio/download.html

You need to install pinocchio and pink from urls above and then add pinocchio path into system path as following.

```python
export PYTHONPATH=$PYTHONPATH:/home/robot/workspace/pinocchio/pinocchio_installation/lib/python3.9/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/robot/anaconda3/envs/bigai-eai/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/robot/workspace/pinocchio/pinocchio_installation/lib
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/robot/workspace/pinocchio/pinocchio_installation/lib/pkgconfig
```



You need to change the folder path to your local pinocchio folder path.



### 2. Perception server installation

Perception python script is in voxposer/perception/molmo_sam_server.py. We need to run it firstly to start perception server.

#### 2.1 environement setup

You need to create an environment from voxposer/perception.yaml to run through molmo_sam_server.py. And after you install corresponding environment, you can run python ./molmo_sam_server.py to start perception server.



### 3.GraspNet server installation

#### 3.1 environment setup

Follow the url below to setup graspnet-baseline environment.

https://github.com/kafei123456/graspnet-baseline-cuda11.3

#### 3.2 Demo

In voxposer/graspnet-baseline-cuda11.3 folder, you can run python ./demo.py --checkpoint_path ./checkpoints/checkpoint-rs.tar to start graspnet server.

