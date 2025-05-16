import pickle
import random as rd
import time

import cv2
import magnum as mn
import numpy as np
import quaternion
import scipy
import torch
from PIL import Image
from pynput import keyboard
from scipy.spatial.transform import Rotation

from utils.arkit_connector import ARKitConnector, KalmanFilterWrapper, PoseViz
from utils.functions import save_image_frames, transform_rgb_bgr
from transforms3d.quaternions import mat2quat, quat2mat

from voxposer.utils_main import convert_xyz_to_xzy_rotation_matrix, get_3d_point_cloud
from voxposer.utils_main import *

FORWARD_KEY = "w"
BACKWARD_KEY = 's'
LEFT_KEY = "a"
RIGHT_KEY = "d"
LOOK_UP_KEY = '1'
LOOK_DOWN_KEY = '2'
GRAB_RELEAESE_KEY = 'b'
OPEN_CLOSE_KEY = 'e'
INFO_KEY = 'i'
FINISH = "f"
DEPTH_KEY = "l"

ARM_KEYS = ["t", "y", "u", "i", "o", "p", "["]
ARM_KEYS_REVERSE = ["g", "h", "j", "k", "l", ";", "'"]
ARM_RESET_KEY = "r"

keyboard_status = []
action_pool = ["no_op", "move_forward", "move_backward", "turn_left",
               "turn_right", "look_up", "look_down", "grab_release", 'open_close', 'info']


def on_press(key):
    if key not in keyboard_status:
        keyboard_status.append(key)


def on_release(key):
    if key in keyboard_status:
        keyboard_status.remove(key)


def discrete_keyboard_control(sim, crosshair=False, type='sim'):
    num_of_agents = sim.num_of_agents
    action_dict = dict()
    for id in range(0, num_of_agents):
        action_dict[id] = "no_op"
    if type == 'sim':
        obs = sim.step(action_dict)
    else:
        obs, reward = sim.step(action_dict)
    print(obs)
    rgb_1st = transform_rgb_bgr(obs[0]["rgb_1st"], crosshair)
    rgb_3rd = transform_rgb_bgr(obs[0]["rgb_3rd"], False)
    obs = np.concatenate([rgb_1st, rgb_3rd], axis=1)
    cv2.imshow("Observations", obs)
    while True:
        sgn = False
        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            print("action: FORWARD")
            action = 'move_forward'
        elif keystroke == ord(BACKWARD_KEY):
            print("action: BACKWARD")
            action = 'move_backward'
        elif keystroke == ord(LEFT_KEY):
            print("action: LEFT")
            action = 'turn_left'
        elif keystroke == ord(RIGHT_KEY):
            print("action: RIGHT")
            action = 'turn_right'
        elif keystroke == ord(LOOK_UP_KEY):
            print("action: LOOK UP")
            action = 'look_up'
        elif keystroke == ord(LOOK_DOWN_KEY):
            print("action: LOOK DOWN")
            action = 'look_down'
        elif keystroke == ord(GRAB_RELEAESE_KEY):
            print("action: GRAB_RELEASE")
            action = 'grab_release'
        elif keystroke == ord(OPEN_CLOSE_KEY):
            print("action: GRAB_RELEASE")
            action = 'interact'
        elif keystroke == ord(INFO_KEY):
            print("action: INFO")
            action = 'info'
        elif keystroke == ord(DEPTH_KEY):
            print("action: DEPTH")
            sgn = True
            action = 'no_op'
        elif keystroke == ord(FINISH):
            print("action: FINISH")
            break
        else:
            print("INVALID KEY")
            continue
        action_dict = dict()
        action_dict[0] = action
        for id in range(1, num_of_agents):
            action_dict[id] = rd.choice(action_pool)
        if type == 'sim':
            obs = sim.step(action_dict)
        else:
            obs, reward = sim.step(action_dict)
        rgb_1st = transform_rgb_bgr(obs[0]["rgb_1st"], crosshair)
        rgb_3rd = transform_rgb_bgr(obs[0]["rgb_3rd"], False)
        if sgn:
            camera_matrix, projection_matrix = sim.get_camera_info("depth_1st")
            with open('tmp/hssd_scene_snap_3.pkl', 'wb') as f:
                content = dict()
                content['depth'] = obs[0]["depth_1st"]
                content['rgba'] = obs[0]["rgb_1st"]
                content['semantic'] = obs[0]["semantic_1st"]
                rotation = sim.get_agent_state(
                ).sensor_states['depth_1st'].rotation
                print(sim.get_agent_state().sensor_states['depth_1st'])
                rotation = quaternion.as_rotation_matrix(rotation)
                print('rotation: ', rotation)
                translation = sim.get_agent_state(
                ).sensor_states['depth_1st'].position
                print('translation: ', translation)
                transformation = np.eye(4)
                transformation[0:3, 0:3] = rotation
                transformation[0:3, 3] = translation
                content['transformation'] = transformation
                print('transformation: ', transformation)
                pickle.dump(content, f)
        obs = np.concatenate([rgb_1st, rgb_3rd], axis=1)
        cv2.imshow("Observations", obs)


def continuous_keyboard_control(sim, crosshair=False, type='sim', save_dir=None):
    num_of_agents = sim.num_of_agents
    dt = 1 / sim._fps
    grab_lock = False
    interact_lock = False
    with keyboard.Listener(on_press=on_press, on_release=on_release) as lsn:
        while True:
            action = "no_op"
            keystroke = keyboard_status[0].char if len(
                keyboard_status) > 0 else ""
            if keystroke == GRAB_RELEAESE_KEY:
                interact_lock = False
                if grab_lock:
                    action = "no_op"
                else:
                    print("action: GRAB_RELEASE")
                    action = 'grab_release'
                    grab_lock = True
            elif keystroke == OPEN_CLOSE_KEY:
                grab_lock = False
                if interact_lock:
                    action = "no_op"
                else:
                    print("action: INTERACT")
                    action = 'open_close'
                    interact_lock = True
            else:
                interact_lock = False
                grab_lock = False
                if keystroke == FORWARD_KEY:
                    print("action: FORWARD")
                    action = 'move_forward'
                elif keystroke == BACKWARD_KEY:
                    print("action: BACKWARD")
                    action = 'move_backward'
                elif keystroke == LEFT_KEY:
                    print("action: LEFT")
                    action = 'turn_left'
                elif keystroke == RIGHT_KEY:
                    print("action: RIGHT")
                    action = 'turn_right'
                elif keystroke == LOOK_UP_KEY:
                    print("action: LOOK UP")
                    action = 'look_up'
                elif keystroke == LOOK_DOWN_KEY:
                    print("action: LOOK DOWN")
                    action = 'look_down'
                elif keystroke == INFO_KEY:
                    print("action: INFO")
                    action = 'info'
                elif keystroke == FINISH:
                    print("action: FINISH")
                    break
            action_dict = dict()
            action_dict[0] = action
            for id in range(1, num_of_agents):
                action_dict[id] = rd.choice(action_pool)
            if type == 'sim':
                obs = sim.step(action_dict)
            else:
                obs, reward = sim.step(action_dict)
            rgb_1st = transform_rgb_bgr(obs[0]["rgb_1st"], crosshair)
            rgb_3rd = transform_rgb_bgr(obs[0]["rgb_3rd"], False)
            if save_dir is not None:
                save_image_frames(
                    frame=obs[0]["rgb_1st"][:, :, :3], path=save_dir)
            obs = np.concatenate([rgb_1st, rgb_3rd], axis=1)
            cv2.waitKey(int(dt * 1000))
            cv2.imshow("Observations", obs)


def fetchbot_keyboard_control(env, agent, fps=60, crosshair=False, controller=None, openvla=None, openvla_processor=None, port=1368):
    dt = 1 / fps
    grab_lock = False
    connector = ARKitConnector(port=port)
    connector.start()
    filter = KalmanFilterWrapper(6, 6)
    viz = PoseViz()

    def step_env():
        env._simulator.step_physics(dt)
        agent.update()
        obs = env._simulator.get_sensor_observations()

        # obs = np.concatenate([obs["head"], obs["third"],
        #                      obs["side"], obs['side2'], obs['front']], axis=1)
        obs = np.concatenate([obs["front"], obs["head"],
                            obs["third"], obs["side"], obs["chest"], obs["side2"]], axis=1)
        # print(np.array(agent.base_transformation)[:3, -1])
        cv2.imshow("obs", transform_rgb_bgr(obs, True))
        cv2.waitKey(int(dt * 1000))

    with keyboard.Listener(on_press=on_press, on_release=on_release) as lsn:
        while True:
            try:
                keystroke = keyboard_status[0].char if len(
                    keyboard_status) > 0 else ""
            except:
                continue
            if keystroke == 'z':
                print('action: STOP')
                agent.base_action(0, 0)
            elif keystroke == FORWARD_KEY:
                print("action: FORWARD")
                action = 'move_forward'
                agent.base_action(0.5, 0)
            elif keystroke == BACKWARD_KEY:
                print("action: BACKWARD")
                agent.base_action(-0.5, 0)
            elif keystroke == LEFT_KEY:
                print("action: LEFT")
                agent.base_action(0, 0.5)
            elif keystroke == RIGHT_KEY:
                print("action: RIGHT")
                agent.base_action(0, -0.5)
            elif keystroke in ARM_KEYS:
                id = ARM_KEYS.index(keystroke)
                print("action: ARM", id)
                agent.arm_action(1, id)
            elif keystroke in ARM_KEYS:
                id = ARM_KEYS.index(keystroke)
                print("action: ARM", id)
                agent.arm_action(1, id)
            elif keystroke in ARM_KEYS_REVERSE:
                id = ARM_KEYS_REVERSE.index(keystroke)
                print("action: ARM REVERSE", id)
                agent.arm_action(-1, id)
            elif keystroke == ARM_RESET_KEY:
                print("action: ARM RESET")
                interpolate_arm_control(
                    env, agent, dt, 1, agent.params.arm_init_params)
                controller.gripper_control(1)
            elif keystroke == GRAB_RELEAESE_KEY:
                print("action: MANIPULATE ARMS")
                obj_pos = env.get_object_position(env.object_handle)
                obj_pos = controller.robot.sim_obj.transformation.inverted().transform_point(obj_pos)
                # dummy rotation
                target_pos = np.concatenate([obj_pos, np.array([0, 0, 0])])
                target = controller.manipulate_by_6dof_target(
                    target_pos, local=True, impl='customized', ignore_rot=True)
                seconds = 0.5
                interpolate_arm_control(env, agent, dt, seconds, target)
            elif keystroke == "c" and not grab_lock:
                print("action: GRASP")
                agent.snap_by_id(env._rigid_obj_mgr.get_object_id_by_handle(
                    env.object_handle), force=True)
                grab_lock = True
            elif keystroke == "v" and grab_lock:
                print("action: UNGRASP")
                agent.desnap()
                grab_lock = False
            elif keystroke == FINISH:
                print("action: FINISH")
                break
            elif '1' <= keystroke <= '9':
                if keystroke == '8':
                    controller.gripper_control(0)
                elif keystroke == '9':
                    controller.gripper_control(1)
                else:
                    delta = np.zeros(6)
                    delta[(int(keystroke)-1) %
                          3] = 0.1 if int(keystroke) <= 3 else -0.1
                    target = controller.manipulate_by_6dof_delta(
                        delta, frame='intuitive', impl='pink')
                    seconds = 0.5
                    interpolate_arm_control(env, agent, dt, seconds, target)
            elif keystroke == '`':
                prompt = input('openvla prompt:')
                prompt = prompt[prompt.lower().find('`')+1:]
                print('Running: ', prompt)
                for _ in range(40):
                    img = env._simulator.get_sensor_observations()["head"]
                    img = Image.fromarray(img).convert("RGB")
                    inputs = openvla_processor(prompt, img).to(
                        "cuda:0", dtype=torch.bfloat16)
                    raw_action = openvla.predict_action(
                        **inputs, unnorm_key="fractal20220817_data", do_sample=False)
                    print('raw act:', raw_action)
                    gripper_action = 1 if raw_action[-1] > 0.5 else 0
                    delta = controller._openx_raw_action_process(
                        raw_action, impl='pink')
                    target_qpos = controller.manipulate_by_6dof_delta(
                        delta[:6], frame='intuitive', impl='pink')
                    seconds = 0.5
                    interpolate_arm_control(
                        env, agent, dt, seconds, target_qpos)
                    controller.gripper_control(gripper_action)
                    step_env()
            elif keystroke == '=':
                # teleop mode
                def msg2pose(data, base=None):
                    rotation = Rotation.from_matrix(
                        data['rotation']).as_rotvec()
                    position = np.array(data['position'])
                    pose = np.concatenate([position, rotation])
                    pose = filter.apply_online(pose)
                    button = data['button']
                    toggle = data['toggle']
                    if base is not None:
                        rel_pose = pose - base
                    else:
                        rel_pose = pose
                    return rel_pose, pose, button, toggle
                _, last_pose, _, _ = msg2pose(connector.get_latest_data())
                while True:
                    seconds = 0.05
                    data = connector.get_latest_data()
                    if data['rotation'] is not None:
                        # phone has been connected
                        delta, last_pose, button, toggle = msg2pose(
                            data, last_pose)
                        # Note: I found the following booster works well with teleoperation
                        delta[:3] = delta[:3] * 2.5
                        delta[3:] = delta[3:] * 0.5
                        if button:
                            interpolate_arm_control(
                                env, agent, dt, seconds, agent.params.arm_init_params)
                            continue
                        target_qpos = controller.manipulate_by_6dof_delta(
                            delta, local=True, impl='pink', ignore_rot=False)
                        # In teleoperation mode, we don't want to interpolate the arm control
                        agent.arm_motor_pos = target_qpos
                        controller.gripper_control(int(not toggle))
                    step_env()
            elif keystroke == '-':
                from voxposer.arguments import get_config
                from voxposer.envs.hab_env import VoxPoserHab
                from voxposer.interfaces_hab import setup_LMP
                from voxposer.utils_main import set_lmp_objects
                config = get_config('rlbench')
                voxposer_env = VoxPoserHab(env, agent, controller, frame='base')

                print(f'controller state is {controller.gripper_state}')
                while True:
                    pass

                # target_pose = np.concatenate([translation, ee_quat])
                # target_grasp = np.array([1])

                # for i in range(30):
                #     voxposer_env.apply_action(np.concatenate([target_pose, target_grasp]))

                lmps, lmp_env = setup_LMP(voxposer_env, config, debug=False)
                voxposer_ui = lmps['plan_ui']
                while True:
                    # TODO: voxposer also allows "left", "right" and they seems to be legit only when base frame is used
                    voxposer_env.reset()
                    # from IPython import embed; embed()
                    prompt = input('voxposer prompt:')
                    prompt = prompt[prompt.lower().find('-')+1:]
                    main_object_name = voxposer_env.get_object_names_vlm(prompt)
                    set_lmp_objects(lmps, main_object_name) #voxposer_env.get_object_names())
                    print('Running: ', prompt)
                    voxposer_ui(prompt)
                    step_env()

            step_env()

def interpolate_arm_control(env, agent, dt, seconds, target):
    # 在时间步T内逐渐旋转过去，而不是瞬间移动过去，旋转过程中robot不可控
    # 如果要瞬间移动可以使用agent.arm_motor_pos = target
    curs = np.array(agent.arm_motor_pos)
    diff = target - curs
    T = int(seconds / dt)
    for i in range(T):
        agent.arm_motor_pos = diff * (i + 1) / T + curs
        env._simulator.step_physics(dt)
        agent.update()
        obs = env._simulator.get_sensor_observations()
        obs = np.concatenate([obs["articulated_agent_arm"], obs["head"],
                             obs["third"], obs['side'], obs['front']], axis=1)
        cv2.imshow("obs", transform_rgb_bgr(obs, True))
        cv2.waitKey(int(dt * 1000))
