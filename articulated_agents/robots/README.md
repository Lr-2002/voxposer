Robot Design
==============================

Habitat supports three types of robots: Fetch from Fetch Robotics, Stretch from Hello Robot, and Spot from Boston Dynamics. This readme file details the design of robot modules. In addition, we provide Stretch's design specification as one running example.

---

## Misc

- How to add a new robot

    1. Obtain URDF, [ref](../../robots/README.md)

    2. Inherit `Manipulator` or `MobileManipulator` and create a new class in this folder. Define basic manipulator params. You may refer to how other sims introduce these robots + sensors, ex. ManiSkills,
        - [joint definition](https://github.com/simpler-env/ManiSkill2_real2sim/tree/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37/mani_skill2_real2sim/agents/robots).
        - [initial pose](https://github.com/simpler-env/ManiSkill2_real2sim/blob/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37/mani_skill2_real2sim/envs/custom_scenes/base_env.py#L261-L295)
        - [gripper](https://github.com/simpler-env/ManiSkill2_real2sim/blob/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37/mani_skill2_real2sim/agents/robots/googlerobot.py#L58-L62), the limit is read from URDF
        - [end effector](https://github.com/simpler-env/ManiSkill2_real2sim/blob/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37/mani_skill2_real2sim/agents/configs/google_robot/defaults.py#L141C14-L141C26)

- How to obtain link/joint ID <-> link/joint name in URDF

    ```python
    sim = Simulator(sim_settings, agents_settings)
    mgr = sim._simulator.get_articulated_object_manager()
    robot = mgr.add_articulated_object_from_urdf('./robots/hab_fetch/robots/hab_suction.urdf')
    for i in robot.get_link_ids():
        # link and their child joint
        print(i, '      ', robot.get_link_name(i), '      ', robot.get_link_joint_name(i))
    ```

- Other URDF tips

    - Usually, only joints with `<limit>` filed are meant to be controlled.
    - **Coordinate rotation in habitat**: you may set a rotation for this to the base link in URDF (the `<inertial>` field) just as `robots/hab_suction.urdf` did.

## Robot's Component Design

1. **Robot**
    - `fetch_robot.py` is built upon `mobile_manipulator.py`.
    - `franka_robot.py` is built upon `static_manipulator.py`.
    - `stretch_robot.py` is built upon `mobile_manipulator.py`.
    - `spot_robot.py` is built upon `mobile_manipulator.py`.

1. **Robot Parameters**
    - Robot parameters such as the camera's transformation, end-effector position, control gain, and more are specified in each robot class (e.g., inside `stretch_robot.py`).

1. **Miscellanies**
    - Q: Where is the robot being initialized? A: `habitat/tasks/rearrange/articulated_agent_manager.py` is a class that imports all the robots (e.g., it calls `stretch_robot.py`). `habitat/tasks/rearrange/rearrange_sim.py` is a class that imports `articulated_agent_manager.py` and the robot is being initialized.
    - Q: Where are the robot-related parameters being defined? A: They are defined in each robot class. For example, for Stretch, we define the parameters in `stretch_robot.py`.
    - Q: Which robot should I use? A: It depends on the task. For the clustering environment, we might want to use Stretch due to its small base. On the other hand, for the task that requires climbing up stairs, Spot can whereas Stretch cannot.

## Stretch Specification

In this section, we describe the basic function of Stretch in Habitat as one example.

1. **Control Interface**:
    - **Original Arm Action Space**: We define the action space that jointly controls (1) arm extension (horizontal), (2) arm height (vertical), (3) gripper wrist’s roll, pitch, and yaw, and (4) the camera’s yaw and pitch. The resulting size of the action space is 10.
        - **Arm extension** (size: 4): It consists of 4 motors that extend the arm: `joint_arm_l0` (index 28 in robot interface), `joint_arm_l1` (27), `joint_arm_l2` (26), `joint_arm_l3` (25)
        - **Arm height** (size: 1): It consists of 1 motor that moves the arm vertically: `joint_lift` (23)
        - **Gripper wrist** (size: 3): It consists of 3 motors that control the roll, pitch, and yaw of the gripper wrist: `joint_wrist_yaw` (31),  `joint_wrist_pitch` (39),  `joint_wrist_roll` (40)
        - **Camera** (size 2): It consists of 2 motors that control the yaw and pitch of the camera: `joint_head_pan` (7), `joint_head_tilt` (8)
        - As a result, the original action space is the order of `[joint_arm_l0, joint_arm_l1, joint_arm_l2, joint_arm_l3, joint_lift, joint_wrist_yaw, joint_wrist_pitch, joint_wrist_roll, joint_head_pan, joint_head_tilt]` defined in `habitat/robots/stretch_robot.py`

    - **Exposed Arm Action Space**: In the real hardware, the arm extension is further reduced to a single control motor via an action wrapper ArmRelPosKinematicReducedActionStretch specified in `habitat-lab/habitat/config/benchmark/rearrange/play_stretch.yaml`, and defined in `habitat-lab/habitat/tasks/rearrange/actions/actions.py`
        - **arm_joint_mask**: This specifies the exposed control interface, we set it to be `[1,0,0,0,1,1,1,1,1,1]`, so the motor angles added or reduced in the first motor will roll over to the rest of the three motors (i.e., `joint_arm_l1`, `joint_arm_l2`, `joint_arm_l3`)
        - As a result, the exposed action space is the size of 7.
        - Right now the arm control is kinematically simulated.

    - **Gripper Open-close**: The gripper open-close control is the same as the one in Fetch and Spot. Right now we use MagicGraspAction specified in `habitat-lab/habitat/config/benchmark/rearrange/play_stretch.yaml`

    - **Base Velocity Control**: The base velocity control is the same as the one in Fetch and Spot. Right now we use BaseVelAction specified in `habitat-lab/habitat/config/benchmark/rearrange/play_stretch.yaml`

1. **Camera**:
    - **head_rgb_sensor**
        - width 120
        - height 212
    - **head_depth_sensor**
        - width 120
        - height 212

1. **Interactive Testing**: Using your keyboard and mouse to control a Stretch robot in a ReplicaCAD environment:
    ```bash
    # Pygame for interactive visualization, pybullet for inverse kinematics
    pip install pygame==2.0.1 pybullet==3.0.4

    # Download Stretch asset
    python -m habitat_sim.utils.datasets_download --uids hab_stretch --data-path /path/to/data/

    # Interactive play script
    python examples/interactive_play.py --never-end --cfg /path/to/data/play_stretch.yaml
    ```

    Note that we modified `interactive_play.py` so that it can control Stretch via a keyboard.
    - Key Q and Key 1: jointly control `joint_arm_l0`, `joint_arm_l1`, `joint_arm_l2`, `joint_arm_l3`
    - Key W and Key 2: `joint_lift`
    - Key E and Key 3: `joint_wrist_yaw`
    - Key R and Key 4: `joint_wrist_pitch`
    - Key T and Key 5: `joint_wrist_roll`
    - Key Y and Key 6: `joint_head_pan`
    - Key U and Key 7: `joint_head_tilt`

1. **Unit Test Example**
    - `habitat-lab/test/test_robot_wrapper.py` provides an example of initializing the robot and controlling its joints.
