import argparse
import signal
from functools import partial
import threading

import numpy as np
import torch

from aloha_lower.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER2MUJOCO_GRIPPER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
    LEADER_GRIPPER_POSITION_NORMALIZE_FN,
)
from aloha_lower.robot_utils import (
    enable_gravity_compensation,
    disable_gravity_compensation,
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from data_collection.cams.real_cams import get_last_img
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rclpy

from data_collection.config import BaseConfig as bc
from data_collection.config import SharedVars
def opening_ceremony(
    leader_bot_left: InterbotixManipulatorXS,
    leader_bot_right: InterbotixManipulatorXS,
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    # reboot gripper motors, and set operating modes for all motors
    follower_bot_left.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_left.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    leader_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    leader_bot_left.core.robot_set_operating_modes('single', 'gripper', 'position')
    follower_bot_left.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes(
        'single', 'gripper', 'current_based_position'
    )
    leader_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    leader_bot_right.core.robot_set_operating_modes('single', 'gripper', 'position')
    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    torque_on(follower_bot_left)
    torque_on(leader_bot_left)
    torque_on(follower_bot_right)
    torque_on(leader_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [leader_bot_left, follower_bot_left, leader_bot_right, follower_bot_right],
        [start_arm_qpos] * 4,
        moving_time=4.0,
    )
    # move grippers to starting position
    move_grippers(
        [leader_bot_left, follower_bot_left, leader_bot_right, follower_bot_right],
        [LEADER_GRIPPER_JOINT_MID, FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5
    )

def move_all_arms(leader_bot_left, leader_bot_right, follower_bot_left, follower_bot_right):
    torque_on(follower_bot_left)
    torque_on(leader_bot_left)
    torque_on(follower_bot_right)
    torque_on(leader_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [leader_bot_left, follower_bot_left, leader_bot_right, follower_bot_right],
        [start_arm_qpos] * 4,
        moving_time=2.0,
    )
    # move grippers to starting position
    move_grippers(
        [leader_bot_left, follower_bot_left, leader_bot_right, follower_bot_right],
        [LEADER_GRIPPER_JOINT_MID, FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5
    )

def move_one_pair(left_bot, right_bot):
    
    torque_on(left_bot)
    torque_on(right_bot)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [left_bot, right_bot],
        [start_arm_qpos] * 4,
        moving_time=2.0,
    )
    # move grippers to starting position
    move_grippers(
        [left_bot, right_bot],
        [FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5
    )

def press_to_start(
    leader_bot_left: InterbotixManipulatorXS,
    leader_bot_right: InterbotixManipulatorXS,
    gravity_compensation: bool,
) -> None:
    # press gripper to start teleop
    # disable torque for only gripper joint of leader robot to allow user movement
    leader_bot_left.core.robot_torque_enable('single', 'gripper', False)
    leader_bot_right.core.robot_torque_enable('single', 'gripper', False)
    print('Close the grippers to start')
    pressed = False
    while rclpy.ok() and not pressed:
        pressed = (
            (leader_bot_left.core.joint_states.position[6] < LEADER_GRIPPER_CLOSE_THRESH) and
            (leader_bot_right.core.joint_states.position[6] < LEADER_GRIPPER_CLOSE_THRESH)
        )
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)
    if gravity_compensation:
        enable_gravity_compensation(leader_bot_left)
        enable_gravity_compensation(leader_bot_right)
    else:
        torque_off(leader_bot_left)
        torque_off(leader_bot_right)
    print('Started!')



def signal_handler(sig, frame, leader_bot_left, leader_bot_right):
    print('You pressed Ctrl+C!')
    disable_gravity_compensation(leader_bot_left)
    disable_gravity_compensation(leader_bot_right)
    exit(1)




def initialize_bots():


    node = create_interbotix_global_node('aloha')
    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )
    leader_bot_left = InterbotixManipulatorXS(
        robot_model='wx250s',
        robot_name='leader_left',
        node=node,
        iterative_update_fk=False,
    )
    leader_bot_right = InterbotixManipulatorXS(
        robot_model='wx250s',
        robot_name='leader_right',
        node=node,
        iterative_update_fk=False,
    )

    signal.signal(signal.SIGINT, partial(signal_handler, leader_bot_left=leader_bot_left, leader_bot_right=leader_bot_right))

    disable_gravity_compensation(leader_bot_left)
    disable_gravity_compensation(leader_bot_right)

    print("robots_up")

    robot_startup(node)

    return leader_bot_left, leader_bot_right, follower_bot_left, follower_bot_right, node



def initialize_bots_replay():


    node = create_interbotix_global_node('aloha')
    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )

    print("robots_up")

    robot_startup(node)

    # thread = threading.Thread(target=push_state, args=(follower_bot_left, follower_bot_right))
    # thread.start()
    return follower_bot_left, follower_bot_right

def initialize_bots_teleop_sim():


    node = create_interbotix_global_node('aloha')
    leader_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='leader_left',
        node=node,
        iterative_update_fk=False,
    )
    leader_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='leader_right',
        node=node,
        iterative_update_fk=False,
    )

    print("robots_up")

    robot_startup(node)

    # thread = threading.Thread(target=push_state, args=(follower_bot_left, follower_bot_right))
    # thread.start()
    return leader_bot_left, leader_bot_right


def opening_replay(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    # reboot gripper motors, and set operating modes for all motors
    follower_bot_left.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_left.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
  
    follower_bot_left.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes(
        'single', 'gripper', 'current_based_position'
    )
    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    torque_on(follower_bot_left)
  
    torque_on(follower_bot_right)


    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [ follower_bot_left,follower_bot_right],
        [start_arm_qpos] * 4,
        moving_time=4.0,
    )
    # move grippers to starting position
    move_grippers(
        [follower_bot_left, follower_bot_right],
        [LEADER_GRIPPER_JOINT_MID, FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5
    )


def opening_leaders_for_sim(
    leader_bot_left: InterbotixManipulatorXS,
    leader_bot_right: InterbotixManipulatorXS,
) -> None:
    leader_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    leader_bot_left.core.robot_set_operating_modes('single', 'gripper', 'position')
    leader_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    leader_bot_right.core.robot_set_operating_modes('single', 'gripper', 'position')
   

    torque_on(leader_bot_left)

    torque_on(leader_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms(
        [leader_bot_left, leader_bot_right],
        [start_arm_qpos] * 4,
        moving_time=4.0,
    )
    # move grippers to starting position
    move_grippers(
        [leader_bot_left, leader_bot_right],
        [LEADER_GRIPPER_JOINT_MID, FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
        moving_time=0.5
    )



def get_action(bot_left, bot_right, leader:bool, simulation:bool):
    action = np.zeros(14) # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = bot_left.core.joint_states.position[:6]
    action[7:7+6] = bot_right.core.joint_states.position[:6]

    # TODO check if normalisation like this works Gripper actions 
    if simulation:
        if leader:
            action[6] = LEADER2MUJOCO_GRIPPER_JOINT_FN(
                    bot_left.core.joint_states.position[6]
                )
            action[7+6] = LEADER2MUJOCO_GRIPPER_JOINT_FN(
                    bot_right.core.joint_states.position[6]
            )
           
    else:
        if leader:
            action[6] = LEADER2FOLLOWER_JOINT_FN(
                    bot_left.core.joint_states.position[6]
                )
            action[7+6] =  LEADER2FOLLOWER_JOINT_FN(
                    bot_right.core.joint_states.position[6]
                )
        else:
            action[6] = bot_left.core.joint_states.position[6]
        
            action[7+6] = bot_right.core.joint_states.position[6]


    return action

def teleoperation_step(leader_bot_left, leader_bot_right, follower_bot_left, follower_bot_right, gripper_left_command, gripper_right_command, node):
    leader_left_state_joints = leader_bot_left.core.joint_states.position[:6]
    leader_right_state_joints = leader_bot_right.core.joint_states.position[:6]
    follower_bot_left.arm.set_joint_positions(leader_left_state_joints, blocking=False)
    follower_bot_right.arm.set_joint_positions(leader_right_state_joints, blocking=False)
    # sync gripper positions
    gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
    leader_bot_left.core.joint_states.position[6]
    )
    gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
        leader_bot_right.core.joint_states.position[6]
    )
    follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
    follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)
    # sleep DT
    node.get_clock().sleep_for(DT_DURATION)



def step(action , follower_bot_left, follower_bot_right, gripper_left_command, gripper_right_command):
    #print(action.shape)
    state_len = 7
    #print("\n\n\n\n action:" + str(action))
    left_action = action[0][:state_len]
    right_action = action[0][state_len:]
    # print(f"shape of left action {left_action.shape}")
    # print(f"shape of right action {right_action.shape}")
    follower_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
    follower_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
    
    # print(left_action[-1])
    # if float(left_action[-1]) < 0.7:
    #     gripper_left_command.cmd = -0.6213
    # else:
    #     gripper_left_command.cmd = 1.2

    # if float(right_action[-1]) < 0.7:
    #     gripper_right_command.cmd = -0.6213
    # else:
    #     gripper_right_command.cmd = 1.2

    gripper_left_command.cmd = float(left_action[-1])
    gripper_right_command.cmd = float(right_action[-1])
    # print(f"right gripper width: {right_action[-1]}")
            
    follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
    follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)
    #sleep DT
    #get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)
    observation = get_observation(follower_bot_left, follower_bot_right)

    return observation, 0, False



def push_state(bot_left, bot_right):
    while True:
        if SharedVars.NEW_IMAGE_LEFT and SharedVars.NEW_IMAGE_RIGHT and SharedVars.NEW_IMAGES_TOP:
            q_pos = get_action(bot_left, bot_right, False)
            SharedVars.joint_state.append(q_pos)
            SharedVars.NEW_IMAGES_TOP = SharedVars.NEW_IMAGE_LEFT = SharedVars.NEW_IMAGE_RIGHT = False

def get_last_state():
    return SharedVars.joint_state[-1]

def get_observation(bot_left, bot_right, lead:bool=False):
    q_pos = get_action(bot_left, bot_right, lead)#get_last_state()
    images = get_last_img()
    observation = images
    observation["state"] = q_pos
    return observation

def run_robots():
    SharedVars.leader_bot_left, SharedVars.leader_bot_right, SharedVars.follower_bot_left, SharedVars.follower_bot_right, node=initialize_bots()
    opening_ceremony( SharedVars.leader_bot_left, SharedVars.leader_bot_right, SharedVars.follower_bot_left, SharedVars.follower_bot_right)
    gripper_left_command = JointSingleCommand(name='gripper')
    gripper_right_command = JointSingleCommand(name='gripper')
    press_to_start(SharedVars.leader_bot_left, SharedVars.leader_bot_right, False)
    SharedVars.BOT_READY = True
    while True:
        action = torch.zeros((1,14))
        action[0]=torch.tensor(get_action(SharedVars.leader_bot_left, SharedVars.leader_bot_right, leader=True))
        step(action[0], SharedVars.follower_bot_left, SharedVars.follower_bot_right , gripper_left_command, gripper_right_command)
    bc.BOT_READY = False
    robot_shutdown(node)