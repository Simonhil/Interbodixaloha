import threading
import numpy as np
from collections import deque


class BaseConfig:
    
    JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate", 
    ]

    # Single arm velocity limits, taken from:
    # https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
    VELOCITY_LIMITS = {k: np.pi for k in JOINT_NAMES}
    IMAGE_WIDTH= 680
    IMAGE_HIGHT=680
    END_WIDTH = 224
    END_HIGHT = 224
    FREQ = 0.02
    PHYSICSTIME = 0.005
    STEPSPEED = 0.02
    IMAGE_DUMP = "./trash"
    # LOGITECH_CAM_NAMES = ['CAM_TOP', 'CAM_LEFT', 'CAM_RIGHT']
    LOGITECH_CAM_NAMES = ['CAM_TOP', 'CAM_LEFT', 'CAM_RIGHT']
    # LOGITECH_CAM_NAMES = ['video2', 'video4', 'video0']

    # LOGITECH_CAM_NAMES = []
    REALCAMS = ['cam_high','cam_left_wrist', 'cam_right_wrist']
    SIMCAMS=["wrist_cam_left","wrist_cam_right", "overhead_cam"]
    
class MujocoConfig:
    SIMULATION_TASKS = {"transfer_cube" : "gym_aloha/AlohaTransferCube-v0",
                        "transfer_cube_pos" : "gym_aloha/AlohaTransferCube-v1",
                        "block_stacking":"gym_aloha/AlohaBockStacking-v0",
                        "ball_maze" :"gym_aloha/AlohaBallMaze-v0",
                        "peg_construction" : "gym_aloha/AlohaPegConstruction-v0",
                        "join_blocks" : "gym_aloha/AlohaJoinBlocks-v0",
                        
                        }
    CAMERA_NAMES=["wrist_cam_right", "wrist_cam_left", "overhead_cam"]

class SharedVars:
    STOPEVENT = threading.Event()
    NEW_IMAGES_TOP = False
    NEW_IMAGE_LEFT= False
    NEW_IMAGE_RIGHT = False
    BOT_READY = False

    right_cam=deque(maxlen=6000)
    top_cam=deque(maxlen=6000)
    left_cam=deque(maxlen=6000)

    right_cam_raw=deque(maxlen=1)
    top_cam_raw=deque(maxlen=1)
    left_cam_raw=deque(maxlen=1)
    
    joint_state=[]

    leader_bot_left = None
    leader_bot_right = None
    follower_bot_left = None
    follower_bot_right = None
