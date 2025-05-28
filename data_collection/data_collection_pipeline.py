from datetime import datetime
import glob
import os
import threading
import time
import mujoco
import mujoco.viewer
import torch
import numpy as np
from pathlib import Path
#from aloha_lower.constants import DT_DURATION, LEADER2FOLLOWER_JOINT_FN
from cams.real_cams import map_images, LogitechCamController
import mink
import cv2
import shutil
from enum import Enum, auto
from data_collection.mujoco_helper import MujocoController
from interbotix_common_modules.common_robot.robot import (
    robot_shutdown,

)
#from aloha_scripts.real_aloha_example import get_ctrl_id_list
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
#from data_collection.mujoco_helper import get_pair_params_mujoco, mujoco_setup, store_and_capture_cams_mujoco
from data_collection.teleop_helper import *
from data_collection.config import BaseConfig as bc
from data_collection.config import MujocoConfig as mcon
import aloha_lower.real_env as real_envq
from utils.keyboard import KeyManager
class TeleoperationType(Enum):
    JOINT_SPACE = auto()
    TASK_SPACE = auto()


class DataCollectionManager:
    def __init__(
        self,
        task,
        data_dir,
        reward_func,
        simulation= False
      
    ):  
        self.verif_ts = [] #collection of the durations of each collection step
        self.verif_t = time.time()
        self.task = task
        self.step = 0
        self.cam_names = cam_names
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.is_simulation= simulation
        if not simulation:
            self.node = None
            self.leader_bot_left, self.leader_bot_right, self.follower_bot_left, self.follower_bot_right, self.node= initialize_bots()
            self.cam_controller = None
        else:
           self.leader_bot_left, self.leader_bot_right = initialize_bots_teleop_sim()
           self.mj = MujocoController(task)
           self.cam_controller = None

    def reset(self):
        
        if self.is_simulation:
            self.mj.reset()
            move_one_pair(self.leader_bot_left, self.leader_bot_right)
            if self.cam_controller != None:
                del self.cam_controller
        else: 
            bc.STOPEVENT.clear()
            #if self.node != None:
            #     robot_shutdown(self.node)
            #     exit(1)
            # self.leader_bot_left, self.leader_bot_right, self.follower_bot_left, self.follower_bot_right, self.node= initialize_bots()
            #opening_ceremony( self.leader_bot_left, self.leader_bot_right, self.follower_bot_left, self.follower_bot_right)
            move_all_arms( self.leader_bot_left, self.leader_bot_right, self.follower_bot_left, self.follower_bot_right)
            if self.cam_controller != None:
                del self.cam_controller
        
        print("reset complete")
 


    def start_key_listener(self):
        km = KeyManager()
        print("Press 'n' to collect new data or 'q' to quit data collection")
        if self.is_simulation:
            opening_leaders_for_sim(self.leader_bot_left, self.leader_bot_right)
        else:
            opening_ceremony( self.leader_bot_left, self.leader_bot_right, self.follower_bot_left, self.follower_bot_right)
        while km.key != "q":
            if km.key == "n":
                
                print("\nPreparing for new data collection")
                self.reset()
                press_to_start(self.leader_bot_left, self.leader_bot_right, False)
                if not self.is_simulation:
                    self.cam_controller = LogitechCamController()
                    self.gripper_left_command = JointSingleCommand(name='gripper')
                    self.gripper_right_command = JointSingleCommand(name='gripper')
                    self.threads = self.cam_controller.start_capture()
                    
                self.__create_new_recording_dir()
                self.__create_empty_data()
                print("start_capture...")
                print("Start! Press 's' to save collected data or 'd' to discard.")
                self.timestep = 0
                while km.key not in ["s", "d"]:
                    self.__collection_step(self.timestep)
                    self.timestep += 1
                    km.pool()

                else:
                    print("stopping")
                    bc.STOPEVENT.set() 
                    if self.is_simulation:
                        self.mj.terminate()
                        time.sleep(0.1)
                    else:
                        for thread in self.threads:
                            thread.join()
                        # self.left_thread.join()
                        # self.right_thread.join()
                        del self.threads
                        self.threads = []
                    if km.key == "s":
                        print("\nSaving data")

                        self.__save_data()

                        print("Saved!")
                    elif km.key == "d":
                        print("\nDiscarding data")
                        shutil.rmtree(self.record_dir)
                        print("Discarded!")
                    print(np.mean(self.verif_ts))
                    del self.verif_ts
                    del self.leader_joints
                    del self.follower_joints
                    del self.leader_time
                    del self.follower_time
                    self.verif_ts = []
                    bc.right_cam = []
                    bc.left_cam = []
                    bc.top_cam = []
                    print(
                        "Press 'n' to collect new data or 'q' to quit data collection"
                    )

            km.pool()

        print("\nEnding data collection...")
        km.close()


    def __create_new_recording_dir(self):
        self.record_dir = self.data_dir / datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.record_dir.mkdir()

        self.image_dir = self.record_dir / "images"
        self.image_dir.mkdir()
        print("self.record_dir", self.record_dir)
        print("self.image_dir", self.image_dir)
        if self.is_simulation:
            for name in mcon.CAMERA_NAMES:
                device_dir = self.image_dir / f"{name}_orig"
                device_dir.mkdir()
        else:
            for device in bc.LOGITECH_CAM_NAMES:
                device_dir = self.image_dir / f"{device}_orig"
                device_dir.mkdir()

    def __create_empty_data(self):
        self.leader_joints = []
        self.leader_time = []
        self.follower_joints = []
        self.follower_time = []
        

    def real_robot_collection(self):
        #gets and temporary stores joint position data
        leader_time = time.time()
        leader_params = get_action(self.leader_bot_left, self.leader_bot_right,leader=True, simulation=self.is_simulation)

        follower_time = time.time()
        follower_params = get_action(self.follower_bot_left, self.follower_bot_right, leader=False, simulation=self.is_simulation)
        
        self.leader_joints.append(torch.tensor(leader_params))
        self.leader_time.append(leader_time)
        
        self.follower_joints.append(torch.tensor(follower_params))
        self.follower_time.append(follower_time)

        
    def __collection_step(self, timestep: int):
        if self.is_simulation:
            new_t = time.time()

            leader_time = time.time()
            action = get_action(self.leader_bot_left, self.leader_bot_right,leader=True, simulation=self.is_simulation)
            follower_time = time.time()
            observation, _ ,_ ,_,_ =self.mj.step(action)
            self.leader_joints.append(torch.tensor(action))
            self.follower_joints.append(torch.tensor(observation['agent_pos']))
            self.leader_time.append(leader_time)
            self.follower_time.append(follower_time)
            


            avg = new_t - self.verif_t
            self.verif_ts.append(avg)
            self.verif_t = time.time()
            time.sleep(bc.STEPSPEED)

        else :
            new_t = time.time()
            teleoperation_step(self.leader_bot_left, 
                            self.leader_bot_right, 
                            self.follower_bot_left,
                            self.follower_bot_right,
                            self.gripper_left_command,
                            self.gripper_right_command,
                            self.node)
            #using all three cameras as trigger for the collection of new joint data
            if bc.NEW_IMAGE_RIGHT and bc.NEW_IMAGE_LEFT and bc.NEW_IMAGES_TOP:
                self.real_robot_collection()
                #images_times = store_and_capture_cams_real(self.env.image_recorder, self.image_dir,self.timestep)
                bc.NEW_IMAGES_TOP = False
                bc.NEW_IMAGE_LEFT = False
                bc.NEW_IMAGE_RIGHT = False
            avg = new_t - self.verif_t
            self.verif_ts.append(avg)
            self.verif_t = time.time()

    def __save_data(self, cutoff=10):
        # Cutoff the first and last data and saves to disc
        self.leader_joints = self.leader_joints[cutoff:-cutoff]
        self.follower_joints = self.follower_joints[cutoff:-cutoff]

        self.leader_time = self.leader_time[cutoff:-cutoff]
        self.follower_time = self.follower_time[cutoff:-cutoff]

        leader_joint_pos_list = torch.stack(self.leader_joints)
        follower_joint_pos_list = torch.stack(self.follower_joints)
        

        torch.save(leader_joint_pos_list, self.record_dir / "leader_joint_pos.pt")
        torch.save(self.leader_time, self.record_dir / "leader_time.pt")

        torch.save(follower_joint_pos_list, self.record_dir / "follower_joint_pos.pt")
        torch.save(self.follower_time, self.record_dir / "follower_time.pt")

        if self.is_simulation:
            for key in self.mj.cams.keys():
                dir_path = os.path.join(self.image_dir, f"{key}_orig")
                img_num = 0
                imgs = self.mj.cams[key][cutoff:-cutoff]
                for img in imgs:
                    filename = os.path.join(dir_path, f"{img_num}.jpg")
                    cv2.imwrite(filename, img)
                    img_num +=1

        else:
            map_images(self.leader_time, self.image_dir)

#rewards
from data_collection.reward import place_holder


if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent
    cam_names=[str]
    data_collection_manager = DataCollectionManager(
       task=  "transfer_cube",
        # data_dir=Path("/home/simon/collections/Left_to_right_tranfer_single_cube"),
        data_dir=Path("/home/simon/collections/Simulation/cube_transfer_right_2_left_50"),
        reward_func = place_holder,
        simulation= True
       
    )

    data_collection_manager.start_key_listener()