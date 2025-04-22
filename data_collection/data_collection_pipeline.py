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
from aloha_lower.constants import DT_DURATION, LEADER2FOLLOWER_JOINT_FN
from cams.real_cams import map_images, LogitechCamController
import mink
import cv2
import shutil
from enum import Enum, auto
from interbotix_common_modules.common_robot.robot import (
    robot_shutdown,

)
#from aloha_scripts.real_aloha_example import get_ctrl_id_list
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
#from data_collection.mujoco_helper import get_pair_params_mujoco, mujoco_setup, store_and_capture_cams_mujoco
from data_collection.teleop_helper import  collection_step, get_action, initialize_bots, move_all_arms, opening_ceremony, press_to_start, signal_handler, step
from data_collection.config import BaseConfig as bc
import aloha_lower.real_env as real_env
from utils.keyboard import KeyManager
class TeleoperationType(Enum):
    JOINT_SPACE = auto()
    TASK_SPACE = auto()


class DataCollectionManager:
    def __init__(
        self,
        xml_path,
        data_dir,
        cam_names,
        reward_func,
        simulation= False
      
    ):  
        self.verif_ts = []
        self.verif_t = time.time()
        self.xml_path = xml_path
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
           raise NotImplementedError

        print("setting up")

        print("setting up bots")
        # self.master_left, self.puppet_left = teleop("left",simulation, True)
        # self.master_right, self.puppet_right = teleop("right",simulation, False)
        print("bots set up")

        
    

    


    def reset(self):
        bc.STOPEVENT.clear()
        #reset real robots
        
        if self.is_simulation:
            raise NotImplementedError
            self.left_thread, self.right_thread = reset(self.master_left, self.puppet_left, self.master_right, self.puppet_right,self.is_simulation, bc.STOPEVENT)
            (self.viewer, self.right_gripper_actuator, self.right_joint_actuator, self.left_gripper_actuator, self.left_joint_actuator,
                    self.posture_task, self.r_ee_task, self.l_ee_task, 
                    self.configuration, self.data_diractuator_ids,
                   self. model, self.data, self.renderer)=mujoco_setup(self.xml_path)
        else: 
            #if self.node != None:
            #     robot_shutdown(self.node)
            #     exit(1)
            # self.leader_bot_left, self.leader_bot_right, self.follower_bot_left, self.follower_bot_right, self.node= initialize_bots()
            #opening_ceremony( self.leader_bot_left, self.leader_bot_right, self.follower_bot_left, self.follower_bot_right)
            move_all_arms( self.leader_bot_left, self.leader_bot_right, self.follower_bot_left, self.follower_bot_right)
            if self.cam_controller != None:
                del self.cam_controller
        
        print("reset complete")
    #currently missing implementation for real cameras
    def start_key_listener(self):
        km = KeyManager()
        print("Press 'n' to collect new data or 'q' to quit data collection")
        opening_ceremony( self.leader_bot_left, self.leader_bot_right, self.follower_bot_left, self.follower_bot_right)
        while km.key != "q":
            if km.key == "n":
                print()
                print("Preparing for new data collection")
                self.reset()
                test_time = time.time()
                self.cam_controller = LogitechCamController()
                print("\n\n\n\n\n\n" + str((time.time() - test_time)))
                press_to_start(self.leader_bot_left, self.leader_bot_right, False)
                self.gripper_left_command = JointSingleCommand(name='gripper')
                self.gripper_right_command = JointSingleCommand(name='gripper')
                if self.is_simulation:
                    raise NotImplementedError
                    self.l_ee_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, "left/target"))
                    self.r_ee_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, "right/target"))
                self.__create_new_recording_dir()
                self.__create_empty_data()
                print("start_capture...")
                self.threads = self.cam_controller.start_capture()
                

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
                        self.viewer.close()
                        del self.model
                        del self.data
                        time.sleep(0.1)
                    else:
                        for thread in self.threads:
                            thread.join()
                        # self.left_thread.join()
                        # self.right_thread.join()
                        self.threads = []
                    if km.key == "s":
                        print()
                        print("Saving data")

                        self.__save_data()

                        print("Saved!")
                    elif km.key == "d":
                        print()
                        print("Discarding data")
                        #shutil.rmtree(self.record_dir)
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

        print()
        print("Ending data collection...")
        km.close()
        #self.__close_hardware_connections()


    def __create_new_recording_dir(self):
        self.record_dir = self.data_dir / datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.record_dir.mkdir()

        self.image_dir = self.record_dir / "images"
        self.image_dir.mkdir()
        print("self.record_dir", self.record_dir)
        print("self.image_dir", self.image_dir)
        if self.is_simulation:
            raise NotImplementedError
            for name in self.cam_names:
                device_dir = self.image_dir / f"{name}_orig"
                device_dir.mkdir()
        else:
            for device in bc.LOGITECH_CAM_NAMES:
                device_dir = self.image_dir / f"{device}_orig"
                device_dir.mkdir()
            
            # for device in self.continuous_devices:
            #     device_dir = self.image_dir / f"{device.name}_orig"
            #     device_dir.mkdir()

    def __create_empty_data(self):
        self.leader_joints = []
        self.leader_time = []
        self.follower_joints = []
        self.follower_time = []
        

    def collection(self):
        timestep = 0
        leader_time = time.time()
        leader_params = get_action(self.leader_bot_left, self.leader_bot_right,leader=True)
        
        if self.is_simulation:
            raise NotImplementedError
            follower_time = time.time()
            follower_params = get_pair_params_mujoco(self.model, self.data)
            

        else:
            follower_time = time.time()
            follower_params = get_action(self.follower_bot_left, self.follower_bot_right, leader=False)
        
        self.leader_joints.append(torch.tensor(leader_params))
        self.leader_time.append(leader_time)
        
        self.follower_joints.append(torch.tensor(follower_params))
        self.follower_time.append(follower_time)
        timestep += 1

        
    def __collection_step(self, timestep: int):
        if self.is_simulation:
            raise NotImplementedError
        #teleoperation to mujoco
            for index in range(6):
                self.data.ctrl[self.left_joint_actuator[index]] = self.master_left.dxl.joint_states.position[index]
                self.data.ctrl[self.right_joint_actuator[index]] = self.master_right.dxl.joint_states.position[index]
            self.data.ctrl[self.left_gripper_actuator] = self.master_left.dxl.joint_states.position[6]
            self.data.ctrl[self.right_gripper_actuator] = self.master_right.dxl.joint_states.position[6]
            mujoco.mj_step(self.model, self.data)  # Step the simulation
            self.viewer.sync()
            self.collection()
            
            mink.move_mocap_to_frame(self.model, self.data, "left/target", "left/gripper", "site")
            mink.move_mocap_to_frame(self.model, self.data, "right/target", "right/gripper", "site")

        else :
            new_t = time.time()
            collection_step(self.leader_bot_left, 
                            self.leader_bot_right, 
                            self.follower_bot_left,
                            self.follower_bot_right,
                            self.gripper_left_command,
                            self.gripper_right_command,
                            self.node)

            if bc.NEW_IMAGE_RIGHT and bc.NEW_IMAGE_LEFT and bc.NEW_IMAGES_TOP:
                self.collection()
                #images_times = store_and_capture_cams_real(self.env.image_recorder, self.image_dir,self.timestep)
                bc.NEW_IMAGES_TOP = False
                bc.NEW_IMAGE_LEFT = False
                bc.NEW_IMAGE_RIGHT = False
            avg = new_t - self.verif_t
            self.verif_ts.append(avg)
            self.verif_t = time.time()
        #time.sleep(bc.FREQ)

    def __save_data(self):
        leader_joint_pos_list = torch.stack(self.leader_joints)
        follower_joint_pos_list = torch.stack(self.follower_joints)
        

        torch.save(leader_joint_pos_list, self.record_dir / "leader_joint_pos.pt")
        torch.save(self.leader_time, self.record_dir / "leader_time.pt")

        torch.save(follower_joint_pos_list, self.record_dir / "follower_joint_pos.pt")
        torch.save(self.follower_time, self.record_dir / "follower_time.pt")

        cam_time_top = []
        cam_time_left = []
        cam_time_right = []

        map_images(self.leader_time, self.image_dir)
    # def __close_hardwarbe_connections(self):
    #     self.follower_gripper.close()
    #     self.follower_arm.close()
    #     self.leader_gripper.close()
    #     self.leader_arm.close()

    #     for device in self.discrete_devices:
    #         device.close()
        
    #     for device in self.continuous_devices:
    #         device.close()

#from real_robot_env.robot.hardware_azure import Azure
#from real_robot_env.robot.hardware_depthai import DepthAI
#from real_robot_env.robot.hardware_realsense import RealSense
#from real_robot_env.robot.hardware_gopro import GoPro
#from real_robot_env.robot.hardware_audio import AudioInterface


#rewards
from data_collection.reward import place_holder


if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent
    cam_names=[str]
    data_collection_manager = DataCollectionManager(
        xml_path= _HERE / 'mujoco_assets' / "box_transfer.xml",
        data_dir=Path("/home/simon/delete"),
        cam_names = bc.LOGITECH_CAM_NAMES,
        reward_func = place_holder,
        simulation= False
       
    )

    data_collection_manager.start_key_listener()