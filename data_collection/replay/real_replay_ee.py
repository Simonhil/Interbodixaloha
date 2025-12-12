import glob
import os
from pathlib import Path
import time
from data_collection.cams.real_cams import LogitechCamController
from data_collection.utils.cartesian_tools import batch_c_matrix_to_joint, batch_convert_6D_vector_to_Transformationmatrix
from interbotix_xs_msgs.msg import JointSingleCommand
import cv2
import imageio
from matplotlib import pyplot as plt
import numpy as np
import torch

from data_collection.config import BaseConfig as bc
from data_collection.teleop_helper import cartesian_step, cartesian_step_onesided_test, get_action, get_observation, initialize_bots_replay, move_one_pair, opening_replay, step

class EeReplayReal:

    def __init__(
        self,
        data_dir,
        leader:bool,
        reward,
        pos
      
    ):
        self.float_names = []
        self.leader = leader
        self.data_dir = data_dir
        self.follower_bot_left, self.follower_bot_right = initialize_bots_replay()
        self.ee_pos_left, self.ee_pos_right= self.unpack(data_dir)

    def get_observation(self):
        return get_observation(self.follower_bot_left, self.follower_bot_right, lead=True)

    def get_action(self):
        return get_action(self.follower_bot_left, self.follower_bot_right)

    def unpack(self, episode_path): 
        ee_pos_left, ee_pos_right = self.unpack_single_param(episode_path)
        return ee_pos_left, ee_pos_right


    def unpack_single_param(self, episode_path):
        if self.leader:
            file_left = os.path.join(episode_path, 'leader_ee_pos_left.pt')
            file_right = os.path.join(episode_path, 'leader_ee_pos_right.pt')
        else:
            file_left = os.path.join(episode_path, 'follower_ee_pos_left.pt')
            file_right = os.path.join(episode_path, 'follower_ee_pos_right.pt')
       
        return torch.load(file_left), torch.load(file_right)


    def move_robot_ee(self, plot):
        cam_controller = LogitechCamController()
        cam_controller.start_capture()

      
        opening_replay(self.follower_bot_left, self.follower_bot_right)
        self.gripper_left_command = JointSingleCommand(name='gripper')
        self.gripper_right_command = JointSingleCommand(name='gripper')
        verif_t = time.time()
        verif_ts = []
        #self.ee_pos = batch_convert_6D_vector_to_Transformationmatrix(self.ee_pos)
        joints_left= batch_c_matrix_to_joint(self.follower_bot_left, self.ee_pos_left)
        joints_right = batch_c_matrix_to_joint(self.follower_bot_right, self.ee_pos_right)
        for i in range(len(self.ee_pos_left)):
            action_left = joints_left[i]
            action_right = joints_right[i]
            #step( action, self.follower_bot_left, self.follower_bot_right, self.gripper_left_command, self.gripper_right_command)
            cartesian_step(action_left, action_right , self.follower_bot_left, self.follower_bot_right,collision_avoidance=False)
            new_t = time.time()
            verif_ts.append(new_t-verif_t)
            verif_t = new_t
            time.sleep(bc.STEPSPEED)  # Control the simulation speed
        #     cv2.imshow("top",observations['images']['cam_high'])
        # if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to break early
        #     pass
        time.sleep(1)
        cv2.destroyAllWindows()
        move_one_pair(self.follower_bot_left, self.follower_bot_right)
        print("\n\n\n mean: " +str(np.mean(verif_ts)))

def single_replay(replay, leader, reward, dir, plot,pos):
    if replay :
        rp = EeReplayReal(
            # xml_path="/home/sihi/Desktop/Bachelor/aloha/mujoco_assets/box_transfer.xml",
            # data_dir="/home/sihi/delete/download/EXAMPLE",
            #xml_path="/home/i53/student/shilber/aloha/mujoco_assets/box_transfer.xml",
            data_dir= dir,
            # data_dir="/home/simonhilber/delete/2025_04_03-09_26_22",
            leader=leader, reward=reward,
            pos=pos)
        

        rp.move_robot_ee(plot)

if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent.parent
    replay = True
    video = True
   
    
    # data_path = "/home/simon/collections/Left_to_right_tranfer_single_cube/2025_04_22-17_58_59"
    data_path = "/home/simon/collections/real/ee_test"
    sub_folder = [sd for sd in os.listdir(data_path) if "2025" in sd]
    sub_folder.sort()
    # print(data_path)
    for sf in sub_folder:
        sf = data_path + "/" + sf
        print("Playing ", sf)
        single_replay(replay, leader=False,  reward=None, dir=sf, plot=False, pos= True)
    # exit(1)
    # generate_all_replay_video(data_path)
    
