import glob
import os
from pathlib import Path
import time
from data_collection.cams.real_cams import LogitechCamController
from data_collection.mujoco_helper import MujocoController
import cv2
import imageio
from matplotlib import pyplot as plt

import numpy as np
import torch

from data_collection.config import BaseConfig as bc



from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata



class JointReplayRaw:

    def __init__(
        self,
        data_dir,
        leader:bool,
        reward,
        task,
        pos
      
    ):
        self.float_names = []
        self.leader = leader
        self.data_dir = data_dir
        self.mc = MujocoController(task)
        self.mc.reset()
        self.jointpositions, self.leader_time = self.unpack(data_dir,pos)


    def unpack(self, episode_path, pos):
        if not pos:
            joints = self.unpack_single_param(episode_path,"joint_vel")
        else:  
            joints = self.unpack_single_param(episode_path,"joint_pos")
        return joints

    def unpack_delta(self):
        """
        Return delta_action and time_line
        """
        leader_states, follower_states, time_line = self.unpack_single_param_delta(
            self.data_dir,"joint_pos")
        delta_action = leader_states[1:, :] - follower_states[:-1, :]
        # delta_action = leader_states[1:, :] - leader_states[:-1, :]

        # self.plot_joints(leader_states, follower_states)  

        return delta_action, time_line

    def unpack_single_param(self, episode_path, param):
        if self.leader:
            file = os.path.join(episode_path, f'leader_{param}.pt')
        else:
            file = os.path.join(episode_path, f'follower_{param}.pt')
        #path = os.path.join(episode_path, "*.pickle")
        timepath =  os.path.join(episode_path, f'leader_time.pt')
            # Keys contained in .pickle:
            # 'joint_state', 'joint_state_velocity', 'des_joint_state', 'des_joint_vel', 'end_effector_pos', 'end_effector_ori', 'des_gripper_width', 'delta_joint_state',
            # 'delta_des_joint_state', 'delta_end_effector_pos', 'delta_end_effector_ori', 'language_description', 'traj_length'
            #pt_file_path = os.path.join(episode_path, file)
        return torch.load(file), torch.load(timepath)

    def unpack_single_param_delta(self, episode_path, param):
        """
        Unpackt the difference of current robot action and joint state
        Return: leader states, follower states, time line

        """
        leader_file = os.path.join(episode_path, f'leader_{param}.pt')
        follower_file = os.path.join(episode_path, f'follower_{param}.pt')   
        time_file =  os.path.join(episode_path, f'leader_time.pt')
        return torch.load(leader_file), torch.load(follower_file), torch.load(time_file)

    def get_image_timestamps(self,folder_path):
        for fname in os.listdir(folder_path):
            if fname.endswith('.jpg'):
                name_without_ext = os.path.splitext(fname)[0]
                float_val = float(name_without_ext)
                self.float_names.append(float_val)
   
    def move_robot_joint(self, plot):
        new_joint_positions = []
        verif_t = time.time()
        verif_ts = []
        for i in range(0,len(self.jointpositions)):
            action_all_joint = self.jointpositions[i]
            print(i)
            #step( action, self.follower_bot_left, self.follower_bot_right, self.gripper_left_command, self.gripper_right_command)
            observation, _, _, _, _= self.mc.step(np.asarray(action_all_joint))
           
            joints= torch.tensor(observation['agent_pos'])
            new_joint_positions.append(joints)
            new_t = time.time()
            verif_ts.append(new_t-verif_t)
            verif_t = new_t
            time.sleep(bc.STEPSPEED)  # Control the simulation speed
        #     cv2.imshow("top",observations['images']['cam_high'])
        # if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to break early
        #     pass
        time.sleep(1)
        cv2.destroyAllWindows()
        self.mc.reset()
        print("\n\n\n mean: " +str(np.mean(verif_ts)))
        if plot:
            self.plot_joints(self.jointpositions, np.array(new_joint_positions))
            plt.show()


    def move_robot_delta_action(self, delta_action, time_line):
        """
        Pi zero compute the diffrence between action and joint state
        This function take the current joint state of the ROBOTS and
        use the action from the DATASET to rollout.
        """
        cam_controller = LogitechCamController()
        cam_controller.start_capture()

        new_joint_positions = []
        verif_t = time.time()
        verif_ts = []

        # Geto observation to set real action
        observations = self.get_observation()
        for i in range(0,len(delta_action)-1):
            action_all_joint = delta_action[i]
            state = observations["state"]
            new_joint_positions.append(state)

            # Sum the delta action to the current joint state
            action = [action_all_joint.numpy() + state]
            observation, _, _, _, _= self.mc.step(action)
            new_t = time.time()
            verif_ts.append(new_t-verif_t)
            verif_t = new_t
            time.sleep(time_line[i+1] - time_line[i])  # Control the simulation speed
        
        # Update the last joint state
        state = torch.tensor(observation['agent_pos'])
        new_joint_positions.append(state)

        time.sleep(1)
        cv2.destroyAllWindows()
        self.mc.reset()
        print("\n\n\n mean: " +str(np.mean(verif_ts)))

        self.plot_joints(self.jointpositions, np.array(new_joint_positions))

        plt.figure()
        plt.plot(delta_action)

        plt.show()
        self.mc.terminate()
 
    def plot_joints(self, first, second):
        # Number of positions in each inner array
       

        # Create subplots
        num_plots = 2
        num_colums = 7
        fig, sup = plt.subplots(num_plots, num_colums, figsize=(6 * num_colums, 4 * num_plots))
        # Plot each position separately
        fig.set_label("")
        fig.tight_layout(pad=4.0, h_pad=20) 
        fig.subplots_adjust(wspace=1, hspace=5)
        for i in range(num_plots):
            for j in range(num_colums):
                index = j + i*num_colums
                sup[i][j].plot(first[:, index], label=f'Joints, Position {index}', marker='o')
                sup[i][j].plot(second[:, index], label=f'Second, Position {index}', marker='s')
                sup[i][j].set_xticks(np.arange(0, len(first), 200))
                sup[i][j].set_title(f'Plot for Position {index}')
                sup[i][j].set_xlabel('Index')
                sup[i][j].set_ylabel('Value')
                sup[i][j].legend()
                sup[i][j].grid(True)
           
        # Adjust layout
        plt.tight_layout()
        
class JointReplayLerobot:

    def __init__(
        self,
        data,
        leader:bool,
        
        task,
        pos
      
    ):
        self.float_names = []
        self.leader = leader
        self.data = data
        self.mc = MujocoController(task)
        self.mc.reset()
        self.jointpositions, self.leader_time = self.unpack(data)


   
   

    def unpack(self, episode_path):
        positions =[]
        for step in self.data:
            if self.leader:
                positions.append(step['action'])
            else:
                positions.append(step['observation.state'])
        #path = os.path.join(episode_path, "*.pickle")
    
            # Keys contained in .pickle:
            # 'joint_state', 'joint_state_velocity', 'des_joint_state', 'des_joint_vel', 'end_effector_pos', 'end_effector_ori', 'des_gripper_width', 'delta_joint_state',
            # 'delta_des_joint_state', 'delta_end_effector_pos', 'delta_end_effector_ori', 'language_description', 'traj_length'
            #pt_file_path = os.path.join(episode_path, file)
        return positions, None

   

   
    def move_robot_joint(self, plot):
        new_joint_positions = []
        verif_t = time.time()
        verif_ts = []
        for i in range(0,len(self.jointpositions)):
            action_all_joint = self.jointpositions[i]
            print(i)
            #step( action, self.follower_bot_left, self.follower_bot_right, self.gripper_left_command, self.gripper_right_command)
            print()
            print(len(action_all_joint))
            observation, _, _, _, _= self.mc.step(np.asarray(action_all_joint))
            print("step")
            joints= torch.tensor(observation['agent_pos'])
            new_joint_positions.append(joints)
            new_t = time.time()
            verif_ts.append(new_t-verif_t)
            verif_t = new_t
            time.sleep(bc.STEPSPEED)  # Control the simulation speed
        #     cv2.imshow("top",observations['images']['cam_high'])
        # if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to break early
        #     pass
        time.sleep(1)
        cv2.destroyAllWindows()
        self.mc.reset()
        print("\n\n\n mean: " +str(np.mean(verif_ts)))
        if plot:
            self.plot_joints(self.jointpositions, np.array(new_joint_positions))
            plt.show()


   
 
    def plot_joints(self, first, second):
        # Number of positions in each inner array
       

        # Create subplots
        num_plots = 2
        num_colums = 7
        fig, sup = plt.subplots(num_plots, num_colums, figsize=(6 * num_colums, 4 * num_plots))
        # Plot each position separately
        fig.set_label("")
        fig.tight_layout(pad=4.0, h_pad=20) 
        fig.subplots_adjust(wspace=1, hspace=5)
        for i in range(num_plots):
            for j in range(num_colums):
                index = j + i*num_colums
                sup[i][j].plot(first[:, index], label=f'Joints, Position {index}', marker='o')
                sup[i][j].plot(second[:, index], label=f'Second, Position {index}', marker='s')
                sup[i][j].set_xticks(np.arange(0, len(first), 200))
                sup[i][j].set_title(f'Plot for Position {index}')
                sup[i][j].set_xlabel('Index')
                sup[i][j].set_ylabel('Value')
                sup[i][j].legend()
                sup[i][j].grid(True)
           
        # Adjust layout
        plt.tight_layout()

def create_img_vector(img_folder_path):
    cam_list = []
    img_paths = glob.glob(os.path.join(img_folder_path, '*.jpg'))
    
    img_paths = sorted(
    img_paths,
    key=lambda path: float(os.path.splitext(os.path.basename(path))[0])
)
    #assert len(img_paths)==trajectory_length, "Number of images does not equal trajectory length!"

    # for img_path in img_paths:
    #     # img = cv2.imread(img_path)
    #     # cv2.imshow("", img)
    #     img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
    #     cam_list.append(img_array)
    #     # print(img_path)
    #     if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to break early
    #         break
    
    cam_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR) for img_path in img_paths]
    cv2.destroyAllWindows()

    return cam_list



# takes the images from img_fir and saves them as video in dir
def make_video(img_dir, name, dir):
    print("img  " + str(img_dir))
    frames = create_img_vector(img_dir)
    imageio.mimsave(f"{dir}/{name}.mp4", np.stack(frames), fps=50)

def single_replay(replay, video, leader, reward, dir, plot,pos, task, raw=True):
    if replay :
        if raw:
            rp = JointReplayRaw(
                # xml_path="/home/sihi/Desktop/Bachelor/aloha/mujoco_assets/box_transfer.xml",
                # data_dir="/home/sihi/delete/download/EXAMPLE",
                #xml_path="/home/i53/student/shilber/aloha/mujoco_assets/box_transfer.xml",
                data_dir= dir,
                # data_dir="/home/simonhilber/delete/2025_04_03-09_26_22",
                leader=leader, reward=reward,
                task=task,
                pos=pos)
        else :
            rp = JointReplayLerobot(data=dir, leader=leader, task=task, pos=pos)
        

        rp.move_robot_joint(plot)
    if video :
        make_video(dir + str("/images/CAM_TOP_orig"), "top",dir)
        # make_video(dir + str ("/images/CAM_LEFT_orig"), "left[100:,:,:]",dir)
        # make_video(dir+ str ("/images/CAM_RIGHT_orig"), "right[100:,:,:]", dir)

def single_replay_delta(replay, video, leader, reward, dir, plot,pos, raw=True):
    if replay :
        if raw:
            rp = JointReplayRaw(
                data_dir= dir,
                leader=leader, 
                reward=reward,
                pos=pos)
        
        delta_action, time_line = rp.unpack_delta()

        rp.move_robot_delta_action(delta_action, time_line)


def generate_all_replay_video(dir):
    sub_dir_list = [d for d in os.listdir(dir) if "2025" in d]

    for sub_dir in sub_dir_list:
        sub_dir = dir + "/" + sub_dir

        make_video(sub_dir + str("/images/CAM_TOP_orig"), "top", sub_dir)
        # make_video(sub_dir + str ("/images/CAM_LEFT_orig"), "left", sub_dir)
        # make_video(sub_dir + str ("/images/CAM_RIGHT_orig"), "right", sub_dir)

def load_raw():
    data_path = "/home/i53/student/shilber/Downloads/Simulation/cube_transfer_right_2_left_50/"
    sub_folder = [sd for sd in os.listdir(data_path) if "2025" in sd]
    sub_folder.sort()
    print(len(sub_folder))
    for sf in sub_folder:
        sf = data_path + "/" + sf
        print("Playing ", sf)
        single_replay(True, video=False, leader=True,  reward=None,task="transfer_cube", dir=sf, plot=False, pos= True)




def load_lerobot():
    repo_id = "simon/aloha_cube_transfer"
    data=LeRobotDataset(repo_id)
    meta_data = LeRobotDatasetMetadata(repo_id)
    print(meta_data)

    return  data
if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent.parent
    data = load_lerobot()
    load_raw()
    #single_replay(True, video=False, leader=True,  reward=None,task="transfer_cube", dir=data, plot=False, pos= True, raw=True)
    
   
    
