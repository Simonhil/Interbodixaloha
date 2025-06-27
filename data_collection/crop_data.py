from datetime import datetime
import glob
import os
from pathlib import Path

import pandas as pd

import cv2

import torch


def get_names_and_ends(path):
    import pandas as pd

    # Excel-Datei laden
    df = df = pd.read_excel(path, engine="odf")

    # Tupel für jede Zeile erstellen
    tupel_liste = [tuple(row) for row in df.itertuples(index=False, name=None)]

    # Ausgabe zur Kontrolle
    return tupel_liste


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
    
    cam_list = [cv2.imread(img_path) for img_path in img_paths]
    cv2.destroyAllWindows()

    return cam_list

def parse_example(episode_path):
    data = {}
    # leader_path = os.path.join(episode_path, 'leader/*.pt')
    # follower_path = os.path.join(episode_path, 'follower/*.pt')
    path = os.path.join(episode_path,'*.pt')
    #path = os.path.join(episode_path, "*.pickle")
    for file in glob.glob(path):

        # Keys contained in .pickle:
        # 'joint_state', 'joint_state_velocity', 'des_joint_state', 'des_joint_vel', 'end_effector_pos', 'end_effector_ori', 'des_gripper_width', 'delta_joint_state',
        # 'delta_des_joint_state', 'delta_end_effector_pos', 'delta_end_effector_ori', 'language_description', 'traj_length'
        #pt_file_path = os.path.join(episode_path, file)
        name = Path(file).stem
        data.update({name : torch.load(file)})
    # for file in glob.glob(episode_path):
    #     name = 'des_' + Path(file).stem
    #     data.update({name : torch.load(file)})
    trajectory_length = len(data[list(data.keys())[0]])
    
    for feature in list(data.keys()):
        for i in range(len(data[feature])):
            data[f'delta_{feature}'] = torch.zeros_like(torch.tensor(data[feature]))
            if i == 0:
                data[f'delta_{feature}'][i] = 0
            else:
                data[f'delta_{feature}'][i] = data[feature][i] - data[feature][i-1]
  
    top_cam_path = os.path.join(episode_path, 'images/overhead_cam_orig')
    wrist_left_cam_path = os.path.join(episode_path, 'images/wrist_cam_left_orig')
    wrist_right_cam_path = os.path.join(episode_path, 'images/wrist_cam_right_orig')
    # top_cam_path = os.path.join(episode_path, 'images/cam_high_orig')
    # wrist_left_cam_path = os.path.join(episode_path, 'images/cam_left_wrist_orig')
    # wrist_right_cam_path = os.path.join(episode_path, 'images/cam_right_wrist_orig')
    top_cam_vector = create_img_vector(top_cam_path)
    wrist_left_cam_vector = create_img_vector(wrist_left_cam_path)
    wrist_right_cam_vector = create_img_vector(wrist_right_cam_path)
    # cam1_image_vector = create_img_vector(cam1_path, trajectory_length)
    # cam2_image_vector = create_img_vector(cam2_path, trajectory_length)
    data.update({
                'image_top': top_cam_vector, 
                'image_wrist_left' : wrist_left_cam_vector, 
                'image_wrist_right' : wrist_right_cam_vector
                })

    return data


def save_data(data, path, cutoff=0, simulation = True):
    # Cutoff the first and last data and saves to disc
    
    print(data.keys())

    leader_joint_pos_list = data['leader_joint_pos'][:cutoff]
    follower_joint_pos_list = data['follower_joint_pos'][:cutoff]



    torch.save(leader_joint_pos_list, Path(path) / "leader_joint_pos.pt")
    #torch.save(self.leader_time, path / "leader_time.pt")

    torch.save(follower_joint_pos_list,Path(path) / "follower_joint_pos.pt")
    #torch.save(self.follower_time, path / "follower_time.pt")
    img_path =os.path.join(path, "images")
    if simulation:
        for key in (k for k in data.keys() if "image" in k):
            
            dir_path = os.path.join(img_path, f"{key}_orig")
            os.makedirs(dir_path)
            img_num = 0
            imgs = data[key][:cutoff]
            for img in imgs:
                filename = os.path.join(dir_path, f"{img_num}.jpg")
                cv2.imwrite(filename, img)
                img_num +=1
        print("done")
    else:
        pass
    #map_images(self.leader_time, self.image_dir)



def create_doc(path, name):
  
    folder_names = [name for name in os.listdir(path)
                    if os.path.isdir(os.path.join(path, name))]
   
    def parse_folder_name(folder_name):
        return datetime.strptime(folder_name, "%Y_%m_%d-%H_%M_%S")
    sorted_folders = sorted(folder_names, key=parse_folder_name)



    df = pd.DataFrame(sorted_folders, columns=["name"])
    output_file = f"{name}croplist.ods"
    output_path=os.path.join("/home/simon/Documents", output_file)
    df.to_excel(output_path, engine="odf", index=False)




if __name__ == "__main__":
    exl_path = "/home/simon/Documents/join_wall_100.0croplist.ods"
    original_path = "/home/simon/collections/Simulations/join_wall_100.0"
    new_path = "/home/simon/collections/Simulations/join_wall_100.0_cropped"
    #create_doc(original_path, "join_wall_100.0")
    crop_list = get_names_and_ends(exl_path)
    for n, i in crop_list:
        print(n)
        print(i)
        name=n
        cutoff=i
        old_episode=os.path.join(original_path, name)
        new_episode = os.path.join(new_path, name)
        os.makedirs(new_episode)
        img_path =os.path.join(new_episode, "images")
        os.makedirs(img_path)
        data = parse_example(old_episode)
        save_data(data, new_episode, cutoff)