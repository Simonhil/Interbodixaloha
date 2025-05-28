import cv2
import gymnasium as gym
import numpy as np
import gym_aloha
from dm_control import mujoco
from mujoco import viewer as mj_viewer
from data_collection.config import MujocoConfig as mj
from data_collection.config import BaseConfig as bc

class MujocoController:

    def __init__(
        self,
        task,
      
    ):
        id =  mj.SIMULATION_TASKS[task]
        self.env = gym.make(id)
        
        self.cams = {}
        for cam in mj.CAMERA_NAMES:
            self.cams[cam] = []

    def reset(self):
        self.env.reset()
        del self.cams
        self.cams = {}
        for cam in mj.CAMERA_NAMES:
            self.cams[cam] = []

    def step(self,action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        images = self.env.render()
        self.show_images(images)
        for cam in mj.CAMERA_NAMES:
            image = images[cam][..., ::-1]
            image = self.crop_img(image, cam)
            self.cams[cam].append(image)
      
        return observation, reward, terminated, truncated, info

    def terminate(self):
        self.env.close()
        cv2.destroyAllWindows()

    def show_images(self,images):

        # Show with OpenCV
        scale_factor = 0.75
        combined = np.concatenate(list(images.values()), axis=1)
        resized = cv2.resize(combined, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Multi-Camera Views", resized [..., ::-1])  # RGB to BGR

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    

    def crop_img(self,img, cam_name):
        img = img
        if cam_name == "overhead_cam":
            # img = img[:350,50:500,:]#[80:,50:630,:] #[:,:,:]
            # img = img[50:690, 260:900:, :]
            img = img[:, :, :]
            # img=cv2.resize(img, (420, 340))
            img=cv2.resize(img, (224, 224))
        elif cam_name == "wrist_cam_left":
            img = img[:,:,:]#[:,:,:]
            img=cv2.resize(img, (224, 224))
            
        elif cam_name == "wrist_cam_right":
            img = img[:,:,:]#[:,:,:]
            img=cv2.resize(img, (224, 224))
        else:
            raise NotImplementedError
        return img



#########################TEST CODE#############################

if __name__ == "__main__":
    
    mc = MujocoController("block_stacking")
    mc.reset()
    for _ in range(1000):
        action = np.array([0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,0.024 ,0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,0.024])
        #print(len(action))
        observation, reward, terminated, truncated, info = mc.step(action)
        print(observation['agent_pos'])
        

