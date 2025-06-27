import random
import einops
import torch
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
        env_cls = gym.vector.SyncVectorEnv
        # self.env = env_cls(
        #     [lambda: gym.make(id, disable_env_checker=True, ) for _ in range(1)]
        # )
        self.env =gym.make(id)
        self.cams = {}
        for cam in mj.CAMERA_NAMES:
            self.cams[cam] = []

    def reset(self, seed=None):
        self.env.reset()
        del self.cams
        self.cams = {}
        for cam in mj.CAMERA_NAMES:
            self.cams[cam] = []

    def step(self,action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        #images = self.env.render()
        images = observation["pixels"]
        img =self.env.render()
        self.show_images(img)
        for cam in mj.CAMERA_NAMES:
            image = images[cam]
            #image = self.crop_img(image, cam)
            self.cams[cam].append(image)
        

        #print(observation.keys())
        return observation, reward, terminated, truncated, info

    def terminate(self):
        self.env.close()
        cv2.destroyAllWindows()

    def show_images(self,images):
        # Show with OpenCVnnd
        scale_factor = 0.75
        combined = np.concatenate(list(images.values()), axis=1)
        resized = cv2.resize(combined, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Multi-Camera Views", resized[..., ::-1])  # RGB to BGR

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    


#########################TEST CODE#############################

if __name__ == "__main__":
    
    mc = MujocoController("transfer")
    mc.reset()
    for _ in range(1000):
        action = np.array([0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,0.024 ,0 ,-0.96, 1.16 ,0 ,-0.3 ,0 ,0.024])
        #print(len(action))
        observation, reward, terminated, truncated, info = mc.step(action)
        print(observation['agent_pos'])
        

