from typing import Callable
import functools
import os

from hydra import compose, initialize
import hydra

from typing import Optional

from accelerate import Accelerator

from safetensors.torch import load_model

from flower_vla.agents.lang_encoders.florence_tokens import TokenVLM
from flower_vla.dataset.oxe.transforms import generate_policy_prompt, get_action_space_index
from flower_vla.agents.utils.action_index import ActionIndex
from flower_vla.dataset.utils.frequency_mapping import DATASET_FREQUENCY_MAP

import numpy as np
import torch
import tensorflow as tf

from einops import rearrange

import json

# from real_robot_sim import RealRobot
print('successfully import real robot sim')



class VLPWrapper:
    def __init__(self, 
                 saved_model_base_dir, 
                 saved_model_path, 
                 act_min_max_path, 
                 language_instruction,
                 device: str = "cuda",
                 pred_action_horizon: int = 100,
                 replan_after_nsteps: int = 10,
                 ):
        self.saved_model_base_dir = saved_model_base_dir
        self.saved_model_path = saved_model_path
        self.device = device

        with open(act_min_max_path) as f:
            d = json.load(f)
            self.min_values = np.array(d['action']['p01'])
            self.max_values = np.array(d['action']['p99'])
            self.proprio_min_values = torch.tensor(d['proprio']['p01']).to(device)
            self.proprio_max_values = torch.tensor(d['proprio']['p99']).to(device)
            print("loaded action min max values")

        self.lang_embed_model = TokenVLM("microsoft/Florence-2-base")

        file_path = os.path.dirname(os.path.abspath(__file__))
        weights_path_relative = os.path.relpath(saved_model_base_dir, file_path)

        with initialize(config_path=os.path.join(weights_path_relative, ".hydra")):
            cfg = compose(config_name="config")


        cfg.trainer.agent.agent.act_window_size = pred_action_horizon
        cfg.trainer.agent.agent.multistep = replan_after_nsteps

        agent = hydra.utils.instantiate(cfg.trainer.agent, device=device, process_id=0)

        accelerator = Accelerator()
        agent = accelerator.prepare(agent)
        checkpoint_path = os.path.join(saved_model_base_dir, saved_model_path)
        print(f"checkpoint path: {checkpoint_path}")
        missing, unexpected = load_model(agent, os.path.join(checkpoint_path, 'model.safetensors'))

        print(missing)
        print(unexpected)

        # TODO: check if we need to specify the proprio flag
        # agent.agent.use_proprio = True
        agent.to(dtype=torch.bfloat16)
        agent.eval()

        print("Model loaded successfully")


        self.agent = agent
        self.observation = None
        self.task_description = None
        self.task_description_embedding = None
        
        self.format_instruction = functools.partial(
                             generate_policy_prompt,
                             robot_name="Franka Panda",
                             action_space="joint position",
                             num_arms="1",
                             prompt_style='minimal')

        language_instruction = self.format_instruction(language_instruction) 
        self._initialize_task_description(language_instruction)

        self.action_space_index = torch.tensor([get_action_space_index('JOINT_POS', 1,
                                                                       'position', return_tensor=False)])

        self.image_size = 224

    def rescale_to_range(self, value):
        max_values = self.max_values
        min_values = self.min_values
        new_min = -np.ones_like(value)
        new_max = np.ones_like(value)
        rescaled_tensor = (value - new_min) / (new_max - new_min) * (max_values - min_values) + min_values
        return rescaled_tensor

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def _initialize_task_description(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            # print("task description: ", task_description)
            self.task_description = task_description
            self.task_description_embedding = self.lang_embed_model([self.task_description])
            self.task_description_embedding['input_ids'] = \
                self.task_description_embedding['input_ids']
            self.task_description_embedding['attention_mask'] = \
                self.task_description_embedding['attention_mask']
        else:
            self.task_description = ""
            self.task_description_embedding = tf.zeros((512,), dtype=tf.float32)

    def reset(self) -> None:
        #TODO: call the reset function for the underlying agent
        self.agent.agent.reset()


    def predict(self, observation: dict) -> tuple[
        dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """

        image_primary = observation["images_top"]
        image_secondary = observation["images_wrist_left"]
        image_wrist = observation["images_wrist_right"]

        # image_primary = observation["ORB_0_image"]
        # image_secondary = observation["ORB_1_image"]

        assert image_primary.dtype == np.uint8
        assert image_secondary.dtype == np.uint8
        assert image_wrist.dtype == np.uint8

        image_primary = torch.from_numpy(self._resize_image(image_primary)).unsqueeze(0).unsqueeze(0).to(self.device)
        image_secondary = torch.from_numpy(self._resize_image(image_secondary)).unsqueeze(0).unsqueeze(0).to(self.device)
        image_wrist = torch.from_numpy(self._resize_image(image_wrist)).unsqueeze(0).unsqueeze(0).to(self.device)

        image_primary = rearrange(image_primary, 'b l h w c-> b l c h w')
        image_secondary = rearrange(image_secondary, 'b l h w c-> b l c h w')
        image_wrist = rearrange(image_wrist, 'b l h w c-> b l c h w')


        input_observation = {
            "image_primary": image_primary,
            "image_secondary": image_secondary,
            "image_wrist": image_wrist,
            "pad_mask_dict": {"image_primary": torch.ones(image_primary.shape[0], 1).bool().to(device=self.device),
                              "image_seconday": torch.ones(image_secondary.shape[0], 1).bool().to(device=self.device),
                              "image_wrist": torch.ones(image_wrist.shape[0], 1).bool().to(device=self.device)},
        }

        input_observation = {
            "observation": input_observation,
            "task": {
                "language_instruction": self.task_description_embedding,
            }
        }
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                unscaled_raw_actions = self.agent(input_observation)
        
        unscaled_raw_actions = unscaled_raw_actions.to(torch.float32).detach().cpu().numpy()

        scaled_action = self.rescale_to_range(unscaled_raw_actions)
        return scaled_action