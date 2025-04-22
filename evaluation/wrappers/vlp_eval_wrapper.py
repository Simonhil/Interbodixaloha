from collections import deque
import functools
import json
import os
import numpy as np
import tensorflow as tf
import torch
from accelerate import Accelerator

from safetensors.torch import load_model
from typing import Callable, Optional


import hydra
from hydra import compose, initialize
from flower_vla.agents.utils.diffuser_ema import EMAModel
from flower_vla.agents.lang_encoders.florence_tokens import TokenVLM
from flower_vla.dataset.oxe.transforms import generate_policy_prompt, get_action_space_index
from flower_vla.agents.utils.action_index import ActionIndex
from flower_vla.dataset.utils.frequency_mapping import DATASET_FREQUENCY_MAP


def get_param_hash(model):
    """
    Generate a hash of model parameters to track changes in model state.

    Args:
        model: PyTorch model whose parameters will be hashed

    Returns:
        str: A hash string representing the current state of model parameters
    """
    import hashlib

    # Initialize hasher
    hasher = hashlib.md5()

    # Iterate through all parameters
    for param in model.parameters():
        # Get numpy representation of the parameter
        param_data = param.detach().cpu().numpy()
        # Update hash with parameter data
        hasher.update(param_data.tobytes())

    # Return hexadecimal representation of hash
    return hasher.hexdigest()

class VLPEvalWrapper:
    def __init__(self, 
                 saved_model_base_dir, 
                 saved_model_path, 
                 use_ema,
                 ema_path, 
                 act_min_max_path, 
                 device: str = "cuda",
                 pred_action_horizon: int = 100,
                 replan_after_nsteps: int = 10,
                 ensemble_strategy: str = "cogact",
                 adaptive_ensemble_alpha: float = 0.1,
                 exp_decay: float = 0.0,
                 num_parallel_envs: int = 4,
                 ):
        self.saved_model_base_dir = saved_model_base_dir
        self.saved_model_path = saved_model_path
        self.ema_path = ema_path
        self.device = device

        ### action chunking
        assert ensemble_strategy in ["false", "cogact", "act"]
        self.replan_after_nsteps = replan_after_nsteps
        assert pred_action_horizon % replan_after_nsteps == 0, "pred_action_horizon must be divisible by replan_after_nsteps"
        self.act_chunk_deque = deque(maxlen=pred_action_horizon//replan_after_nsteps)
        self.ensemble_strategy = ensemble_strategy
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.exp_decay = exp_decay


        with open(act_min_max_path) as f:
            d = json.load(f)
            # self.min_values = torch.tensor(d['action']['p01']).to(device)
            # self.max_values = torch.tensor(d['action']['p99']).to(device)
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

        self.num_parallel_envs = num_parallel_envs

        cfg.batch_size = num_parallel_envs
        cfg.trainer.agent.agent.act_window_size = pred_action_horizon
        cfg.trainer.agent.agent.multistep = replan_after_nsteps if ensemble_strategy == "false" else 1
        agent = hydra.utils.instantiate(cfg.trainer.agent, device=device, process_id=0)

        accelerator = Accelerator()
        agent = accelerator.prepare(agent)
        checkpoint_path = os.path.join(saved_model_base_dir, saved_model_path)
        print(f"checkpoint path: {checkpoint_path}")
        missing, unexpected = load_model(agent, os.path.join(checkpoint_path, 'model.safetensors'))

        after_model_hash = get_param_hash(agent)

        self.use_ema=use_ema

        # ema_path = "random_states_0.pkl" 
        # if self.use_ema:
        #     ema_helper = EMAModel(
        #         parameters=agent.parameters(),
        #         decay=cfg.decay,
        #         min_decay=0.0,
        #         update_after_step=0,
        #         use_ema_warmup=True,
        #         inv_gamma=1.0,
        #         power=2/3,
        #         foreach=False,
        #         model_cls=type(agent),
        #         model_config=agent.config if hasattr(agent, 'config') else None
        #     )
        #
        #     ema_path = os.path.join(saved_model_base_dir, ema_path)
        #     print(f"ema path: {ema_path}")
        #     if os.path.exists(ema_path):
        #         ema_state = torch.load(ema_path, map_location=device)
        #         ema_helper.load_state_dict(ema_state)
        #         print("Loaded EMA weights successfully")
        #         ema_helper.copy_to(agent.parameters())
        # else:
        #     print("Not using EMA")
        
        ema_path_base = "/home/hongyi/Codes/flower_rss24/flower_vla/logs/node8/14-51-17"

        custom_ema_path = os.path.join(ema_path_base, "custom_checkpoint_0.pkl")
        ema_weights_path = os.path.join(ema_path_base, "200_ema_weights.pt")
        ema_dir = os.path.join(ema_path_base, "ema_200")
        legacy_ema_path = os.path.join(ema_path_base, "legacy_600000_ema.pt")

        if self.use_ema:
            ema_loaded = False

            # Try loading from the custom_checkpoint_0.pkl file first (most likely format based on logs)
            if os.path.exists(custom_ema_path):
                print(f"Found custom EMA checkpoint at {custom_ema_path}")
                try:
                    ema_state = torch.load(custom_ema_path, map_location=device)
                    print(f"Custom EMA state keys: {list(ema_state.keys())}")

                    # Check if this is an EMAModel state
                    if hasattr(ema_state, 'shadow_params') or 'shadow_params' in ema_state:
                        print("Found shadow_params in custom EMA checkpoint")

                        # Create EMA helper
                        ema_helper = EMAModel(
                            parameters=agent.parameters(),
                            decay=getattr(cfg, 'decay', 0.9999),
                        )

                        # Store parameter sample before EMA application
                        param_sample = next(agent.parameters())
                        before_sample = param_sample[0, 0].item() if param_sample.numel() > 0 else None

                        if isinstance(ema_state, dict):
                            if 'shadow_params' in ema_state:
                                # Direct assignment if it's a simple dict with shadow_params
                                print("Loading shadow_params directly")
                                ema_helper.shadow_params = [p.to(device) for p in ema_state['shadow_params']]
                            else:
                                print("Loading full state dict")
                                ema_helper.load_state_dict(ema_state)
                        else:
                            # It might be the actual EMAModel object
                            print("Loading from EMAModel object")
                            ema_helper = ema_state

                        # Apply the EMA weights to the model
                        print("Applying EMA weights to model...")
                        ema_helper.copy_to(agent.parameters())

                        # Check if parameter changed
                        after_sample = param_sample[0, 0].item() if param_sample.numel() > 0 else None
                        print(f"Sample param before: {before_sample}, after: {after_sample}")
                        ema_loaded = True
                except Exception as e:
                    print(f"Error loading custom EMA checkpoint: {e}")
                    import traceback
                    traceback.print_exc()

            # Check for explicit EMA weights file if custom checkpoint failed
            if not ema_loaded and os.path.exists(ema_weights_path):
                print(f"Loading explicit EMA weights from {ema_weights_path}")
                ema_state_dict = torch.load(ema_weights_path, map_location=device)

                # Store parameter sample before EMA application
                sample_key = next(iter(ema_state_dict.keys()))
                if sample_key in agent.state_dict():
                    before_sample = agent.state_dict()[sample_key][0, 0].item()
                    ema_sample = ema_state_dict[sample_key][0, 0].item()
                    print(f"Sample param before: {before_sample}, EMA value: {ema_sample}")

                agent.load_state_dict(ema_state_dict, strict=False)

                # Check if parameter changed
                if sample_key in agent.state_dict():
                    after_sample = agent.state_dict()[sample_key][0, 0].item()
                    print(f"Sample param after: {after_sample}")

                ema_loaded = True

            # Try directory format if previous methods failed
            if not ema_loaded and os.path.exists(ema_dir):
                print(f"Found new EMA format directory at {ema_dir}")
                ema_agent_path = os.path.join(ema_dir, "ema_agent.pt")

                if os.path.exists(ema_agent_path):
                    print(f"Loading EMA agent weights from {ema_agent_path}")
                    ema_agent_state = torch.load(ema_agent_path, map_location=device)

                    # Store parameter sample before EMA application
                    sample_key = next(iter(ema_agent_state.keys()))
                    if sample_key in agent.state_dict():
                        before_sample = agent.state_dict()[sample_key][0, 0].item()
                        ema_sample = ema_agent_state[sample_key][0, 0].item()
                        print(f"Sample param before: {before_sample}, EMA value: {ema_sample}")

                    agent.load_state_dict(ema_agent_state, strict=False)

                    # Check if parameter changed
                    if sample_key in agent.state_dict():
                        after_sample = agent.state_dict()[sample_key][0, 0].item()
                        print(f"Sample param after: {after_sample}")

                    ema_loaded = True

            # Finally fall back to legacy format
            if not ema_loaded and os.path.exists(legacy_ema_path):
                print(f"Using legacy EMA format from {legacy_ema_path}")
                try:
                    ema_state = torch.load(legacy_ema_path, map_location=device)
                    print(
                        f"Legacy EMA state keys: {list(ema_state.keys()) if isinstance(ema_state, dict) else 'Not a dict'}")

                    # Check if this contains EMA data
                    if isinstance(ema_state, dict) and ('ema_state' in ema_state or 'shadow_params' in ema_state):
                        print("Found EMA data in legacy file")

                        # Create EMA helper
                        ema_helper = EMAModel(
                            parameters=agent.parameters(),
                            decay=getattr(cfg, 'decay', 0.9999),
                        )

                        # Store parameter sample before EMA application
                        param_sample = next(agent.parameters())
                        before_sample = param_sample[0, 0].item() if param_sample.numel() > 0 else None

                        if 'ema_state' in ema_state:
                            ema_helper.load_state_dict(ema_state['ema_state'])
                        else:
                            ema_helper.load_state_dict(ema_state)

                        # Apply the EMA weights to the model
                        print("Applying EMA weights to model...")
                        ema_helper.copy_to(agent.parameters())

                        # Check if parameter changed
                        after_sample = param_sample[0, 0].item() if param_sample.numel() > 0 else None
                        print(f"Sample param before: {before_sample}, after: {after_sample}")
                        ema_loaded = True
                except Exception as e:
                    print(f"Error loading legacy EMA weights: {e}")
                    import traceback
                    traceback.print_exc()

            after_ema_hash = get_param_hash(agent)
            print(f"Before EMA hash: {after_model_hash}")
            print(f"After EMA hash: {after_ema_hash}")
            print(f"Params changed after EMA loading: {after_model_hash != after_ema_hash}")
            print(f"EMA loaded: {ema_loaded}")

            if not ema_loaded:
                print("WARNING: EMA was enabled but no EMA weights were loaded!")
        else:
            print("Not using EMA")

        print(missing)
        print(unexpected)

        agent.agent.use_proprio = True
        agent.to(dtype=torch.bfloat16)
        agent.eval()
        if self.ensemble_strategy != "false":
            agent.agent.return_act_chunk = True
        print("Model loaded successfully")


        self.agent = agent
        self.observation = None
        self.task_description = None
        self.task_description_embedding = None
        
        self.format_instruction = functools.partial(
                             generate_policy_prompt,
                             robot_name="ViperX",
                             action_space="joint_position",
                             num_arms="2",
                             prompt_style='minimal')

        self.action_space_index = torch.tensor([get_action_space_index('JOINT_POS_BIMANUAL', 2,
                                                                       'position', return_tensor=False)]).repeat(self.num_parallel_envs)
        self.frequency = torch.tensor([DATASET_FREQUENCY_MAP[70]]).unsqueeze(0).repeat(self.num_parallel_envs, 1)
        self.action_index = ActionIndex()
        self.image_size = 224

        self.pred_action_sequence = None
        self.execution_count = 0

    def rescale_to_range(self, value):
        max_values = self.max_values
        min_values = self.min_values
        new_min = -np.ones_like(value)
        new_max = np.ones_like(value)
        rescaled_tensor = (value - new_min) / (new_max - new_min) * (max_values - min_values) + min_values
        return rescaled_tensor

    def scale_proprio_to_range(self, tensor) -> torch.Tensor:
        self.proprio_max_values = self.proprio_max_values.to(tensor.device)
        self.proprio_min_values = self.proprio_min_values.to(tensor.device)
        denominator = self.proprio_max_values - self.proprio_min_values
        denominator[denominator == 0] = 1  # Avoid division by zero
        normalized_data = 2 * (tensor - self.proprio_min_values) / denominator - 1
        return normalized_data

    def ensemble_action(self, pred_ac_sequence):
        # ac = pred_ac_sequence.cpu().detach().numpy().squeeze()
        ac = pred_ac_sequence
        self.act_chunk_deque.append(ac)

        num_actions = len(self.act_chunk_deque)

        curr_act_preds = np.stack(
            [
                pred_actions[:, self.replan_after_nsteps*i : self.replan_after_nsteps*(i+1)]
                for (i, pred_actions) in zip(
                range(num_actions - 1, -1, -1), self.act_chunk_deque
            )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.exp_decay * np.arange(num_actions))
        weights = weights / weights.sum()

        # return the weighted average across all predictions for this timestep
        return np.sum(weights[:, None, None, None] * curr_act_preds, axis=0)

    def cognitive_ensemble_action(self, pred_ac_sequence):
        """
        from CogACT
        Cognitive ensemble strategy using cosine similarity with thresholding.
        Input: act_chunk of shape [1, T, D_action]
        Output: single action step
        """
        # ac = pred_ac_sequence.cpu().detach().numpy().squeeze()
        ac = pred_ac_sequence
        self.act_chunk_deque.append(ac)
        num_actions = len(self.act_chunk_deque)
        if ac.ndim == 1:
            curr_act_preds = np.stack(self.act_chunk_deque)
        else:
            curr_act_preds = np.stack(
                [pred_actions[:, self.replan_after_nsteps * i : self.replan_after_nsteps * (i+1)]
                 for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.act_chunk_deque)]
            )

        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions - 1, ...]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=-1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=-1)
        norm_ref = np.linalg.norm(ref, axis=-1)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum(axis=0)

        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[..., None] * curr_act_preds, axis=0)

        return cur_action


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
                self.task_description_embedding['input_ids'].repeat(self.num_parallel_envs, 1)
            self.task_description_embedding['attention_mask'] = \
                self.task_description_embedding['attention_mask'].repeat(self.num_parallel_envs, 1)
        else:
            self.task_description = ""
            self.task_description_embedding = tf.zeros((512,), dtype=tf.float32)

    def reset(self, task_description: str) -> None:
        self._initialize_task_description(task_description)
        self.act_chunk_deque.clear()
        self.curr_horizon_index = 0
        self.pred_action_sequence = None
        self.execution_count = 0
        # self.act_history = deque(maxlen=self.predict_horizon)



    def step(self, observation: dict, task_description: Optional[str] = None, *args, **kwargs) -> tuple[
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
        # print("step:", self.execution_count)
        if self.execution_count % self.replan_after_nsteps == 0:
            self.pred_action_sequence = None
            self.execution_count = 0
        #
        if self.pred_action_sequence is not None:
            self.execution_count += 1
            return self.pred_action_sequence[:, self.execution_count - 1]

        task_description = self.format_instruction(task_description)
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)
                self.agent.agent.reset()
                self.act_chunk_deque.clear()
                self.execution_count = 0
                self.pred_action_sequence = None

        image_primary = observation['images']["cam_high"]
        image_secondary = observation['images']["cam_left_wrist"]
        image_third = observation['images']["cam_right_wrist"]
        state = observation["state"]

        image_primary = torch.from_numpy(self._resize_image(image_primary)).unsqueeze(0).unsqueeze(0).to(self.device)
        image_secondary = torch.from_numpy(self._resize_image(image_secondary)).unsqueeze(0).unsqueeze(0).to(self.device)
        image_third = torch.from_numpy(self._resize_image(image_third)).unsqueeze(0).unsqueeze(0).to(self.device)


        proprio = torch.tensor(state).unsqueeze(0)
        zero_padding = torch.zeros((proprio.shape[0], 2,))
        proprio = self.scale_proprio_to_range(proprio)
        proprio = torch.cat([proprio, zero_padding], dim=-1)


        input_observation = {
            "image_primary": image_primary,
            "image_secondary": image_secondary,
            "image_wrist": image_third,
            "proprio": proprio.to(dtype=torch.bfloat16),
            "pad_mask_dict": {"image_primary": torch.ones(image_primary.shape[0], 1).bool().to(device=self.device),
                              "image_secondary": torch.ones(image_primary.shape[0], 1).bool().to(device=self.device),
                              "image_wrist": torch.ones(image_primary.shape[0], 1).bool().to(device=self.device)},
        }
        input_observation = {
            "observation": input_observation,
            "task": {
                "language_instruction": self.task_description_embedding,
                "frequency": self.frequency,
                "action_space_index": self.action_space_index,
            }
        }
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                unscaled_raw_actions = self.agent(input_observation)
        
        unscaled_raw_actions = unscaled_raw_actions.to(torch.float32).detach().cpu().numpy()

        if self.ensemble_strategy == "act":
            act_chunk = unscaled_raw_actions[:, :, :self.action_index.get_action_dim(self.action_space_index[0])]
            unscaled_raw_actions = self.ensemble_action(act_chunk)
        elif self.ensemble_strategy == "cogact":
            act_chunk = unscaled_raw_actions[:, :, :self.action_index.get_action_dim(self.action_space_index[0])]
            unscaled_raw_actions = self.cognitive_ensemble_action(act_chunk)
        elif self.ensemble_strategy == "false":
            pass
        else:
            raise ValueError("Invalid ensemble strategy, choose from 'false', 'act', 'cogact'")

        finger_1 = unscaled_raw_actions[...,6]
        finger_2 = unscaled_raw_actions[...,13]
        scaled_action = self.rescale_to_range(unscaled_raw_actions)
        scaled_action[...,6] = finger_1
        scaled_action[...,13] = finger_2

        scaled_action[..., 6] = 2.0 * (scaled_action[..., 6]>0.5) - 1.0
        scaled_action[..., 13] = 2.0 * (scaled_action[..., 13]>0.5) - 1.0

        self.pred_action_sequence = None if self.ensemble_strategy == "false" else scaled_action
        self.execution_count += 1
        scaled_action = scaled_action[:, 0]

        if scaled_action.ndim == 3:
            scaled_action = scaled_action.squeeze(1)

        return scaled_action