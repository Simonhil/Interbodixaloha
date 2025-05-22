
import time
from typing import Callable, Optional
import functools
import os

from data_collection import teleop_helper
from data_collection.cams.real_cams import LogitechCamController
# from evaluation.wrappers.vlp_new_wrapper import VLPWrapper
from flower_vla.eval.aloha.vlp_real_aloha import VLPWrapper
from utils.keyboard import KeyManager
import threading
from pynput import keyboard
import wandb
from typing import Callable
from hydra import compose, initialize
import imageio
import numpy as np
import torch
import tensorflow as tf
import itertools
import tqdm
from tqdm import trange


import imageio
import datetime
from data_collection.config import BaseConfig as bc
from data_collection.teleop_helper import initialize_bots_replay, opening_replay, move_one_pair
from interbotix_xs_msgs.msg import JointSingleCommand

def rollout(
    bot_left,bot_right,
    gripper_left_command, gripper_right_command,
    policy: VLPWrapper,
    task_description: Optional[str] = None,
    max_episode_steps: int = 400,
    record_from_top: bool = False
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A a dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE the that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.

    Returns:
        The dictionary described above.
    """
    # Reset the policy and environments.
    policy.reset()
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    done = False
    
    #generated progress bar
    progbar = trange(
        max_episode_steps,
        desc=f"Running rollout with at most {max_episode_steps} steps",
        disable=True,
        leave=False,
    )


    #TODO getting images is questionable
    observation = teleop_helper.get_observation(bot_left, bot_right)

    start_event = threading.Event()
    stop_event = threading.Event()
    success_event = threading.Event()

    def on_press(key):
        try:
            if key.char == 's':
                print("s pressed - success")
                success_event.set()
            if key.char == 'r':
                print("r pressed - reset")
                stop_event.set()
            if key.char == 'l':
                print("l pressed - launch!")
                start_event.set()

        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Press 'l' to launch the policy, 's' if success, press 'r' to reset. ")
    desired_fps = 60.0
    desired_dt = 1 / desired_fps

    # Waiting for command..
    while not start_event.is_set():
        time.sleep(0.1)

    top_cam_raw = []
    left_cam_raw = []
    right_cam_raw = []

    if record_from_top:
        top_cam_raw.append(observation["images_top_raw"])
        left_cam_raw.append(observation["images_left_raw"])
        right_cam_raw.append(observation["images_right_raw"])

    del observation["images_top_raw"]
    del observation["images_left_raw"]
    del observation["images_right_raw"]


    while not done and step < max_episode_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.

        starting_t = time.time()
        with torch.inference_mode():
            action = policy.predict(observation)[None,...]
            # print(f"shape for action: {action.shape}")

        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, new_done= teleop_helper.step(action,bot_left, bot_right, gripper_left_command, gripper_right_command)
        end_t = time.time()
        time.sleep(max(0, desired_dt - (end_t - starting_t)))

        if record_from_top:
            top_cam_raw.append(observation["images_top_raw"])
            left_cam_raw.append(observation["images_left_raw"])
            right_cam_raw.append(observation["images_right_raw"])

        del observation["images_top_raw"]
        del observation["images_left_raw"]
        del observation["images_right_raw"]
        


        if success_event.is_set():
            is_success = True
            new_done = True

        if stop_event.is_set():
            break

        done = done | new_done

        # all_actions.append(torch.from_numpy([action]))
        # all_rewards.append(torch.from_numpy([reward]))
        # all_dones.append(torch.tensor([new_done]))
        # all_successes.append(torch.tensor([is_success]))

        step += 1
        progbar.update()

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        # "action": torch.stack(all_actions, dim=1),
        # "reward": torch.stack(all_rewards, dim=1),
        # "success": torch.stack(all_successes, dim=1),
        # "done": torch.stack(all_dones, dim=1),
    }

    if record_from_top:
        now = datetime.now()

        # Format it for a filename (e.g., 2025-05-13_15-42-10)
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")     
        imageio.mimsave(f"/home/simon/xi_checkpoints/video/top_cam_{timestamp_str}.mp4", np.stack(top_cam_raw), fps=60)
        imageio.mimsave(f"/home/simon/xi_checkpoints/video/left_cam_{timestamp_str}.mp4", np.stack(left_cam_raw), fps=60)
        imageio.mimsave(f"/home/simon/xi_checkpoints/video/right_cam_{timestamp_str}.mp4", np.stack(right_cam_raw), fps=60)

    return ret


def main():
    
    os.environ['MUJOCO_GL'] = 'egl'

    task_description = "Pick up the yellow cube with right arm, transfer it from the right arm to the left arm and then go to a safe position."

    seed = 0
    cam_controller = LogitechCamController()
    cam_controller.start_capture()
    bot_left, bot_right = initialize_bots_replay()
    opening_replay(bot_left, bot_right)
    gripper_left_command = JointSingleCommand(name='gripper')
    gripper_right_command = JointSingleCommand(name='gripper')



    
    with initialize(config_path="../config"):
        cfg = compose(config_name="real_aloha_transfer_dataset_rollout")

    vlp_agent = VLPWrapper(
                    saved_model_base_dir = cfg.path.saved_model_base_dir,
                    saved_model_path = cfg.path.checkpoint,
                    act_min_max_path = cfg.path.dataset_statistics_path,
                    language_instruction = cfg.language_instruction,
                    device = cfg.device,
                    pred_action_horizon = cfg.pred_horizon,
                    replan_after_nsteps = cfg.replan_after_nsteps,
                    sampling_strategy=cfg.sampling_strategy,
                    num_k=cfg.num_k,
                    init_pos=cfg.init_pos,
                    ensemble=cfg.ensemble_strategy,
                    use_proprio=cfg.use_proprio,
                    model_type=cfg.model_type
                    )

    vlp_agent.reset()
    n_episodes = 10
    all_successes = 0


    for episode in tqdm.tqdm(range(n_episodes)):
        move_one_pair(bot_left, bot_right)
        rollout_data = rollout(bot_left, bot_right, gripper_left_command, gripper_right_command, vlp_agent, task_description, 5000,
                               record_from_top=False)

    print(f"\n\n\n\n\n {all_successes} of {n_episodes} succeded \n\n\n\n\n")

if __name__=="__main__":
    main()