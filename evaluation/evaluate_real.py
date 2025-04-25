
import time
from typing import Callable, Optional
import functools
import os

from data_collection import teleop_helper
from data_collection.cams.real_cams import LogitechCamController
from evaluation.wrappers.vlp_new_wrapper import VLPWrapper
from utils.keyboard import KeyManager
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

from data_collection.config import BaseConfig as bc
from data_collection.teleop_helper import initialize_bots_replay, opening_replay, move_one_pair
from interbotix_xs_msgs.msg import JointSingleCommand

def rollout(
    bot_left,bot_right,
    gripper_left_command, gripper_right_command,
    policy: VLPWrapper,
    task_description: Optional[str] = None,
    max_episode_steps: int = 400,
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



    km = KeyManager()
    while not done and step < max_episode_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.



        #TODO WHY? Do i need this?
        #observation["pixels"] = np.stack([obs['top'] for obs in observation['pixels']])


        with torch.inference_mode():
            action = policy.predict(observation).squeeze(0)
            print(f"shape for action: {action.shape}")

        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, new_done= teleop_helper.step(action,bot_left, bot_right, gripper_left_command, gripper_right_command)
        time.sleep(bc.STEPSPEED)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available of none of the envs finished.
        # successes = []
        # for info_dict in info:
        #     truncated.append(info_dict["TimeLimit.truncated"])
        #     successes.append(info_dict["is_success"])
        # successes = np.array(successes)
        # truncated = np.array(truncated)

        # Keep track of which environments are done so far.

        is_success = False

        if km.key == "s":
            is_success = True
            new_done = True

        if km.key == "a":
            return

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

    return ret


def main():

    os.environ['MUJOCO_GL'] = 'egl'

    # with initialize(config_path="../config"):
        # cfg = compose(config_name="vlp_aloha")

    # if cfg.gpu_id is not None:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    # n_parallel_envs = cfg.evaluation.num_parallel_envs
    # max_episode_steps = cfg.evaluation.max_episode_steps
    # replan_after_nsteps_list = cfg.evaluation.replan_after_nsteps
    # ensemble_strategy_list = cfg.evaluation.ensemble_strategy


        #TODO generalise
        # if cfg.evaluation.task == "transfer":
        #     env_id = "gym_aloha/AlohaTransferCube-v0"
    task_description = ""
    # elif cfg.evaluation.task == "insertion":
    #     env_id = "gym_aloha/AlohaInsertion-v0"
    #     task_description = "Insert the peg into the socket."
    # else:
    #     raise ValueError("Invalid task")


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
                    replan_after_nsteps = cfg.replan_after_nsteps)

    vlp_agent.reset()


    n_episodes = 10



    all_successes = 0


    for episode in tqdm.tqdm(range(n_episodes)):
        move_one_pair(bot_left, bot_right)
        rollout_data = rollout(bot_left, bot_right, gripper_left_command, gripper_right_command, vlp_agent, task_description, 1000)

    print(f"\n\n\n\n\n {all_successes} of {n_episodes} succeded \n\n\n\n\n")

if __name__=="__main__":
    main()