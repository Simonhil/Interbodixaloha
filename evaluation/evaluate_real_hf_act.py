
import time
from typing import Callable, Optional
import functools
import os

from data_collection import teleop_helper
from data_collection.cams.real_cams import LogitechCamController
import threading
from pynput import keyboard
import torch
import numpy as np
import tqdm
from tqdm import trange

from data_collection.config import BaseConfig as bc
from data_collection.teleop_helper import initialize_bots_replay, opening_replay, move_one_pair
from interbotix_xs_msgs.msg import JointSingleCommand
import matplotlib.pyplot as plt

# Hugging face
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.configs import parser
import einops


def convert_observation_to_hf_format(observation, device):
    top = observation["images_top"] / 255.0
    left = observation["images_wrist_left"] / 255.0
    right = observation["images_wrist_right"] / 255.0

    state = torch.from_numpy(observation["state"]).to(device=device, dtype=torch.float32)
    state = einops.rearrange(state, "s -> 1 s" )

    top = einops.rearrange(torch.from_numpy(top), 'h w c -> 1 c h w').to(device=device, dtype=state.dtype)
    left = einops.rearrange(torch.from_numpy(left), 'h w c -> 1 c h w').to(device=device, dtype=state.dtype) 
    right = einops.rearrange(torch.from_numpy(right), 'h w c -> 1 c h w').to(device=device, dtype=state.dtype) 

    # images = torch.vstack([top, left, right]).to(device=device)
    obs = {"images_top": top,
            "images_wrist_left": left,
            "images_wrist_right": right,
            "observation.state": state}
    
    return obs



def rollout(
    bot_left,
    bot_right,
    gripper_left_command, 
    gripper_right_command,
    policy: PreTrainedPolicy,
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
    follower_joint_states = []
    actions = []

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
    follower_joint_states.append(observation["state"])
    observation = convert_observation_to_hf_format(observation, device="cuda")

    with torch.inference_mode():
        action = policy.select_action(observation)

    start_event = threading.Event()
    stop_event = threading.Event()
    success_event = threading.Event()

    def on_press(key):
        try:
            if key.char == 's':
                print("s pressed - success")
                start_event.set()
            if key.char == 'r':
                print("r pressed - reset")
                stop_event.set()
            if key.char == 'l':
                print("l pressed - launch!")
                stop_event.set()


        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Press 'launch' to launch the policy, 's' if success, press 'r' to reset. ")
    desired_fps = 60.0
    desired_dt = 1 / 60.0

    # Waiting for command..
    while not start_event.is_set():
        time.sleep(0.1)

    while not done and step < max_episode_steps:
        # Apply the next action.
        starting_t = time.time()
        with torch.inference_mode():
            action = policy.select_action(observation)
            # Convert to CPU / numpy.
            action = action.to("cpu").numpy()
            actions.append(action)
            # print(f"shape for action: {action.shape}")

        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"
        observation, reward, new_done= teleop_helper.step(action,bot_left, bot_right, gripper_left_command, gripper_right_command)
        follower_joint_states.append(observation["state"])
        end_t = time.time()
        time.sleep(max(0, desired_dt - (end_t - starting_t)))

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

        observation = convert_observation_to_hf_format(observation, device="cuda")

    plot_episode(actions=actions, states=follower_joint_states)


    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        # "action": torch.stack(all_actions, dim=1),
        # "reward": torch.stack(all_rewards, dim=1),
        # "success": torch.stack(all_successes, dim=1),
        # "done": torch.stack(all_dones, dim=1),
    }

    return ret


def plot_episode(actions: list, states: list):
    actions = np.vstack(actions)
    states = np.vstack(states)

    fig, axs = plt.subplots(7, 1)
    for i in range(7):
        axs[i].plot(actions[:, i], "b", label="L-Actions" if i == 0 else "")
        axs[i].plot(states[:, i], "r--", label="L-States" if i == 0 else "")

    # Create one legend for the whole figure
    lines = [
        axs[0].lines[0],  # First line plotted (trajectory)
        axs[0].lines[1],  # Second line plotted (action)
    ]
    labels = ["L-Actions", "L-States"]
    fig.legend(lines, labels, loc="upper right")
    plt.tight_layout()

    fig, axs = plt.subplots(7, 1)
    for i in range(7):
        axs[i].plot(actions[:, i+7], "b", label="R-Actions" if i == 0 else "")
        axs[i].plot(states[:, i+7], "r--", label="R-States" if i == 0 else "")

    # Create one legend for the whole figure
    lines = [
        axs[0].lines[0],  # First line plotted (trajectory)
        axs[0].lines[1],  # Second line plotted (action)
    ]
    labels = ["R-Actions", "R-States"]
    fig.legend(lines, labels, loc="upper right")
    plt.tight_layout()

    plt.show()


@parser.wrap()
def act_hf_main(cfg: TrainPipelineConfig):
    cfg.validate()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    dataset = make_dataset(cfg)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    policy.eval()
    policy.to(device="cuda")

    n_episodes = 10

    all_successes = 0

    # original rollout pipeline
    os.environ['MUJOCO_GL'] = 'egl'

    task_description = "Pick up the yellow cube with right arm, transfer it from the right arm to the left arm and then go to a safe position."

    seed = 0
    cam_controller = LogitechCamController()
    cam_controller.start_capture()
    bot_left, bot_right = initialize_bots_replay()
    opening_replay(bot_left, bot_right)
    gripper_left_command = JointSingleCommand(name='gripper')
    gripper_right_command = JointSingleCommand(name='gripper')

    for episode in tqdm.tqdm(range(n_episodes)):
        move_one_pair(bot_left, bot_right)
        rollout_data = rollout(bot_left, bot_right, gripper_left_command, gripper_right_command, policy, task_description, 5000)
        policy.reset()

    print(f"\n\n\n\n\n {all_successes} of {n_episodes} succeded \n\n\n\n\n")

if __name__=="__main__":
    act_hf_main()