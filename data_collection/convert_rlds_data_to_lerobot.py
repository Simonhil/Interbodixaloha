# Source code: https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py


"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import gc

REPO_NAME = "hXroboXh/aloha_right_left_transfer_big_mix_lerobot"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "aloha_right_left_transfer_99",
    "aloha_right_left_transfer_mix"
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def import_dataset_visualize():
    data_dir = "/home/huang/other_workspace/hongyi/tensorflow_datasets/aloha_right_left_transfer/1.0.0"
    
    # data_dir = "/home/huang/other_workspace/hongyi/real_aloha_cube_transfer"

    dataset = tfds.builder_from_directory(data_dir).as_dataset(split='train[:10%]')
    dataset = dataset.prefetch(1)

    raw_dataset = tfds.load("aloha_right_left_transfer", data_dir="/home/huang/other_workspace/hongyi/tensorflow_datasets", split="train")

    for episode in dataset.take(10):
        print("Episode keys:", episode.keys())

        # Print first step
        steps_ds = episode['steps']
        jnt_state = []
        for step in steps_ds:  # Take the first step of the episode
            action = step["action"].numpy()
            jnt_state.append(action)

        print("step keys: ", step.keys())


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="aloha",
        fps=60,
        features={
            "images_top": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "images_wrist_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "images_wrist_right": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["action"],
            },
            # "timestamp": {
            #     "dtype": "float32",
            #     "shape": (1,),
            #     "names": ["timestamp"],
            # },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    count = 0
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train", read_config=tfds.ReadConfig(try_autocache=False))
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "images_top": step["observation"]["images_top"],
                        "images_wrist_left": step["observation"]["images_wrist_left"],
                        "images_wrist_right": step["observation"]["images_wrist_right"],
                        "observation.state": step["observation"]["state"],
                        
                        "action": step["action"],
                        "task": step["language_instruction"].decode(),
                        # "timestamp": step["timestamp"]
                    }
                )
            dataset.save_episode()
            if count % 10 == 0:
                gc.collect()
                print("Episode ----- ", count)
            count += 1

    # Consolidate the dataset, skip computing stats since we will do that later
    # dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["aloha", "rlds"],
            private=False,
            push_videos=False,
            license="apache-2.0",
        )


if __name__ == "__main__":
    
    # import_dataset_visualize()
    
    tyro.cli(main)