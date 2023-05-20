import gym
import d4rl
from d4rl.gym_mujoco import *

import pickle
import argparse
import os

# Downloads and save D4RL datasets as .pkl files.
# Run using D4RL dependencies, but can load .pkl files using updated Minari dependencies.

envs = ['halfcheetah', 'hopper', 'walker2d']
dataset_types = ['expert']
#dataset_types = ['medium', 'medium-replay', 'expert']


def download_files(args):
    save_dir = args.dir
    # create save_dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for env_name in envs:
        for dataset_type in dataset_types:
            env_id = f'{env_name}-{dataset_type}-v2'
            env = gym.make(env_id)
            dataset = env.get_dataset()

            # save dataset
            with open(os.path.join(save_dir, env_id + '.pkl'), 'wb') as f:
                pickle.dump(dataset, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./old_datasets')
    args = parser.parse_args()
    download_files(args)
    