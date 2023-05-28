import gym
import d4rl_atari
import numpy as np

import pickle
import argparse
import os

# Downloads and save D4RL Atari datasets as .pkl files.
# Run using Atari D4RL dependencies, but can load .pkl files using updated Minari dependencies.

games = ['breakout']
# games = ['adventure', 'air-raid', 'alien', 'amidar', 'assault', 'asterix',
#         'asteroids', 'atlantis', 'bank-heist', 'battle-zone', 'beam-rider',
#         'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede',
#         'chopper-command', 'crazy-climber', 'defender', 'demon-attack',
#         'double-dunk', 'elevator-action', 'enduro', 'fishing-derby', 'freeway',
#         'frostbite', 'gopher', 'gravitar', 'hero', 'ice-hockey', 'jamesbond',
#         'journey-escape', 'kangaroo', 'krull', 'kung-fu-master',
#         'montezuma-revenge', 'ms-pacman', 'name-this-game', 'phoenix',
#         'pitfall', 'pong', 'pooyan', 'private-eye', 'qbert', 'riverraid',
#         'road-runner', 'robotank', 'seaquest', 'skiing', 'solaris',
#         'space-invaders', 'star-gunner', 'tennis', 'time-pilot', 'tutankham',
#         'up-n-down', 'venture', 'video-pinball', 'wizard-of-wor',
#         'yars-revenge', 'zaxxon']


dataset_types = ['expert']
#dataset_types = ['medium', 'mixed', 'expert']

seeds = [0]
#seeds = [0, 1]
#seeds = [0, 1, 2, 3, 4]

def download_files(args):
    save_dir = args.dir

    # create save_dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for game in games:
        for dataset_type in dataset_types:
            for seed in seeds:
                env_id = f'{game}-{dataset_type}-v{seed}'
                env = gym.make(env_id)
                #import pdb; pdb.set_trace()
                dataset = env.get_dataset()

                # save dataset
                with open(os.path.join(save_dir, env_id + '.npz'), 'wb') as f:
                    np.savez(f, **dataset)
                    #np.savez_compressed(f, **dataset)
                    #pickle.dump(dataset, f)
                del dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./old_datasets')
    args = parser.parse_args()
    download_files(args)
