import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import minari
import numpy as np

import tqdm

import pickle
import collections
import argparse
import os

from utils import create_atari_env


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
#seeds = [0, 1, 2, 3, 4]

max_episode_steps = 108000 // 4

def convert_name(short):
    return ''.join([w.capitalize() for w in short.split('-')])

def convert_files(args):
    # convet each dataset
    for game in games:
        for dataset_type in dataset_types:
            for seed in seeds:
                convert_file(game, dataset_type, seed, args)

def convert_file(game, dataset_type, seed, args):
    # load old file
    dataset_name = f'{game}-{dataset_type}-v{seed}'
    data_path = os.path.join(args.dir, f'{dataset_name}.npz')

    # load memory-mapped file
    np_dataset = np.load(data_path)
    #import pdb; pdb.set_trace()
    dataset = {}
    for k in np_dataset.keys():
        dataset[k] = np_dataset[k]
    
    N = np_dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    trajectories = []
    start_index = 0
    #import pdb; pdb.set_trace()
    for i in tqdm.tqdm(range(N)):
        done = bool(dataset['terminals'][i])  

        if done:
            episode_data = {}
            episode_data['actions'] = dataset['actions'][start_index:i+1]#np.array(data_['actions'])
            episode_data['rewards'] = dataset['rewards'][start_index:i+1]#np.array(data_['rewards'])

            ep_len = len(episode_data['rewards'])
            if ep_len < max_episode_steps:
                timeout = False
            else:
                timeout = True
            terminal = not timeout
            terminations = np.zeros_like(episode_data['rewards'], dtype=bool)
            truncations = np.zeros_like(episode_data['rewards'], dtype=bool)
            terminations[-1] = terminal
            truncations[-1] = timeout
            episode_data['terminations'] = terminations
            episode_data['truncations'] = truncations

            episode_data['observations'] = dataset['observations'][start_index:i+2,0,:]#np.array(data_['observations'])
            trajectories.append(episode_data)
            start_index = i + 1
            #data_ = collections.defaultdict(list)
    for key in list(dataset.keys()):
        del dataset[key]
    del dataset

    # pad the last observation in the last trajectory if needed, other trajectories are padded
    if len(trajectories[-1]['observations']) == len(trajectories[-1]['rewards']):
        trajectories[-1]['observations'] = np.concatenate([trajectories[-1]['observations'], np.expand_dims(trajectories[-1]['observations'][-1],axis=0)], axis=0)

    env_game_name = convert_name(game)
    prefix = f'ALE/{env_game_name}'
    env_name = f'{prefix}-v5'
    # env = gym.make(env_name, frameskip=1, repeat_action_probability=0.25)
    # env = AtariPreprocessing(env, frame_skip=4, noop_max=0)
    env = create_atari_env(env_name)

    #env_name = f'{env_game_name}-{dataset_type}' 

    #dataset_id = f'{prefix}-{dataset_type}_s{seed}-v0'

    dataset_id = f'{env_game_name}-{dataset_type}_s{seed}-v0'
    
    print("saving")
    # saving
    minari_dataset = minari.create_dataset_from_buffers(
        dataset_id = dataset_id,
        env = env,
        buffer = trajectories,
        algorithm_name = 'dqn',
        author = args.author,
        author_email = args.author_email,
        code_permalink = args.code_permalink
    )
    print(f'saved dataset {dataset_id}')

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./old_datasets')
    parser.add_argument("--author", type=str, help="name of the author of the dataset", default=None)
    parser.add_argument("--author_email", type=str, help="email of the author of the dataset", default=None)
    parser.add_argument("--code_permalink", type=str, help="link to the code used to generate the dataset", default=None)
    args = parser.parse_args()
    convert_files(args)