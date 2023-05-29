import gymnasium as gym
import minari
import numpy as np

import pickle
import collections
import argparse
import os

# loads .pkl files and converts to Minari datasets
# Run using Minari dependencies

envs = ['halfcheetah', 'hopper', 'walker2d']
gym_envs = ['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3']
dataset_types = ['expert']



def convert_files(args):
    # convet each dataset
    for env_name in envs:
        for dataset_type in dataset_types:
            convert_file(env_name, dataset_type, args)

def convert_file(env_name, dataset_type, args):
    # load old file
    dataset_name = f'{env_name}-{dataset_type}-v2' 
    data_path = os.path.join(args.dir, f'{dataset_name}.pkl')
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    trajectories = []
    for i in range(N):
        done = bool(dataset['terminals'][i])
        timeout = bool(dataset['timeouts'][i])
        
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'timeouts']:
            data_[k].append(dataset[k][i])
        if done or timeout:
            #episode_data = {k: np.array(v) for k, v in data_.items()}
            episode_data = {}
            episode_data['actions'] = np.array(data_['actions'])
            episode_data['rewards'] = np.array(data_['rewards'])

            # rename
            episode_data['terminations'] = np.array(data_['terminals'])
            episode_data['truncations'] = np.array(data_['timeouts'])
            
            episode_data['observations'] = np.array(data_['observations'])
            # add last next_observation to episode_data observations
            episode_data['observations'] = np.concatenate([episode_data['observations'], data_['next_observations'][-1].reshape(1,-1)], axis=0)
            
            trajectories.append(episode_data)
            data_ = collections.defaultdict(list)

    env = gym.make(gym_envs[envs.index(env_name)])
    # Create Minari dataset

    dataset_id = f'd4rl_{dataset_name}'
    
    minari_dataset = minari.create_dataset_from_buffers(
        dataset_id = dataset_id,
        env = env,
        buffer = trajectories,
        algorithm_name = dataset['metadata/algorithm'].decode("utf-8"),
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