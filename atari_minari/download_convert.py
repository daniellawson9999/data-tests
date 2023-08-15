import argparse
import gzip
import os
from os.path import expanduser
from subprocess import Popen
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import minari

from utils import create_atari_env
from atari_games import ALL_GAMES

URI = 'gs://atari-replay-datasets/dqn/{}/{}/replay_logs/'
BASE_DIR = '/media/daniel/Seagate Portable Drive/.d4rl/datasets'
# BASE_DIR = os.environ.get('D4RL_DATASET_DIR',
#                           os.path.join(expanduser('~'), '.d4rl', 'datasets'))

# Utility Functions for downloading / loading data provided by https://github.com/takuseno/d4rl-atari/blob/master/d4rl_atari/offline_env.py

def inspect_dir_path(env, index, epoch, base_dir=BASE_DIR):
    path = get_dir_path(env, index, epoch, base_dir)
    if not os.path.exists(path):
        return False
    for name in ['observation', 'action', 'reward', 'terminal']:
        if not os.path.exists(os.path.join(path, name + '.gz')):
            return False
    return True


def _download(name, env, index, epoch, dir_path):
    file_name = '$store$_{}_ckpt.{}.gz'.format(name, epoch)
    uri = URI.format(env, index) + file_name
    path = os.path.join(dir_path, '{}.gz'.format(name))
    p = Popen(['gsutil', '-m', 'cp', '-R', uri, path])
    p.wait()
    if p.returncode != 0:
        raise Exception('Failed to download {}'.format(uri))
    return path

def _load(name, dir_path):
    path = os.path.join(dir_path, name + '.gz')
    with gzip.open(path, 'rb') as f:
        #print('loading {}...'.format(path))
        return np.load(f, allow_pickle=False)    
    
def get_dir_path(env, index, epoch, base_dir=BASE_DIR):
    return os.path.join(base_dir, env, str(index), str(epoch))

def download_dataset(env, index, epoch, base_dir=BASE_DIR):
    dir_path = get_dir_path(env, index, epoch, base_dir)
    _download('observation', env, index, epoch, dir_path)
    _download('action', env, index, epoch, dir_path)
    _download('reward', env, index, epoch, dir_path)
    _download('terminal', env, index, epoch, dir_path)

def load_chunk(env, index, epoch):
    dir_path = get_dir_path(env, index, epoch)
    observation = _load('observation', dir_path)
    action = _load('action', dir_path)
    reward = _load('reward', dir_path)
    terminal = _load('terminal', dir_path)
    chunk_dict = {
        'observation': observation,
        'action': action,
        'reward': reward,
        'terminal': terminal
    }
    return chunk_dict

def timestep_to_chunk_index(timestep):
    return (timestep // 1000000)

def download_game(game, args):
    print('Checking for downloads...')
    num_chunks = 50
    for epoch in tqdm.tqdm(range(num_chunks)):
        path = get_dir_path(game, args.index, epoch)
        if not inspect_dir_path(game, args.index, epoch):
            os.makedirs(path, exist_ok=True)
            try:
                download_dataset(game, args.index, epoch)
            except:
                num_chunks = epoch - 1
                print(f'Only loaded {num_chunks} chunks for {game}')
                break

    # Just load dones, and rewards
    # Determine start / end of each trajectory and return per trajectory
    rewards = []
    dones = []
    print('Loading rewards and timesteps...')
    for epoch in tqdm.tqdm(range(num_chunks)):
        path = get_dir_path(game, args.index, epoch)
        rewards.append(_load('reward', path))
        terms = _load('terminal', path)
        truncs = np.zeros_like(terms)
        # last timestep in chunk should be a done and not continued to next chunk
        if terms[-1] == 0:
            truncs[-1] = 1
        dns = terms | truncs
        dones.append(dns)
    rewards = np.hstack(rewards)
    dones = np.hstack(dones)

    ep_returns = []
    starts = []
    ends = [] # inclusive last timestep

    starts.append(0)
    for i in range(len(dones)):
        if dones[i] or i == len(dones) - 1:
            ends.append(i)
            ep_return = np.sum(rewards[starts[-1]:i+1])
            ep_returns.append(ep_return)
            if i != len(dones) - 1:
                starts.append(i+1)
    
    # argsort the returns
    sorted_return_index = np.argsort(ep_returns)
    # get the top 10% of returns
    n_top_10 = int(len(sorted_return_index)*0.1)
    n_top_1 = int(len(sorted_return_index)*0.01)
    top_10_indices = sorted_return_index[-int(n_top_10):]
    top_1_indices = sorted_return_index[-int(n_top_1):]
    percentile_10 = np.percentile(ep_returns, 90)
    percentile_1 = np.percentile(ep_returns, 99)

    n,x,_ = plt.hist(ep_returns, bins=11)
    plt.close()
    bin_centers = 0.5*(x[1:]+x[:-1])
    plt.plot(bin_centers,n) ## using bin_centers rather than edges
    # draw red bar at 90th percentile
    plt.axvline(percentile_10, color='red')
    # draw green bar at 99th percentile
    plt.axvline(percentile_1, color='green')

    # hide y axis
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.title(game)

    plt.savefig(os.path.join(args.distrib_viz_folder, game + '.png'))

    # Optionally convert too!
    if args.convert:
        starts = np.array(starts)
        ends = np.array(ends)
        ep_returns = np.array(ep_returns)

        # get starts and ends of top 1%
        top_returns = ep_returns[top_1_indices]
        mean_top_return = np.mean(top_returns) # verify this later
        top_starts = starts[top_1_indices]
        top_ends = ends[top_1_indices]

        # sort the top 1% by start index order (ascending)
        sorted_top_indices = np.argsort(top_starts)
        top_starts = top_starts[sorted_top_indices]
        top_ends = top_ends[sorted_top_indices]
        
        start_chunks = np.array([timestep_to_chunk_index(t) for t in top_starts])
        end_chunks = np.array([timestep_to_chunk_index(t) for t in top_ends])

        assert((start_chunks != end_chunks).sum() == 0), 'expect to have all episodes start/end in same chunk'

        current_chunk_data = {}
        current_chunk_index = -1

        # observations = None
        # actions = None
        # rewards = None
        # terminals = None
        # truncations = None
        trajectories = []

        returns =  []
        # iterate over all chunks
        print('Reading top 1 from chunks...')
        for i in tqdm.tqdm(range(len(top_starts))):
            start_index = top_starts[i]
            end_index = top_ends[i]

            timestep_chunk_index = timestep_to_chunk_index(start_index)

            # Only load one chunk at a time to save memory!
            if timestep_chunk_index != current_chunk_index:
                # load new chunk
                current_chunk_index = timestep_chunk_index
                current_chunk_data = load_chunk(game, args.index, current_chunk_index)
            
            # get the start and end indices of the episode in the chunk
            chunk_steps_offset = ( (current_chunk_index) * 1000000)
            relative_start_index = start_index - chunk_steps_offset
            relative_end_index = end_index - chunk_steps_offset

            ep_obs = current_chunk_data['observation'][relative_start_index:relative_end_index+1]
            # if n is number if actions, minari wants n+1 observations, but we don't have the last observation, so we just duplicate the last one
            ep_obs = np.concatenate([ep_obs, ep_obs[-1:]], axis=0)

            ep_actions = current_chunk_data['action'][relative_start_index:relative_end_index+1]
            ep_rewards = current_chunk_data['reward'][relative_start_index:relative_end_index+1]
            returns.append(np.sum(ep_rewards))
            ep_terminals = current_chunk_data['terminal'][relative_start_index:relative_end_index+1]
            ep_truncations = np.zeros_like(ep_terminals)
            if ep_terminals[-1] == 0:
                ep_truncations[-1] = 1

            ep_data = {}
            ep_data['observations'] = ep_obs
            ep_data['actions'] = ep_actions
            ep_data['rewards'] = ep_rewards
            ep_data['terminations'] = ep_terminals
            ep_data['truncations'] = ep_truncations
            trajectories.append(ep_data)
            # actions = np.concatenate([actions, ep_actions], axis=0)
            # rewards = np.concatenate([rewards, ep_rewards], axis=0)
            # terminals = np.concatenate([terminals, ep_terminals], axis=0)
            # truncations = np.concatenate([truncations, ep_truncations], axis=0)
        prefix = f'ALE/{game}'
        env_name = f'{prefix}-v5'
        env = create_atari_env(env_name)

        dataset_id = f'{game}-top1-s{args.index}-v0'
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

def main(args):
    if args.target_game_only:
        games = [args.game]
    else:
        games = ALL_GAMES
    for game in games:
        download_game(game, args)
    
if __name__ == '__main__':
    # Parse Input
    parser = argparse.ArgumentParser(description='Download Atari Replay Datasets')
    parser.add_argument('--distrib_viz_folder', type=str, default='./distribs', help='Folder to save distribution visualizations')
    parser.add_argument('--target_game_only', default=False, action='store_true', help='Only download the target game')
    parser.add_argument('--game', type=str, default='Breakout', help='Atari Game')
    parser.add_argument('--index', type=int, default=1, help='Index of the dataset') 
    parser.add_argument('--convert', default=False, action='store_true', help='Convert to Minari')
    
    parser.add_argument("--author", type=str, help="name of the author of the dataset", default=None)
    parser.add_argument("--author_email", type=str, help="email of the author of the dataset", default=None)
    parser.add_argument("--code_permalink", type=str, help="link to the code used to generate the dataset", default='https://github.com/daniellawson9999/data-tests')

    args = parser.parse_args()
    main(args)