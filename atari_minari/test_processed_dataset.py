import minari
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformReward 
import numpy as np
from utils import create_atari_env

# load a dataset
dataset = minari.load_dataset('Breakout-expert_s0-v0')

# sample an episode
episode = dataset.sample_episodes(n_episodes=1)[0]

# get the environment
dataset_env = dataset.recover_environment()

# or create manually
manual_env = create_atari_env('ALE/Breakout-v5')

# remove random action repeat for some evaluation, clip rewards
eval_env = create_atari_env('ALE/Breakout-v5', repeat_action_probability=0.0, clip_rewards=True)

import pdb; pdb.set_trace()