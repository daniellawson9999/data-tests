import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformReward 
import numpy as np


def create_atari_env(env_name, repeat_action_probability=0.25, clip_rewards=False):
    assert 'v5' in env_name

    env = gym.make(env_name, frameskip=1, repeat_action_probability=repeat_action_probability) # e.g. 'ALE/Breakout-v5'
    env = AtariPreprocessing(env, frame_skip=4, noop_max=0)
    if clip_rewards:
        env = TransformReward(env, lambda r: np.clip(r, -1.0, 1.0))

    return env
