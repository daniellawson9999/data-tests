Experimenting with Minari, through importing older D4RL datasets (tested in Python 3.10.11).

## Port D4RL MuJoCo to Minari

Download D4RL and Minari:
 - clone https://github.com/Farama-Foundation/Minari
 - clone D4RL https://github.com/Farama-Foundation/D4RL
 - Setup separate dependencies, e.g, conda environment for each repo

```
cd 4rl_mujoco_minari
```

Activate D4RL environment and run:
```bash
python mujoco_d4rl_to_pkl.py --dir={save_dir}
```
where save_dir is the directory to store D4RL .pkl files.


Activate Minari environment and run:
```bash
python mujoco_pkl_to_minari.py --dir={save_dir}
```
Where author, author_email, code_permalink can be added optionally.

<!-- python mujoco_pkl_to_minari.py  --author "Daniel Lawson" --author_email daniellawson9999@gmail.com -->


Test loading new environments:

```python
import minari

dataset = minari.load_dataset('d4rl_halfcheetah-expert-v2')

env = dataset.recover_environment() # Deprecated HalfCheetah-v3 environment

# Sample an episode
episode = dataset.sample_episodes(n_episodes=1)[0]
```


## Port Atari  to Minari
in progress

Download D4RL and Minari:
 - clone https://github.com/Farama-Foundation/Minari (do not have to clone again if already followed setup in previous step)
 - clone Atari https://github.com/takuseno/d4rl-atari
 - Setup separate dependencies, e.g, conda environment for each repo
 - in Atari environment, run: `pip install gym[atari]`, `pip install gym[accept-rom-license]`


Activate Atari environment and run:
```bash
python atari_to_pkl.py --dir={save_dir}
```
where save_dir is the directory to store D4RL .npz files.


To convert datasets, run:
To test, activate Minari environment and run:
```bash
python atari_pkl_to_minari.py.py --dir={save_dir}
```
This will create dataset(s), with the name {env_name}-{dataset_type}_s{seed}-v0, where env_name is the name of the environment, e.g. Breakout. Seed and dataset_type follow from https://github.com/takuseno/d4rl-atari, where we test with expert, which contains datasets consisting of the last 1M steps of training. _s{seed} specified which trained agent to use, which is referred to as -v in Takuma's Github, but renamed to seed (_s) as -v is used to specify dataset version in Minari.

Example of loading a dataset:

```python
import minari
from atari_minari.utils import create_atari_env

dataset = minari.load_dataset('Breakout-expert_s0-v0')

base_env = dataset.recover_environment() # Recommended to instead build env, as follows:
env = create_atari_env('ALE/Breakout-v5', repeat_action_probability=0.25, clip_rewards=True)
# disable action_repeat for some evaluation
env = create_atari_env('ALE/Breakout-v5', repeat_action_probability=0.0, clip_rewards=True)

# Sample an episode
episode = dataset.sample_episodes(n_episodes=1)[0]
```

There are several things to note:
- dataset.recover_environment() will return the environment without reward_clipping due to issues serializing TransformReward(). To load with environment clipping, recreate the environment with create_atari_env() and pass clip_rewards=True
- While the dataset is [Optimistic Perspective on Offline Reinforcement Learning](https://arxiv.org/pdf/1907.04543.pdf) is collected with repeat_action_probability=0.25, two recent papers, [Multi-Game Decision Transformers](https://arxiv.org/abs/2205.15241), [Scaled QL](https://arxiv.org/abs/2211.15144). which aim at creating generalist Atari agents use this dataset for training, but set repeat_action_probability=0.0 during evaluation.
- Both the dataset,and the environment, return un-scaled 84x84 observations, with values ranging from 0 to 255. One should normalize these values before network input, such as by dividing observations by 255 to scale to 0 to 1. 


