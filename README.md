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


Converting to Minari is in progress, currently resoling memory issues:
To test, activate Minari environment and run:
```bash
python atari_pkl_to_minari.py.py --dir={save_dir}
