## Port D4RL MuJoCo to Minari
(tested in Python 3.10.11)

Download D4RL and Minari:
 - clone https://github.com/Farama-Foundation/Minari
 - clone D4RL https://github.com/Farama-Foundation/D4RL
 - Setup separate dependencies, e.g, conda environment for each repo

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


Test loading new environments:

```python
import minari

dataset = minari.load_dataset('d4rl_halfcheetah-expert-v2')

env = dataset.recover_environment() # Deprecated HalfCheetah-v3 environment

# Sample an episode
episode = dataset.sample_episodes(n_episodes=1)[0]
```