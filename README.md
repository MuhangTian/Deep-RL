# RL & Bandits
## Project Strucutre
* All runs are initiated with `main.py`
* `algo` contains my own implementation of some algorithms like DQN and PPO
* `model` folder contains the neural network architectures used in the learning algorithms (which are stored in `algo`)
* `trained` contains trained agents (neural networks) stored in `.pt` format using PyTorch, as well as video recordings of agent behavior
* `utils` contains environment wrappers, validation functions, data loaders, and preprocessing functions, which are used in algorithms implemented in `algo`
* `cluster` contains shell scripts used to run experiments using cluster computer (with slurm)
