# RL & Bandits
Experiments relevant to RL and bandit algorithms to get a better understanding of the code implementation.

## Project Strucutre
* `algo` contains my own implementation of some algorithms like DQN and PPO
* `model` folder contains the neural network architectures used in the learning algorithms (which are stored in `algo`)
* `trained` contains trained agents stored in `.pt` format using PyTorch
* `utils` contains environment wrappers, validation functions, data loaders, and preprocessing functions, which are used in algorithms implemented in `algo`
* `cluster` contains shell scripts used to run experiments using cluster computer (with slurm)