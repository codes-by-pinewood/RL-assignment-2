## Reinforcement Learning for CartPole
This repository consists of a comprehensive analysis of the implementation of DQN (Deep Q learning Network) on the CartPole environment. Our analysis consists of two significant parts: a hyperparameter tuning and an ablation study. 


### Dependencies
- Python 3.x
- OpenAI Gym
- Pytorch 
- NumPy


### Installation
- Clone the repository: git clone https://github.com/codes-by-pinewood/RL-assignment-2
- Install the relevant dependencies


### Repositories
#### Hyper-Parameter Tuning 
The section called 'Hyper-Parameter Tuning' includes a number of different .ipynb files where we experimented with the neural network architecture by changing the number of layers, number of neurons, and batch size of our replay buffer.


### Ablation Study 
The section called ablation study includes three different ablations we performed on our neural network. 
- DQN versus DQN-ER: The dqn_and_dqn_er_comparison.ipynb file contains a comparison of the DQN neural Network with DQN neural network without a replay buffer.
- DQN versus DQN-TN: The dqn_tn_and_dqn.ipynb file contains a comparison of the DQN neural Network with DQN neural network without a target network.
- DQN versus DQN-TN-ER: The dqn_and_dqn_tn_er.ipynb file contains a comparison of the DQN neural Network with DQN neural network without either a target network or a replay buffer.
