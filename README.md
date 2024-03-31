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
The section called 'Hyper-Parameter Tuning' includes a number of different .py files where we experimented with the neural network architecture by changing the number of layers, number of neurons, and batch size of our replay buffer.
- batch_size: The dqn_with_diff_batch_size.py file contains a comparison of the full DQN network when changing batch size
- epsilons: The dqn_with_diff_epsilons.py file contains a comparison of the full DQN network when changing epsilon (e-greedy policy implementation)
- temperature: The dqn_with_diff_temps.py file contains a comparison of the full DQN network when changing temperature (Boltzmann policy)
- neurons, layers and learning rates: The diff_layers_and_neurons_lr.py file contains a comparison of the full DQN network when changing the number of neurons and number of layers in the architecture plus changing the learning rate 


### Ablation Study 
The section called ablation study includes three different ablations we performed on our neural network. 
- DQN versus DQN-ER: The dqn_and_dqn_er_comparison.py file contains a comparison of the DQN neural Network with DQN neural network without a replay buffer.
- DQN versus DQN-TN: The dqn_tn_and_dqn_comparison.py file contains a comparison of the DQN neural Network with DQN neural network without a target network.
- DQN versus DQN-TN-ER: The dqn_and_dqn_tn_er_comparison.py file contains a comparison of the DQN neural Network with DQN neural network without either a target network or a replay buffer.


### User Test
Our user_test.py contains a file for users to test different DQN types at their own will. Users can decide on the command line which type of neural network they want to execute. For example, the command ```!python user_test.py``` will execute the full DQN (with target network and experience replay) under epsilon greedy. However a ```!python user_test.py -ER``` will execute a DQN without experience replay and a ```!python user_test.py -TN``` will execute a DQN without a target network.  
Furthermore to experiment with different exploration policies we can use -Softmax and -EpsilonGreedy commands to choose which one we want to implement.
