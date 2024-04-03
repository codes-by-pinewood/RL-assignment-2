# %%
import numpy as np
import torch
import torch.nn as nn
import random

# Components of the agent: online_q_net, target_q_net, experience buffer

class Replaybuffer:
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action
        self.size = 2000 # Memory pool size
        self.batchsize = 10

        # Allocate space for the experience tuple
        self.s = np.empty(shape=(self.size, self.n_state), dtype=np.float32)
        self.a = np.random.randint(low=0, high=n_action, size=self.size, dtype=np.uint8)
        self.r = np.empty(self.size, dtype=np.float32)
        self.done = np.random.randint(low=0, high=2, size=self.size, dtype=np.uint8)
        self.s_ = np.empty(shape=(self.size, self.n_state), dtype=np.float32)

        self.t = 0
        self.tmax = 0  # Initialize tmax attribute

    def add_memo(self, s, a, r, done, s_): # Functionality needed: 1. Add memory after interaction 2. Take out memory when sampling a batch
    # Add memory to the memory pool at step t
        self.s[self.t] = s
        self.a[self.t] = a
        self.r[self.t] = r
        self.done[self.t] = done
        self.s_[self.t] = s_
        self.t = self.t + 1 if self.t + 1 < self.size else 1 # When t reaches 2001, start adding from 1 again
        self.tmax = max(self.tmax, self.t + 1)


    def sample(self):
    # Sampling logic: If the buffer has more experiences than batchsize, then sample; if fewer, then take as many as there are
        if self.tmax > self.batchsize:
           k = self.batchsize  # If the number of samples in the buffer is greater than or equal to the batch size, use the batch size
        else:
           k = self.tmax  # Otherwise, use the actual number of samples in the buffer

        idxes = random.sample(range(0, self.tmax), k)  # Sample with the determined k value

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []

        for idx in idxes: # Sample 64 pieces of data
            batch_s.append(self.s[idx])
            batch_a.append(self.a[idx])
            batch_r.append(self.r[idx])
            batch_done.append(self.done[idx])
            batch_s_.append(self.s_[idx])

        # Convert numpy arrays to torch tensors
        batch_s = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1) # Dimension increase: from (2) to (2,1)
        batch_r = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_done = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_s_ = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return batch_s, batch_a, batch_r, batch_done, batch_s_

class Qnetwork(nn.Module):
      def __init__(self, n_input, n_output):
          super().__init__() #Inherit from the Module superclass

          self.net = nn.Sequential(
              nn.Linear(in_features= n_input, out_features = 128),
              nn.ReLU(),
              nn.Linear(in_features= 128, out_features = n_output))

      def forward(self,x):
           return self.net(x) #forward propagation

      def act(self,obs): #In the face of s, find the maximum Q value (because the neural network outputs more than the maximum Q value) and output the corresponding action
          obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
          q_value = self(obs_tensor.unsqueeze(0)) #convert it to the row vector
          max_q_idx = torch.argmax(input=q_value)
          action = max_q_idx.detach().item() #the action correspoding to the index of the highest Q-value
          return action



class Qnetwork2(nn.Module):
      def __init__(self, n_input, n_output):
          super().__init__()

          self.net = nn.Sequential(
              nn.Linear(in_features= n_input, out_features = 128),
              nn.ReLU(),
              nn.Linear(in_features= 128, out_features = 128),
              nn.ReLU(),
              nn.Linear(in_features= 128, out_features = n_output))

      def forward(self,x):
           return self.net(x)

      def act(self,obs):
          obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
          q_value = self(obs_tensor.unsqueeze(0))
          max_q_idx = torch.argmax(input=q_value)
          action = max_q_idx.detach().item()
          return action

class Qnetwork3(nn.Module):
      def __init__(self, n_input, n_output):
          super().__init__()

          self.net = nn.Sequential(
              nn.Linear(in_features= n_input, out_features = 32),
              nn.ReLU(),
              nn.Linear(in_features= 32, out_features = n_output))

      def forward(self,x):
           return self.net(x)

      def act(self,obs):
          obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
          q_value = self(obs_tensor.unsqueeze(0))
          max_q_idx = torch.argmax(input=q_value)
          action = max_q_idx.detach().item()
          return action


class Qnetwork4(nn.Module):
      def __init__(self, n_input, n_output):
          super().__init__()

          self.net = nn.Sequential(
              nn.Linear(in_features= n_input, out_features = 32),
              nn.ReLU(),
              nn.Linear(in_features= 32, out_features = 32),
              nn.ReLU(),
              nn.Linear(in_features= 32, out_features = n_output))

      def forward(self,x):
           return self.net(x)

      def act(self,obs):
          obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
          q_value = self(obs_tensor.unsqueeze(0))
          max_q_idx = torch.argmax(input=q_value)
          action = max_q_idx.detach().item()
          return action



class Qnetwork5(nn.Module):
      def __init__(self, n_input, n_output):
          super().__init__()

          self.net = nn.Sequential(
              nn.Linear(in_features= n_input, out_features = 128),
              nn.ReLU(), #nn.Tanh(),
              nn.Linear(in_features= 128, out_features = 128),
              nn.ReLU(),
              nn.Linear(in_features= 128, out_features = 128),
              nn.ReLU(),
              nn.Linear(in_features= 128, out_features = n_output))

      def forward(self,x):
           return self.net(x)

      def act(self,obs):
          obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
          q_value = self(obs_tensor.unsqueeze(0))
          max_q_idx = torch.argmax(input=q_value)
          action = max_q_idx.detach().item()
          return action


class Qnetwork6(nn.Module):
      def __init__(self, n_input, n_output):
          super().__init__()

          self.net = nn.Sequential(
              nn.Linear(in_features= n_input, out_features = 256),
              nn.ReLU(), #nn.Tanh(),
              nn.Linear(in_features= 256, out_features = n_output))

      def forward(self,x):
           return self.net(x)

      def act(self,obs):
          obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
          q_value = self(obs_tensor.unsqueeze(0))
          max_q_idx = torch.argmax(input=q_value)
          action = max_q_idx.detach().item()
          return action








class Agent:
   def __init__(self, n_input, n_output):
            self.n_input = n_input
            self.n_output = n_output

            self.Gamma = 0.97
            self.learning_rate = 0.01

            self.memo = Replaybuffer(self.n_input, self.n_output)

            #The two networks have the same structure, so it can be instantiated by a same class
            self.online_net = Qnetwork(self.n_input, self.n_output)
            self.target_net = Qnetwork(self.n_input, self.n_output)

            self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                              lr=self.learning_rate)

class Agent2:
   def __init__(self, n_input, n_output):
            self.n_input = n_input
            self.n_output = n_output

            self.Gamma = 0.97
            self.learning_rate = 0.001

            self.memo = Replaybuffer(self.n_input, self.n_output)

            self.online_net = Qnetwork(self.n_input, self.n_output)
            self.target_net = Qnetwork(self.n_input, self.n_output)

            self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                              lr=self.learning_rate)

class Agent3:
   def __init__(self, n_input, n_output):
            self.n_input = n_input
            self.n_output = n_output

            self.Gamma = 0.97
            self.learning_rate = 0.05

            self.memo = Replaybuffer(self.n_input, self.n_output)

            self.online_net = Qnetwork(self.n_input, self.n_output)
            self.target_net = Qnetwork(self.n_input, self.n_output)

            self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                              lr=self.learning_rate)


#Different NN
class Agent4:
   def __init__(self, n_input, n_output):
            self.n_input = n_input
            self.n_output = n_output

            self.Gamma = 0.97
            self.learning_rate = 0.01

            self.memo = Replaybuffer(self.n_input, self.n_output)

            self.online_net = Qnetwork2(self.n_input, self.n_output)
            self.target_net = Qnetwork2(self.n_input, self.n_output)

            self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                              lr=self.learning_rate)

class Agent5:
   def __init__(self, n_input, n_output):
            self.n_input = n_input
            self.n_output = n_output

            self.Gamma = 0.97
            self.learning_rate = 0.01

            self.memo = Replaybuffer(self.n_input, self.n_output)

            self.online_net = Qnetwork3(self.n_input, self.n_output)
            self.target_net = Qnetwork3(self.n_input, self.n_output)

            self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                              lr=self.learning_rate)


class Agent6:
   def __init__(self, n_input, n_output):
            self.n_input = n_input
            self.n_output = n_output

            self.Gamma = 0.97
            self.learning_rate = 0.01

            self.memo = Replaybuffer(self.n_input, self.n_output)

            self.online_net = Qnetwork4(self.n_input, self.n_output)
            self.target_net = Qnetwork4(self.n_input, self.n_output)

            self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                              lr=self.learning_rate)

class Agent7:
   def __init__(self, n_input, n_output):
            self.n_input = n_input
            self.n_output = n_output

            self.Gamma = 0.97
            self.learning_rate = 0.01

            self.memo = Replaybuffer(self.n_input, self.n_output)

            self.online_net = Qnetwork5(self.n_input, self.n_output)
            self.target_net = Qnetwork5(self.n_input, self.n_output)

            self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                              lr=self.learning_rate)


class Agent8:
   def __init__(self, n_input, n_output):
            self.n_input = n_input
            self.n_output = n_output

            self.Gamma = 0.97
            self.learning_rate = 0.01

            self.memo = Replaybuffer(self.n_input, self.n_output)

            self.online_net = Qnetwork6(self.n_input, self.n_output)
            self.target_net = Qnetwork6(self.n_input, self.n_output)

            self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                              lr=self.learning_rate)


# %%
import gym
import numpy as np
import random
import torch
import torch.nn as nn #Classes and methods required for neural networks
#from agent import Agent, Agent2, Agent3, Agent4, Agent5, Agent6
import matplotlib.pyplot as plt

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode='human')
s = env.reset()

EPSILON_DECAY = 10000
EPSILON_START = 0.9
EPSILON_END = 0.1
TARGET_UPDATE = 5

num_state = len(s)
num_action = env.action_space.n #2

n_episode = 1000
n_step = 500

# %%
Reward_list = np.empty(shape=n_episode)

agent = Agent(n_input=num_state, n_output=num_action)
episode_list = []
reward_list = []

for episode in range(n_episode):
    epi_reward = 0
    for step in range(n_step):
        'epsilon greedy with decay of epsilon'
        epsilon = np.interp(episode * n_step + step, [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END]) # Interpolation
#episode * n_step + step: The point to interpolate in the first step of the current episode
#[0, EPSILON_DECAY] : data point horizontal coordinate [EPSILON_START, EPSILON_END] : data point vertical coordinate
#epsilon value changes linearly from EPSILON_START to EPSILON_END within EPSILON_DECAY

        random_sample = random.random()
        if random_sample <= epsilon:
           a = env.action_space.sample()
        else:
           a = agent.online_net.act(s) #todo
        'Interact with the env'
        #print(env.step(a))
        s_, r, done , _ = env.step(a) #Perform action a，we get s_,r,done,info
        agent.memo.add_memo(s, a, r, done, s_) #Add experiences into the exp buffer
        s = s_ #store transition
        epi_reward += r
        #print(epi_reward)

        if done:
           s = env.reset()
           Reward_list[episode] = epi_reward #Record the total reward of this episode
           break

        '''Sample minibatches from the transition'''
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()

        '''Compute Q_target'''
        target_q_values = agent.target_net(batch_s_)
        target_q = batch_r + agent.Gamma * (1-batch_done) * target_q_values.max(dim=1, keepdim=True)[0]
        '''Compute Q_pred'''
        pred_q_values = agent.online_net(batch_s) # For each state in the batch, it gives the Q value for each action
        pred_q = torch.gather(input=pred_q_values, dim=1, index=batch_a)
       # Based on the action index specified in batch_a, select the corresponding action from the action Q value pred_q_values of each state
        '''Compute Loss, gredient descent'''
        loss = nn.functional.smooth_l1_loss(target_q, pred_q)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step() # Descend according to the gradient

        '''Fix Q-target'''
    if episode % TARGET_UPDATE ==0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        reward = np.mean(Reward_list[episode-10:episode])
        print("Episode:{}".format(episode))
        print("Reward:{}".format(reward))
        episode_list.append(episode)
        reward_list.append(reward)

# %%
#Tuning Learning rates
Reward_list2 = np.empty(shape=n_episode)
agent2 = Agent2(n_input=num_state, n_output=num_action)
episode_list2 = []
reward_list2 = []

for episode in range(n_episode):
    epi_reward = 0
    for step in range(n_step):
        'epsilon greedy with decay of epsilon'
        epsilon = np.interp(episode * n_step + step, [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])
        random_sample = random.random()
        if random_sample <= epsilon:
           a = env.action_space.sample()
        else:
           a = agent2.online_net.act(s)
        'Interact with the env'
        #print(env.step(a))
        s_, r, done, _ = env.step(a)
        agent2.memo.add_memo(s, a, r, done, s_)
        s = s_ #store transition
        epi_reward += r
        #print(epi_reward)

        if done:
           s = env.reset()
           Reward_list2[episode] = epi_reward
           break

        '''Sample minibatches from the transition'''
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent2.memo.sample()

        '''Compute Q_target'''
        target_q_values = agent2.target_net(batch_s_)
        target_q = batch_r + agent2.Gamma * (1-batch_done) * target_q_values.max(dim=1, keepdim=True)[0]
        '''Compute Q_pred'''
        pred_q_values = agent2.online_net(batch_s)
        pred_q = torch.gather(input=pred_q_values, dim=1, index=batch_a)

        '''Compute Loss, gredient descent'''
        loss = nn.functional.smooth_l1_loss(target_q, pred_q)
        agent2.optimizer.zero_grad()
        loss.backward()
        agent2.optimizer.step()

        '''Fix Q-target'''
    if episode % TARGET_UPDATE ==0:
        agent2.target_net.load_state_dict(agent2.online_net.state_dict())
        reward = np.mean(Reward_list2[episode-10:episode])
        print("Episode:{}".format(episode))
        print("Reward:{}".format(reward))
        episode_list2.append(episode)
        reward_list2.append(reward)



Reward_list3 = np.empty(shape=n_episode)
agent3 = Agent3(n_input=num_state, n_output=num_action)
episode_list3 = []
reward_list3 = []

for episode in range(n_episode):
    epi_reward = 0
    for step in range(n_step):
        'epsilon greedy with decay of epsilon'
        epsilon = np.interp(episode * n_step + step, [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])

        random_sample = random.random()
        if random_sample <= epsilon:
           a = env.action_space.sample()
        else:
           a = agent3.online_net.act(s)
        'Interact with the env'
        #print(env.step(a))
        s_, r, done , _ = env.step(a)
        agent3.memo.add_memo(s, a, r, done, s_)
        s = s_ #store transition
        epi_reward += r
        #print(epi_reward)

        if done:
           s = env.reset()
           Reward_list3[episode] = epi_reward
           break

        '''Sample minibatches from the transition'''
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent3.memo.sample()

        '''Compute Q_target'''
        target_q_values = agent3.target_net(batch_s_)
        target_q = batch_r + agent3.Gamma * (1-batch_done) * target_q_values.max(dim=1, keepdim=True)[0]
        '''Compute Q_pred'''
        pred_q_values = agent3.online_net(batch_s)
        pred_q = torch.gather(input=pred_q_values, dim=1, index=batch_a)

        '''Compute Loss, gredient descent'''
        loss = nn.functional.smooth_l1_loss(target_q, pred_q)
        agent3.optimizer.zero_grad()
        loss.backward()
        agent3.optimizer.step()

        '''Fix Q-target'''
    if episode % TARGET_UPDATE ==0:
        agent3.target_net.load_state_dict(agent3.online_net.state_dict())
        reward = np.mean(Reward_list3[episode-10:episode])
        print("Episode:{}".format(episode))
        print("Reward:{}".format(reward))
        episode_list3.append(episode)
        reward_list3.append(reward)

# %%
#Tuning the network
Reward_list4 = np.empty(shape=n_episode)
agent4 = Agent4(n_input=num_state, n_output=num_action)
episode_list4 = []
reward_list4 = []

for episode in range(n_episode):
    epi_reward = 0
    for step in range(n_step):
        'epsilon greedy with decay of epsilon'
        epsilon = np.interp(episode * n_step + step, [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])

        random_sample = random.random()
        if random_sample <= epsilon:
           a = env.action_space.sample()
        else:
           a = agent4.online_net.act(s)
        'Interact with the env'
        #print(env.step(a))
        s_, r, done , _ = env.step(a)
        agent4.memo.add_memo(s, a, r, done, s_)
        s = s_ #store transition
        epi_reward += r
        #print(epi_reward)

        if done:
           s = env.reset()
           Reward_list4[episode] = epi_reward
           break

        '''Sample minibatches from the transition'''
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent4.memo.sample()

        '''Compute Q_target'''
        target_q_values = agent4.target_net(batch_s_)
        target_q = batch_r + agent4.Gamma * (1-batch_done) * target_q_values.max(dim=1, keepdim=True)[0]
        '''Compute Q_pred'''
        pred_q_values = agent4.online_net(batch_s)
        pred_q = torch.gather(input=pred_q_values, dim=1, index=batch_a)

        '''Compute Loss, gredient descent'''
        loss = nn.functional.smooth_l1_loss(target_q, pred_q)
        agent4.optimizer.zero_grad()
        loss.backward()
        agent4.optimizer.step()

        '''Fix Q-target'''
    if episode % TARGET_UPDATE ==0:
        agent4.target_net.load_state_dict(agent4.online_net.state_dict())
        reward = np.mean(Reward_list4[episode-10:episode])
        print("Episode:{}".format(episode))
        print("Reward:{}".format(reward))
        episode_list4.append(episode)
        reward_list4.append(reward)



Reward_list5 = np.empty(shape=n_episode)
agent5 = Agent5(n_input=num_state, n_output=num_action)
episode_list5 = []
reward_list5 = []

for episode in range(n_episode):
    epi_reward = 0
    for step in range(n_step):
        'epsilon greedy with decay of epsilon'
        epsilon = np.interp(episode * n_step + step, [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])
        random_sample = random.random()
        if random_sample <= epsilon:
           a = env.action_space.sample()
        else:
           a = agent5.online_net.act(s)
        'Interact with the env'
        #print(env.step(a))
        s_, r, done , _ = env.step(a)
        agent5.memo.add_memo(s, a, r, done, s_)
        s = s_ #store transition
        epi_reward += r
        #print(epi_reward)

        if done:
           s = env.reset()
           Reward_list5[episode] = epi_reward
           break

        '''Sample minibatches from the transition'''
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent5.memo.sample()

        '''Compute Q_target'''
        target_q_values = agent5.target_net(batch_s_)
        target_q = batch_r + agent5.Gamma * (1-batch_done) * target_q_values.max(dim=1, keepdim=True)[0]
        '''Compute Q_pred'''
        pred_q_values = agent5.online_net(batch_s)
        pred_q = torch.gather(input=pred_q_values, dim=1, index=batch_a)

        '''Compute Loss, gredient descent'''
        loss = nn.functional.smooth_l1_loss(target_q, pred_q)
        agent5.optimizer.zero_grad()
        loss.backward()
        agent5.optimizer.step()

        '''Fix Q-target'''
    if episode % TARGET_UPDATE ==0:
        agent5.target_net.load_state_dict(agent5.online_net.state_dict())
        reward = np.mean(Reward_list5[episode-10:episode])
        print("Episode:{}".format(episode))
        print("Reward:{}".format(reward))
        episode_list5.append(episode)
        reward_list5.append(reward)


# %%
Reward_list7 = np.empty(shape=n_episode)
agent7 = Agent7(n_input=num_state, n_output=num_action)
episode_list7 = []
reward_list7 = []

for episode in range(n_episode):
    epi_reward = 0
    for step in range(n_step):
        'epsilon greedy with decay of epsilon'
        epsilon = np.interp(episode * n_step + step, [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])

        random_sample = random.random()
        if random_sample <= epsilon:
           a = env.action_space.sample()
        else:
           a = agent7.online_net.act(s)
        'Interact with the env'
        #print(env.step(a))
        s_, r, done , _ = env.step(a)
        agent7.memo.add_memo(s, a, r, done, s_)
        s = s_ #store transition
        epi_reward += r
        #print(epi_reward)

        if done:
           s = env.reset()
           Reward_list7[episode] = epi_reward
           break

        '''Sample minibatches from the transition'''
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent7.memo.sample()

        '''Compute Q_target'''
        target_q_values = agent7.target_net(batch_s_)
        target_q = batch_r + agent7.Gamma * (1-batch_done) * target_q_values.max(dim=1, keepdim=True)[0]
        '''Compute Q_pred'''
        pred_q_values = agent7.online_net(batch_s)
        pred_q = torch.gather(input=pred_q_values, dim=1, index=batch_a)

        '''Compute Loss, gredient descent'''
        loss = nn.functional.smooth_l1_loss(target_q, pred_q)
        agent7.optimizer.zero_grad()
        loss.backward()
        agent7.optimizer.step()

        '''Fix Q-target'''
    if episode % TARGET_UPDATE ==0:
        agent7.target_net.load_state_dict(agent7.online_net.state_dict())
        reward = np.mean(Reward_list7[episode-10:episode])
        print("Episode:{}".format(episode))
        print("Reward:{}".format(reward))
        episode_list7.append(episode)
        reward_list7.append(reward)

# %%
Reward_list8 = np.empty(shape=n_episode)
agent8 = Agent8(n_input=num_state, n_output=num_action)
episode_list8 = []
reward_list8 = []

for episode in range(n_episode):
    epi_reward = 0
    for step in range(n_step):
        'epsilon greedy with decay of epsilon'
        epsilon = np.interp(episode * n_step + step, [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])

        random_sample = random.random()
        if random_sample <= epsilon:
           a = env.action_space.sample()
        else:
           a = agent8.online_net.act(s)
        'Interact with the env'
        #print(env.step(a))
        s_, r, done , _ = env.step(a)
        agent8.memo.add_memo(s, a, r, done, s_)
        s = s_ #store transition
        epi_reward += r
        #print(epi_reward)

        if done:
           s = env.reset()
           Reward_list8[episode] = epi_reward
           break

        '''Sample minibatches from the transition'''
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent8.memo.sample()

        '''Compute Q_target'''
        target_q_values = agent8.target_net(batch_s_)
        target_q = batch_r + agent8.Gamma * (1-batch_done) * target_q_values.max(dim=1, keepdim=True)[0]
        '''Compute Q_pred'''
        pred_q_values = agent8.online_net(batch_s)
        pred_q = torch.gather(input=pred_q_values, dim=1, index=batch_a)

        '''Compute Loss, gredient descent'''
        loss = nn.functional.smooth_l1_loss(target_q, pred_q)
        agent8.optimizer.zero_grad()
        loss.backward()
        agent8.optimizer.step()

        '''Fix Q-target'''
    if episode % TARGET_UPDATE ==0:
        agent8.target_net.load_state_dict(agent8.online_net.state_dict())
        reward = np.mean(Reward_list8[episode-10:episode])
        print("Episode:{}".format(episode))
        print("Reward:{}".format(reward))
        episode_list8.append(episode)
        reward_list8.append(reward)

# %%
#Different Lr
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(episode_list, reward_list2, label='lr=0.001')
plt.plot(episode_list, reward_list, label='lr=0.01')
plt.plot(episode_list, reward_list3, label='lr=0.05')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN with different learning rates')
plt.show()

# %%
#Different QN layers
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(episode_list, reward_list, label='1 layer')
plt.plot(episode_list, reward_list4, label='2 layers')
plt.plot(episode_list, reward_list7, label='3 layers')
#plt.plot(episode_list, reward_list6, label='QN4')
plt.legend(fontsize=18)
plt.xlabel('Episode',fontsize=16)
plt.ylabel('Reward',fontsize=16)
plt.xticks(fontsize=16 )
plt.yticks(fontsize=16 )
plt.title('DQN with different Q-network layers',fontsize=24)
plt.show()

# %%
#Different number of neurons
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(episode_list, reward_list, label='64 neurons')
plt.plot(episode_list, reward_list5, label='32 neurons')
plt.plot(episode_list, reward_list8, label='128 neurons')
#plt.plot(episode_list, reward_list6, label='QN4')
plt.legend(fontsize=18)
plt.xlabel('Episode',fontsize=16)
plt.ylabel('Reward',fontsize=16)
plt.xticks(fontsize=16 )
plt.yticks(fontsize=16 )
plt.title('DQN with different Q-network neurons',fontsize=24)
plt.show()


