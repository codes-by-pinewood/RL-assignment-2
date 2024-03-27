# -*- coding: utf-8 -*-
"""dqn_er_comparison.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-T1w_-rNr3wXpQZLpYzmPbC2SQqieWcZ
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import requests
import numpy as np

class Replaybuffer:
    def __init__(self,n_state,n_action):
        self.n_state = n_state
        self.n_action = n_action
        self.size = 2000 #capacity of buffer
        self.batchsize = 10

        #Apply for space for memory buffer
        self.s = np.empty(shape = (self.size, self.n_state), dtype=np.float32)
        self.a = np.random.randint(low=0, high=n_action, size=self.size, dtype=np.uint8)
        self.r = np.empty(self.size, dtype=np.float32)
        self.done = np.random.randint(low=0, high=2, size=self.size, dtype=np.uint8)
        self.s_ = np.empty(shape = (self.size, self.n_state), dtype=np.float32)

        self.t = 0
        self.tmax = 0  # initialise tmax

    def add_memo(self,s,a,r,done,s_):
    #append to memory buffer
        self.s[self.t] = s
        self.a[self.t] = a
        self.r[self.t] = r
        self.done[self.t] = done
        self.s_[self.t] = s_
        self.t = self.t + 1 if self.t + 1 < self.size else 1 #if size exceeds 2001, reset to 1
        self.tmax = max(self.tmax, self.t +1)



    def sample(self):
    #Sampling logic: If the experience in the Buffer is more than batchsize, then sample;
    #if there is less experience than batchsize, then select a few.

        if self.tmax > self.batchsize:
           k = self.batchsize  # if # of buffer samples is >= batch size, use the batch size
        else:
           k = self.tmax  # else, use actual # of samples in buffer

        idxes = random.sample(range(0, self.tmax), k)  #Sampling using a determined k value

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []

        for idx in idxes: #Extract data
            batch_s.append(self.s[idx])
            batch_a.append(self.a[idx])
            batch_r.append(self.r[idx])
            batch_done.append(self.done[idx])
            batch_s_.append(self.s_[idx])

        #convert numpy arrays into torch tensors
        batch_s = torch.as_tensor(np.asarray(batch_s),dtype=torch.float32)
        batch_a = torch.as_tensor(np.asarray(batch_a),dtype=torch.int64).unsqueeze(-1) #Dim from (2,) to (2,1)
        batch_r = torch.as_tensor(np.asarray(batch_r),dtype=torch.float32).unsqueeze(-1)
        batch_done = torch.as_tensor(np.asarray(batch_done),dtype=torch.float32).unsqueeze(-1)
        batch_s_ = torch.as_tensor(np.asarray(batch_s_),dtype=torch.float32)

        return batch_s, batch_a, batch_r, batch_done, batch_s_

class Qnetwork(nn.Module):
      def __init__(self, n_input, n_output):
          super().__init__() #initalise module

          self.net = nn.Sequential(
              nn.Linear(in_features= n_input, out_features = 128),
              nn.ReLU(),
              nn.Linear(in_features= 128, out_features = n_output))

      def forward(self,x):
           return self.net(x) #forward propagation

      def act(self,obs):
        #With obs, find the maximum Q value and output the corresponding action
          obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
          q_value = self(obs_tensor.unsqueeze(0)) #convert to row vector
          max_q_idx = torch.argmax(input=q_value)
          action = max_q_idx.detach().item() #action corresponding to max Q index
          return action


class AgentwoRB:
   def __init__(self, n_input, n_output, Gamma=0.97, learning_rate = 0.01):
            self.n_input = n_input
            self.n_output = n_output
            self.learning_rate = learning_rate
            self.Gamma = Gamma
            self.online_net = Qnetwork(self.n_input, self.n_output)
            self.target_net = Qnetwork(self.n_input, self.n_output)
            self.optimizer = torch.optim.Adam(self.online_net.parameters(),lr=self.learning_rate)

class AgentwRB:
   def __init__(self, n_input, n_output, Gamma=0.97, learning_rate = 0.01):
            self.n_input = n_input
            self.n_output = n_output
            self.learning_rate = learning_rate
            self.Gamma = Gamma
            self.memo = Replaybuffer(self.n_input, self.n_output) #Initialise the Agent, get the memo, and then initialise the Replaybuffer.

            #Initialise online network and target network structure
            self.online_net = Qnetwork(self.n_input, self.n_output)
            self.target_net = Qnetwork(self.n_input, self.n_output)

            self.optimizer = torch.optim.Adam(self.online_net.parameters(),lr=self.learning_rate)

env = gym.make('CartPole-v1')
n_input = env.observation_space.shape[0]
n_output = env.action_space.n

epsilon_decay = 10000
epsilon_start = 1.0 #best value from tuning hyperparameters
epsilon_end = 0.1
target_update = 5
n_step = 500
n_episode = 1000
TARGET_UPDATE = 10
Gamma=0.97
learning_rate = 0.01

#DQN-ER (without experience replay)
agent = AgentwoRB(n_input, n_output)
episode2_array = []
rewards2_array = []
Reward_list = np.empty(shape=n_episode)
n_episode = 1000

s = env.reset()

for episode in range(n_episode):
    epi_reward = 0
    for step in range(n_step):
        epsilon = np.interp(episode * n_step + step, [0, epsilon_decay], [epsilon_start, epsilon_end])
        random_sample = random.random()
        if random_sample <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s)

        s_, r, done, _ = env.step(a)
        epi_reward += r

        # Learning process
        target_q_values = agent.target_net(torch.tensor(s_).unsqueeze(0))
        target_q = r + agent.Gamma * (1 - done) * target_q_values.max(dim=1)[0]

        pred_q_values = agent.online_net(torch.tensor(s).unsqueeze(0))
        pred_q = pred_q_values[0][a]

        loss = nn.functional.smooth_l1_loss(target_q, pred_q)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        s = s_

        if done:
            s = env.reset()
            Reward_list[episode] = epi_reward
            break

    if episode % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        reward = np.mean(Reward_list[episode-10:episode])
        print("Episode:{}".format(episode))
        print("Reward:{}".format(reward))
        episode2_array.append(episode)
        rewards2_array.append(reward)

x2 = episode2_array
y2 = rewards2_array

##DQN (with target network AND experience replay)
s = env.reset()
agent = AgentwRB(n_input, n_output)
episode_array = []
rewards_array = []
Reward_list = np.empty(shape=n_episode)
n_episode = 1000

for episode in range(n_episode):
    epi_reward = 0
    for step in range(n_step):
        'epsilon greedy with decay of epsilon'
        epsilon = np.interp(episode * n_step + step, [0, epsilon_decay], [epsilon_start, epsilon_end])

        random_sample = random.random()
        if random_sample <= epsilon:
           a = env.action_space.sample()
        else:
           a = agent.online_net.act(s) #todo

        'Interact with the env'
        s_, r, done, _ = env.step(a) #execute action
        agent.memo.add_memo(s, a, r, done, s_) #add to replay buffer
        s = s_ #store transition
        epi_reward += r

        if done:
           s = env.reset()
           Reward_list[episode] = epi_reward #record total reward for this episode
           break

        '''Sample minibatches from the transition'''
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()

        '''Compute Q_target'''
        target_q_values = agent.target_net(batch_s_)
        target_q = batch_r + agent.Gamma * (1-batch_done) * target_q_values.max(dim=1, keepdim=True)[0]
        '''Compute Q_pred'''
        pred_q_values = agent.online_net(batch_s) #For each state in the batch, it will give the Q value of each action
        pred_q = torch.gather(input=pred_q_values, dim=1, index=batch_a)
        #According to the action index specified in batch_a, select the corresponding action from the action Q value pred_q_values ​​of each state
        '''Compute Loss, gredient descent'''
        loss = nn.functional.smooth_l1_loss(target_q, pred_q)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step() #descent based on gradient

        '''Fix Q-target'''
    if episode % TARGET_UPDATE ==0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        reward = np.mean(Reward_list[episode-10:episode])
        print("Episode:{}".format(episode))
        print("Reward:{}".format(reward))
        episode_array.append(episode)
        rewards_array.append(reward)

import matplotlib.pyplot as plt
import numpy as np

#set x and y for the final plot
x = episode_array
y = rewards_array

#plotting the comparison between the two
plt.title("Performance of DQN with and without Experience Replay")
plt.plot(x, y, label = "DQN")
plt.plot(x2, y2, label = "DQN-ER")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.legend()
plt.show()