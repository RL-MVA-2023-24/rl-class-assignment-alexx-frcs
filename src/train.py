from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn 
import random
import numpy as np
from tqdm import tqdm
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


device = torch.device("cpu")
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

def greedy_action_dqn(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    
def greedy_action_fqi(Q,s,nb_actions):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)



  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class DQM_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super(DQM_model, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
"""
class ProjectAgent:

    def act(self, observation, use_random=False):
        if use_random:
            a = np.random.choice(4)
            # print(a)
            return a
        else:
            a = greedy_action_dqn(self.model, observation)
            # print(a)
            return a

    def save(self, path = "src/random_DQN_model.pt"):
        print("saving")
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    }, path)
    def load(self):
        print("loading")
        checkpoint = torch.load("src/prioritez_replay_r=f.pt", map_location=torch.device('cpu'))
        self.model = DQM_model(6, 256, 4, 6).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
"""


class ProjectAgent:

    def act(self, observation, use_random=False):
        if use_random:
            a = np.random.choice(4)
            # print(a)
            return a
        else:
            actions = np.zeros(4)
            a_dqn = greedy_action_dqn(self.dqn_model, observation)
            actions[a_dqn] += 1
            a_double_dqn = greedy_action_dqn(self.double_DQN_model, observation)
            actions[a_double_dqn] += 1
            a_et = greedy_action_fqi(self.Q, observation, 4)
            actions[a_et] += 1
            a = np.argmax(actions)
            # print(a)
            return a

    def save(self, path_dqn = "ensembling_double_DQN.pt", path_double_dqn = "ensembling_double_DQN.pt", 
             path_et = "ensembling_et.pkl"):
        print("saving DQN model")
        torch.save({
                    'model_state_dict': self.dqn_model.state_dict(),
                    }, path_dqn)
        print("saving double DQN model")
        torch.save({
                    'model_state_dict': self.double_DQN_model.state_dict(),
                    }, path_double_dqn)
        print("saving ET model")
        with open(path_et, 'wb') as f:
            pickle.dump(self.Q, f)

    def load(self):
        print("loading")
        checkpoint = torch.load("src/prioritez_replay_r=f.pt", map_location=torch.device('cpu'))
        self.dqn_model = DQM_model(6, 256, 4, 6).to(device)
        self.dqn_model.load_state_dict(checkpoint['model_state_dict'])
        self.dqn_model.eval()
        checkpoint = torch.load("src/double_DQN.pt", map_location=torch.device('cpu'))
        self.double_DQN_model = DQM_model(6, 256, 4, 6).to(device)
        self.double_DQN_model.load_state_dict(checkpoint['model_state_dict'])
        self.double_DQN_model.eval()
        with open("src/et.pkl", 'rb') as f:
            self.Q = pickle.load(f)
