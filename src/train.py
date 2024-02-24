from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn 
import random
import numpy as np
import locale
from tqdm import tqdm
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import extratrees

locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français


device = torch.device("cpu")
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

def greedy_action_dqn(network, state):
    device = "cpu"
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

class ProjectAgent:

    def __init__(self):
        pass

    def act(self, observation, use_random=False):
            # print(observation)
            if use_random:
                return np.random.choice(env.action_space.n)
            Qs0a = []
            for a in range(env.action_space.n):
                s0a = np.append(observation,a).reshape(1, -1)
                Qs0a.append(self.Q.predict(s0a))
            return np.argmax(Qs0a)
    
    def save(self, path = "et.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self.Q, f)

    def load(self):
        path = "et.pkl"
        if not os.path.exists(path):
            print("No model to load")
            return
        with open(path, 'rb') as f:
            self.model= pickle.load(f)
            self.model.eval()
