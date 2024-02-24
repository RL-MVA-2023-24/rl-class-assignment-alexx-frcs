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

    def collect_samples(self, env = env, horizon = int(1e4), disable_tqdm=False, print_done_states=False):
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            #dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D
    
    def train(self, iterations = 10, gamma = .9, disable_tqdm = False):
        S,A,R,S2,D = self.collect_samples()
        nb_samples = S.shape[0]
        Qfunctions = []
        nb_actions = env.action_space.n
        SA = np.append(S,A,axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2
            Q = ExtraTreesRegressor()
            Q.fit(SA,value)
            Qfunctions.append(Q)
        return Qfunctions



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
            self.Q = pickle.load(f)
