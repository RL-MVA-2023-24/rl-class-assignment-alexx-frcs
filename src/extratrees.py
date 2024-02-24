from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn 
import random
import numpy as np
import locale
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import pickle
from evaluate import evaluate_HIV
import os

locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français


device = torch.device("cpu")
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

    
def greedy_action_fqi(Q,s,nb_actions):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)

class ProjectAgent():

    def __init__(self):
        self.Q = np.zeros((3000, env.action_space.n))

    def collect_samples(self, horizon, act_randomness,  env = env, disable_tqdm=False, print_done_states=False):
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            if np.random.randn() < act_randomness:
                a = env.action_space.sample()
            else:
                a = self.act(s)
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
    
    def train(self, S, A, R, S2, D, iterations, gamma, disable_tqdm = False):
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
            Q = ExtraTreesRegressor(n_estimators=50, min_samples_split=2)
            Q.fit(SA,value)
            Qfunctions.append(Q)
        self.Q = Q
        return Qfunctions


    def act(self, observation, use_random=False):
            # print(observation)
            if use_random or iteration == 0:
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


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200 # 200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
best_score = -1

# Initialization of the agent. Replace DummyAgent with your custom agent implementation.
agent = ProjectAgent()

nb_iter = 100
gamma = 0.9
nb_iter_fit = 500
nb_sample_per_it = 3_000
for iteration in range(nb_iter):
    print('iteration', iteration)
    randomness_action = 1 - iteration/nb_iter
    S, A, R, S2, D = agent.collect_samples(nb_sample_per_it ,act_randomness=randomness_action) 
    # import pickle
    with open('data.pkl', 'wb') as f:
        pickle.dump((S, A, R, S2, D), f)
    with open('data.pkl', 'rb') as f:
        S, A, R, S2, D = pickle.load(f)
    agent.train(S, A, R, S2, D, nb_iter_fit, gamma) # mean to compute reward
    agent.save()

    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    print(locale.format_string('%d', int(score_agent), grouping=True))

