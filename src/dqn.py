from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn 
import random
import numpy as np
from tqdm import tqdm
import argparse
from evaluate import evaluate_HIV

parser = argparse.ArgumentParser(description="Configurer les paramètres pour l'entraînement du modèle.")

# Étape 2: Ajout des arguments
parser.add_argument('--nb_actions', type=int, help='Nombre d\'actions.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Taux d\'apprentissage.')
parser.add_argument('--gamma', type=float, default=0.99, help='Facteur de remise gamma.')
parser.add_argument('--buffer_size', type=int, default=1000000, help='Taille du buffer.')
parser.add_argument('--epsilon_min', type=float, default=0.01, help='Epsilon minimum pour l\'exploration.')
parser.add_argument('--epsilon_max', type=float, default=1.0, help='Epsilon maximum pour l\'exploration.')
parser.add_argument('--epsilon_decay_period', type=int, default=1000, help='Période de décroissance pour epsilon.')
parser.add_argument('--epsilon_delay_decay', type=int, default=4000, help='Délai avant de commencer la décroissance d\'epsilon.')
parser.add_argument('--batch_size', type=int, default=1024, help='Taille du batch pour l\'entraînement.')
parser.add_argument('--gradient_steps', type=int, default=1, help='Nombre d\'étapes de gradient par mise à jour.')
parser.add_argument('--update_target_strategy', type=str, default='replace', choices=['replace', 'ema'], help='Stratégie de mise à jour du modèle cible.')
parser.add_argument('--update_target_freq', type=int, default=400, help='Fréquence de mise à jour du modèle cible.')
parser.add_argument('--update_target_tau', type=float, default=0.005, help='Tau pour la mise à jour EMA du modèle cible.')
parser.add_argument('--monitoring_nb_trials', type=int, default=1, help='Nombre d\'essais pour le monitoring.')
parser.add_argument('--monitoring_freq', type=int, default=50, help='Fréquence du monitoring.')
parser.add_argument('--save_every', type=int, default=10, help='Fréquence de sauvegarde du modèle.')
parser.add_argument('--depth', type=int, default=2, help='profondeur du réseau de neurones.')
parser.add_argument('--nb_neurons', type=int, default=24, help='Nombre de neurones dans la couche cachée.')

args = parser.parse_args()

device = torch.device("cpu")
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

def greedy_action_dqn(network, state):
    device = "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class ProjectAgent:

    def __init__(self, config, model):
        device = "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.path = 'src/DQNagent.pt'

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(env.action_space.n)
        else:
            return greedy_action_dqn(self.model, observation)

    def save(self, path):
        print("Saving model ...")
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    }, path)


    def load(self):
        self.model.load_state_dict(torch.load(self.path, map_location='cpu'))
        self.model.eval()
        


    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score = -np.inf

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action_dqn(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", int(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
                score = evaluate_HIV(self, env, nb_episode=1)
                if score > best_score:
                    best_score = score
            else:
                state = next_state

        with open('results_dqn.txt', 'a') as file:
            file.write(f'nb neurones: {args.nb_neurons}, depth: {args.depth}, best score: {best_score}\n')

        return episode_return
        


        

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
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
    

state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 

class DQM_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth = 2):
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
    

# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': args.epsilon_decay_period,
          'epsilon_delay_decay': 20,
          'batch_size': args.batch_size,
          'depth': args.depth,
          'nb_neurons': args.nb_neurons,
          }

model = DQM_model(6, args.nb_neurons, 4, args.depth)

# Train agent
agent = ProjectAgent(config, model)
scores = agent.train(env, 2000)
#agent.save('DQNagent.pt')