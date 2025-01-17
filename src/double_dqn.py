from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import locale
from evaluate import evaluate_HIV
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français


parser = argparse.ArgumentParser(description="Configurer les paramètres pour l'entraînement du modèle.")

# Étape 2: Ajout des arguments
parser.add_argument('--nb_actions', type=int, help='Nombre d\'actions.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Taux d\'apprentissage.')
parser.add_argument('--gamma', type=float, default=0.99, help='Facteur de remise gamma.')
parser.add_argument('--buffer_size', type=int, default=1000000, help='Taille du buffer.')
parser.add_argument('--epsilon_min', type=float, default=0.01, help='Epsilon minimum pour l\'exploration.')
parser.add_argument('--epsilon_max', type=float, default=1.0, help='Epsilon maximum pour l\'exploration.')
parser.add_argument('--epsilon_decay_period', type=int, default=15000, help='Période de décroissance pour epsilon.')
parser.add_argument('--epsilon_delay_decay', type=int, default=4000, help='Délai avant de commencer la décroissance d\'epsilon.')
parser.add_argument('--batch_size', type=int, default=1000, help='Taille du batch pour l\'entraînement.')
parser.add_argument('--gradient_steps', type=int, default=1, help='Nombre d\'étapes de gradient par mise à jour.')
parser.add_argument('--update_target_strategy', type=str, default='replace', choices=['replace', 'ema'], help='Stratégie de mise à jour du modèle cible.')
parser.add_argument('--update_target_freq', type=int, default=400, help='Fréquence de mise à jour du modèle cible.')
parser.add_argument('--update_target_tau', type=float, default=0.005, help='Tau pour la mise à jour EMA du modèle cible.')
parser.add_argument('--monitoring_nb_trials', type=int, default=1, help='Nombre d\'essais pour le monitoring.')
parser.add_argument('--monitoring_freq', type=int, default=50, help='Fréquence du monitoring.')
parser.add_argument('--save_every', type=int, default=10, help='Fréquence de sauvegarde du modèle.')
parser.add_argument('--depth', type=int, default=2, help='profondeur du réseau de neurones.')
parser.add_argument('--nb_neurons', type=int, default=24, help='Nombre de neurones dans la couche cachée.')
parser.add_argument('--random_true', type=bool, default='False', help='random domain')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


env = TimeLimit(
    env=HIVPatient(domain_randomization=args.random_true), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


seed = 42

random.seed(seed)
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

# Étape 3: Extraction des arguments

# Utilisation des valeurs d'arguments
config = {
    'nb_actions': env.action_space.n,
    'learning_rate': args.learning_rate,
    'gamma': args.gamma,
    'buffer_size': args.buffer_size,
    'epsilon_min': args.epsilon_min,
    'epsilon_max': args.epsilon_max,
    'epsilon_decay_period': args.epsilon_decay_period,
    'epsilon_delay_decay': args.epsilon_delay_decay,
    'batch_size': args.batch_size,
    'gradient_steps': args.gradient_steps,
    'update_target_strategy': args.update_target_strategy,
    'update_target_freq': args.update_target_freq,
    'update_target_tau': args.update_target_tau,
    'criterion': torch.nn.SmoothL1Loss(),
    'monitoring_nb_trials': args.monitoring_nb_trials,
    'monitoring_freq': args.monitoring_freq,
    'save_every': args.save_every,
}



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
    
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class double_dqn_agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.monitoring_freq = config['monitoring_freq'] if 'monitoring_freq' in config.keys() else 50

    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in tqdm(range(nb_trials)):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)

            # Step 1: Select action using the online model
            with torch.no_grad():  # Ensuring no gradients are computed for this operation
                # Get actions from the online model for next states
                actions_next = self.model(Y).max(1)[1].unsqueeze(1)

            # Step 2: Gather Q-values from the target model for the selected actions
            QYmax = self.target_model(Y).gather(1, actions_next).squeeze()

            # Compute the target Q-value
            update = R + (1 - D) * (self.gamma * QYmax)

            # Get the current Q-values from the model
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))

            # Calculate loss
            loss = self.criterion(QXA, update.unsqueeze(1))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_model = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0 and episode % self.monitoring_freq == 0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", locale.format_string('%d', int(episode_cum_reward), grouping=True),
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", locale.format_string('%d', int(episode_cum_reward), grouping=True),
                          sep='')
                score_agent: float = evaluate_HIV(agent=self, nb_episode=10)
                print("Score: ", locale.format_string('%d', int(score_agent), grouping=True))

                    
                if score_agent > best_model:
                    self.save()
                    print('New best model !')
                    best_model = score_agent
                print('Current best model: ', best_model)
                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
            
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    
    def act(self, observation, use_random=False):
        if use_random:
            a = np.random.choice(4)
            # print(a)
            return a
        else:
            a = greedy_action(self.model, observation)
            # print(a)
            return a

    def save(self, path = "double_DQN.pt"):
        print("saving")
        if args.random_true:
            path = "double_DQN_random.pt"
            torch.save({
                    'model_state_dict': self.model.state_dict(),
                    }, path)
        else:
            path = "double_DQN.pt"
            torch.save({
                        'model_state_dict': self.model.state_dict(),
                        }, path)
    def load(self):
        print("loading")
        checkpoint = torch.load("double_DQN.pt", map_location=torch.device('cpu'))
        self.model = DQM_model(6, 256, 4, 6).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
# Declare network
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

model = DQM_model(6, args.nb_neurons, 4, args.depth).to(device)

# Train agent
agent = double_dqn_agent(config, model)
ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, 300)
plt.plot(ep_length, label="training episode length")
plt.plot(tot_rewards, label="MC eval of total reward")
plt.legend()
plt.figure()
plt.plot(disc_rewards, label="MC eval of discounted reward")
plt.plot(V0, label="average $max_a Q(s_0)$")
plt.legend();