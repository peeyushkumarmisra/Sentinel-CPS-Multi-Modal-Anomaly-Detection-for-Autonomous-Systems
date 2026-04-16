# rl_models.py

import json
import random
import collections
import numpy as np

class RLModels:
    def __init__(self, action_space, alpha=0.1, gamma=0.95, ep=1.0, ep_decay=0.999):
        self.action_space = action_space
        self.alpha     = alpha 
        self.gamma     = gamma 
        self.ep        = ep 
        self.ep_decay  = ep_decay
        self.min_ep    = 0.01
        self.qtable    = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]) 

    def choose_action(self, state):
        if random.random() < self.ep:
            return random.choice(self.action_space) 
        else:
            return np.argmax(self.qtable[state]) 

    def decay_ep(self): 
        self.ep = max(self.min_ep, self.ep * self.ep_decay)

    def save_model(self, filename): 
        serializable_qtable = {str(k): v for k, v in self.qtable.items()}
        with open(filename, 'w') as f:
            json.dump(serializable_qtable, f)
        print(f"Model saved to {filename}")

class QLearning(RLModels):
    def learn(self, state, action, reward, next_state, complete):
        curr_q = self.qtable[state][action]
        target = reward if complete else reward + self.gamma * max(self.qtable[next_state])
        self.qtable[state][action] += self.alpha * (target - curr_q)

class SARSA(RLModels):
    def learn(self, state, action, reward, next_state, next_action, complete):
        curr_q = self.qtable[state][action]
        target = reward if complete else reward + self.gamma * self.qtable[next_state][next_action]
        self.qtable[state][action] += self.alpha * (target - curr_q)