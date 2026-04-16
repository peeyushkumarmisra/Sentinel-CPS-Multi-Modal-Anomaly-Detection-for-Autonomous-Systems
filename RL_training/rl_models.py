import json
import random
import collections
import numpy as np


class TabularAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.95, ep=1.0, ep_decay=0.999):
        self.action_space = action_space
        self.alpha     = alpha # Learning Rate (new info overrides old)
        self.gamma     = gamma # Discount Factor (future exit reward)
        self.ep        = ep # Exploration Rate (1.0 means 100% random moves)
        self.ep_decay  = ep_decay
        self.min_ep    = 0.01
        self.qtable    = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]) # Sparse Q-Table

    def choose_action(self, state):
        # Epsilon-Greedy strategy
        if random.random() < self.ep:
            return random.choice(self.action_space) # Fully random
        else:
            return np.argmax(self.qtable[state]) # # Exploits best known path

    def decay_ep(self): # Will be called at the end of each episode
        self.ep = max(self.min_ep, self.ep * self.ep_decay)

    def save_model(self, filename): # Serialises the sparse Q-table to JSON
        serializable_qtable = {str(k): v for k, v in self.qtable.items()}
        with open(filename, 'w') as f:
            json.dump(serializable_qtable, f)
        print(f"Model saved to {filename}")

class QLearning(TabularAgent):
    """
    Off-policy TD control.
    Updates using the max Q-value of the next state (greedy target),
    regardless of which action is actually taken next.
    """

    def learn(self, state, action, reward, next_state, complete):
        curr_q = self.qtable[state][action]
        target = reward if complete else reward + self.gamma * max(self.qtable[next_state])
        self.qtable[state][action] += self.alpha * (target - curr_q)


class SARSA(TabularAgent):
    """
    On-policy TD control.
    Updates using the Q-value of the actual next action taken,
    making it more conservative than Q-Learning.
    """

    def learn(self, state, action, reward, next_state, next_action, complete):
        curr_q = self.qtable[state][action]
        target = reward if complete else reward + self.gamma * self.qtable[next_state][next_action]
        self.qtable[state][action] += self.alpha * (target - curr_q)