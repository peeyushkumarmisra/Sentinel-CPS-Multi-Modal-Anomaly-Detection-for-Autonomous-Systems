"""
File: qlearn_agent.py

This script implements a Q-Learning agent that learns to navigate a grid-based environment
through trial and error, optimizing for a maximum cumulative reward.
  1. QLearning:
    * init: Initializes the agent with an action space and hyperparameters (alpha, gamma, epsilon).
            Uses a defaultdict to manage a sparse Q-table.
    * choose_action:    Implements the Epsilon-Greedy strategy to balance exploration (random moves)
                        and exploitation (best-known moves).
    * learn:    Updates the Q-values using the Bellman equation, adjusting the agent's "knowledge" 
                ased on the reward received and the maximum potential future reward.
    * decay_ep: Reduces the exploration rate (epsilon) over time, forcing the agent to rely more
                on its learned policy as training progresses.
    * save_model: Exports the learned Q-table to a JSON file for persistence and future use.
  2. visulazie_training:
    * Generates a line graph of the total rewards per episode.
    * Includes a moving average to smooth out noise and visualize the agent's learning trend
      and convergence.
Main Execution:
    * Sets up the MapGenerator and environment.
    * Runs the training loop for 10,000 episodes, updating the agent's brain at every step and
      logging progress periodically.
"""

import time
import json
import random
import collections
import numpy as np
import matplotlib.pyplot as pt
from RL_env.env_wrapper import EnvMap
from RL_env.env_gen import MapGenerator



class QLearning:
    def __init__(self, action_space, alpha = 0.1, gamma = 0.95, ep = 1.0, ep_decay = 0.999):
        self.action_space = action_space
        self.alpha = alpha # Learning Rate (new info overrides old)
        self.gamma = gamma # Discount Factor (future exit reward)
        self.ep = ep # Exploration Rate (1.0 means 100% random moves)
        self.ep_decay = ep_decay
        self.min_ep = 0.01
        self.qtable = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]) # Sparse Q-Table

    def choose_action(self, state):
        # Epsilon-Greedy strategy
        if random.random() < self.ep:
            return random.choice(self.action_space) # Fully random
        else:
            return np.argmax(self.qtable[state]) # # Exploits best known path
    
    def learn(self, state, action, reward, next_state, complete):
        curr_q = self.qtable[state][action]
        if complete:
            target = reward # If the episode ends, there is no future state
        else:
            target = reward + self.gamma * max(self.qtable[next_state]) # looks for max reward
        # Updating the Q val
        self.qtable[state][action] += self.alpha * (target - curr_q)

    def decay_ep(self): # Will be called at the end of each episode
        self.ep = max(self.min_ep, self.ep * self.ep_decay)

    def save_model(self, filename="qlearn_trained_brain.json"): #
        serializable_qtable = {str(k): v for k, v in self.qtable.items()}
        with open(filename, 'w') as f:
            json.dump(serializable_qtable, f)
        print(f"Model saved to {filename}")



def visulazie_training(rewards, time_taken): # To plot training rewards on line grapgh
    window = 10
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode = 'valid')
    else:
        moving_avg = rewards
    # Time at title
    mins = int(time_taken // 60)
    secs = time_taken % 60
    # Plotting
    pt.figure(figsize=(10,8))
    pt.plot(rewards, alpha =0.3, color='gray', label='Reward')
    pt.plot(range(window-1, len(rewards)), moving_avg, color='blue', linewidth=2, label=f'{window}-Ep Moving Avg')
    pt.title(f"Q-Learning Training Performance\nTotal Training Time: {mins}m {secs:.05f}s", fontsize=14, fontweight='bold')
    pt.xlabel("Episodes", fontsize=12)
    pt.ylabel("Total Reward", fontsize=12)
    pt.legend()
    pt.grid(True, linestyle='--', alpha=0.6)
    pt.savefig("/AIR/qlearning_training_plot.png", dpi=300, bbox_inches='tight')
    print("\nTraining graph saved")


# EXCUTION & TRAINING
if __name__ == '__main__':
    # Initializing
    seed_value = 47 # ********** Should be same as per rl_env
    generator = MapGenerator (seed=seed_value)
    grid, spawners = generator.generate_map()
    env = EnvMap(grid, spawners, generator.entry_node, generator.exit_node)
    agent = QLearning(action_space=env.action_space)
    episodes = 10000
    rewards_history = []
    print(f"Starting Q-Learning Training for {episodes} episodes")
    start_time = time.time()

    # Training
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        complete = False
        while not complete:
            action = agent.choose_action(state) # Picking action
            next_state, reward, complete = env.step_scoring(action) # Processes action
            agent.learn(state, action, reward, next_state, complete) # Updating its brain
            state = next_state
            total_reward += reward
        agent.decay_ep()
        rewards_history.append(total_reward)

        # Progress (Prints every 500 episode)
        if (e + 1) % 500 == 0:
            avg_reward = np.mean(rewards_history[-500:])
            print(f"\nEpisode: {e + 1:05d}")
            print(f"Avg Reward (last 500): {avg_reward:.1f}")
            print(f"Epsilon: {agent.ep:.3f}")
    end_time = time.time()
    total_time = end_time - start_time
    print("Training Finished!")
    visulazie_training(rewards_history, total_time)