"""
File: Sarsa_model.py

This script implements a SARSA (State-Action-Reward-State-Action) agent, an on-policy
Reinforcement Learning algorithm that learns a navigation policy based on the specific
actions it actually takes.
  1. SARSA:
    * init: Initializes the agent with an action space, learning rate (alpha), discount factor (gamma),
            and exploration parameters. It uses a sparse Q-table via a dictionary.
    * choose_action:    Employs an Epsilon-Greedy strategy to decide between exploring new moves and
                        exploiting the current best-known path.
    * learn:    The core on-policy update rule. Unlike Q-Learning, SARSA updates the Q-value using
                the reward and the Q-value of the next actual action chosen, making it more
                conservative and aware of the current policy's risks.
    * decay_ep: Progressively reduces the exploration rate after each episode to stabilize the 
                learned policy.
    * save_model:   Serializes the final Q-table into a JSON format for deployment.
  2. visulazie_training:    Creates a performance plot showing raw rewards and a smoothed moving average
                            to track the learning progress and training time efficiency.
Main Execution:
* Generates the map and environment environment using a fixed seed.
* Runs a 10,000-episode training loop where the agent picks actions, observes outcomes, and
  updates its Q-table using the SARSA update rule.
"""

import time
import json
import random
import collections
import numpy as np
import matplotlib.pyplot as pt
from RL_env.env_wrapper import EnvMap
from RL_env.env_gen import MapGenerator



class SARSA:
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
    
    def learn(self, state, action, reward, next_state, next_action, complete):
        curr_q = self.qtable[state][action]
        if complete:
            target = reward # If the episode ends, there is no future state
        else:
            target = reward + self.gamma * self.qtable[next_state][next_action] # looks for max reward
        # Updating the Q val
        self.qtable[state][action] += self.alpha * (target - curr_q)

    def decay_ep(self): # Will be called at the end of each episode
        self.ep = max(self.min_ep, self.ep * self.ep_decay)
    
    def save_model(self, filename="sarsa_trained_brain.json"): #
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
    pt.title(f"SARSA Training Performance\nTotal Training Time: {mins}m {secs:.05f}s", fontsize=14, fontweight='bold')
    pt.xlabel("Episodes", fontsize=12)
    pt.ylabel("Total Reward", fontsize=12)
    pt.legend()
    pt.grid(True, linestyle='--', alpha=0.6)
    pt.savefig("/AIR/sarsa_training_plot.png", dpi=300, bbox_inches='tight')
    print("\nTraining graph saved")


# EXCUTION & TRAINING
if __name__ == '__main__':
    # Initializing
    seed_value = 47 # ********** Should be same as per rl_env
    generator = MapGenerator (seed=seed_value)
    grid, spawners = generator.generate_map()
    env = EnvMap(grid, spawners, generator.entry_node, generator.exit_node)
    agent = SARSA(action_space=env.action_space)
    episodes = 10000
    rewards_history = []
    print(f"Starting SARSA Training for {episodes} episodes")
    start_time = time.time()
    # Training
    for e in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        complete = False
        while not complete:
            next_state, reward, complete = env.step_scoring(action) # Processes action
            next_action = agent.choose_action(next_state) # Picking action
            agent.learn(state, action, reward, next_state, next_action, complete) # Updating its brain
            state = next_state
            action = next_action
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