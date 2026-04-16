"""
File: train_rl_models.py

This script manages the execution and performance comparison of Reinforcement Learning agents,
providing a structured way to train both Q-Learning and SARSA models on a consistent environment map.
  1. run_mission:
    * Sets up the MapGenerator and EnvMap with a fixed seed to ensure fair benchmarking.
    * Dynamically initializes the chosen agent (Q-Learning or SARSA).
    * Executes a 10,000-episode training loop, handling the specific state-action logic required
      for on-policy (SARSA) vs. off-policy (Q-Learning) updates.
    * Serializes the trained agent's Q-table into a JSON "brain" file for later inference.
    * Returns the total training time and reward history.
  2. visualize_comparison:
    * Creates a dual-pane analytical dashboard to evaluate model efficiency.
    * Learning Convergence: Uses a 100-episode moving average to plot and comparethe stability
      and speed of reward maximization for both algorithms.
    * Training Time: Provides a bar chart comparison of the wall-clock time required to complete
      10,000 episodes for each model.
"""



import time
import numpy as np
import matplotlib.pyplot as pt
from RL_env.env_wrapper import EnvMap
from RL_env.env_gen import MapGenerator
from RL_training.sarsaa_model import SARSA
from RL_training.qlearn_model import QLearning



def run_mission(name):
    # Setup
    seed = 47
    gen = MapGenerator(seed=seed)
    grid, spawners = gen.generate_map()
    env = EnvMap(grid, spawners, gen.entry_node, gen.exit_node)

    if name == "Q-Learning":
        agent = QLearning(action_space=env.action_space)
    elif name == "SARSA":
        agent = SARSA(action_space=env.action_space)
    else:
        raise ValueError(f"ERROR: Must be 'Q-Learning' or 'SARSA'")

    print(f"\n--- Training {name} ---")
    rewards_history = []
    start_time = time.time()

    for e in range(10000):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        complete = False
        while not complete:
            next_state, reward, complete = env.step_scoring(action) # Processes action
            next_action = agent.choose_action(next_state) # Picking action
            if name == "Q-Learning":
                agent.learn(state, action, reward, next_state, complete) # Updating its brain
            else:
                next_action = agent.choose_action(next_state) # Picking action
                agent.learn(state, action, reward, next_state, next_action, complete) # Updating its brain
            state = next_state
            action = next_action
            total_reward += reward
        agent.decay_ep()
        rewards_history.append(total_reward)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"{name} Finished in {total_time:.5f}s")

    # Saving Models
    if name == "Q-Learning":
        agent.save_model("/AIR/RL_training/q_brain.json") 
    elif name == "SARSA":
        agent.save_model("/AIR/RL_training/sarsa_brain.json")
    else:
        raise ValueError(f"ERROR: Must be 'Q-Learning' or 'SARSA'")
    return total_time, rewards_history



def visualize_comparison(q_history, s_history, q_time, s_time):
    window = 100
    q_avg = np.convolve(q_history, np.ones(window)/window, mode='valid')
    s_avg = np.convolve(s_history, np.ones(window)/window, mode='valid')
    fig, (ax1, ax2) = pt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Learning Convergence
    ax1.plot(range(window-1, 10000), q_avg, label='Q-Learning', color='blue', linewidth=2)
    ax1.plot(range(window-1, 10000), s_avg, label='SARSA', color='orange', linewidth=2)
    ax1.set_title("Learning Convergence (Moving Avg)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Training Time Bar Chart
    algorithms = ['Q-Learning', 'SARSA']
    times = [q_time, s_time]
    bars = ax2.bar(algorithms, times, color=['blue', 'orange'], alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.5f}s', ha='center', va='bottom', fontweight='bold')
    ax2.set_title("Total Training Time (10k Episodes)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Seconds")
    ax2.set_ylim(0, max(times) * 1.2)
    pt.tight_layout()
    pt.savefig("/AIR/comparison_results.png", dpi=300)
    print("\nComparison graph saved to /AIR/comparison_results.png")
