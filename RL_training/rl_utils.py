# rl_utils.py

import numpy as np
import matplotlib.pyplot as pt


def visulazie_training(rewards, time_taken, title, save_path): # To plot training rewards on line grapgh
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
    pt.title(f"{title}\nTotal Training Time: {mins}m {secs:.5f}s", fontsize=14, fontweight='bold')
    pt.xlabel("Episodes", fontsize=12)
    pt.ylabel("Total Reward", fontsize=12)
    pt.legend()
    pt.grid(True, linestyle='--', alpha=0.6)
    pt.savefig(save_path, dpi=300, bbox_inches='tight')
    pt.close()
    print("\nTraining graph saved")


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