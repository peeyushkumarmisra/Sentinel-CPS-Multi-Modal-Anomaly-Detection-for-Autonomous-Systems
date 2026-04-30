"""
File: rl_inference.py
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from pathlib import Path



class RLInference:
    MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, model_dir, qtable_file):
        model_dir     = Path(model_dir)
        raw           = json.loads((model_dir / qtable_file).read_text())
        self.qtable   = self.parse_qtable(raw)
        print(f"Q-table loaded: {qtable_file} ({len(self.qtable)} states)")

    @staticmethod
    def parse_qtable(raw_dict):
        parsed = {}
        for k, v in raw_dict.items():
            try:
                state = tuple(map(int, k.strip("()").split(',')))
                parsed[state] = v
            except:
                continue
        return parsed

    def run(self, env_map, animate, gif_path):
        t0 = time.perf_counter()
        current_node = env_map.entry_node
        visited      = 0
        steps        = 0
        rewards      = 0
        path         = [(current_node, visited)]

        while current_node != env_map.exit_node and steps < 250:
            steps   += 1
            state    = (current_node[0], current_node[1], visited)
            q_values = self.qtable.get(state)
            action   = int(np.argmax(q_values)) if q_values is not None else np.random.randint(0, 4)
            dy, dx = self.MOVES[action]
            nr, nc = current_node[0] + dy, current_node[1] + dx
            if nr < 0 or nr >= 10 or nc < 0 or nc >= 10 or env_map.grid[nr, nc] == 0:
                rewards -= 5
                continue
            current_node = (nr, nc)
            if current_node in env_map.spawners:
                idx = env_map.spawners.index(current_node)
                if not (visited & (1 << idx)):
                    visited |= (1 << idx)
                    rewards += 50
            if current_node == env_map.exit_node:
                if visited == (1 << len(env_map.spawners)) - 1:
                    rewards += 500
                else:
                    missed   = len(env_map.spawners) - bin(visited).count("1")
                    rewards += max(0, 100 - (missed * 10))
            else:
                rewards -= 1
            path.append((current_node, visited))
        if animate:
            animate_agent(env_map, path, gif_path)
        elapsed = time.perf_counter() - t0
        print(f"[RL] {steps} steps | reward: {rewards} | elapsed: {elapsed:.4f}s")
        return [p[0] for p in path], rewards, elapsed
    

def animate_agent(env_map, path, gif_path):
    cmap = ListedColormap(['#808080', '#FFD700', "#FF9100", '#FF0000', '#00FF00'])
    fig, ax = plt.subplots(figsize=(6, 6))

    vis_grid = np.copy(env_map.grid)
    for r, c in env_map.spawners:
        vis_grid[r, c] = 2
    vis_grid[env_map.entry_node] = 3
    vis_grid[env_map.exit_node]  = 4

    ax.pcolor(vis_grid[::-1], cmap=cmap, edgecolors='black', linewidths=2)
    ax.set_xticks(np.arange(0.5, 10.5, 1));  ax.set_xticklabels(range(10))
    ax.set_yticks(np.arange(0.5, 10.5, 1));  ax.set_yticklabels(reversed(range(10)))
    ax.xaxis.tick_top()

    ims = []
    for step_idx, (node, visited) in enumerate(path):
        x = node[1] + 0.5
        y = (9 - node[0]) + 0.5
        circle, = ax.plot(x, y, marker='o', color='#0000FF', markersize=18, animated=True)
        title   = ax.text(
            0.5, 1.05,
            f"Step {step_idx+1} | Visited: {bin(visited).count('1')}/10",
            transform=ax.transAxes, ha="center",
            fontsize=14, fontweight='bold', animated=True
        )
        ims.append([circle, title])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    ani.save(gif_path, writer='pillow')
    plt.close()