"""
File: env_gen.py

This script implements a procedural map generator that creates a constrained 10x10 grid
environment for Reinforcement Learning agents.
  1. MapGenerator:
    * init: Initializes the random seeds for reproducibility and defines fixed boundary points,
            including the entry (start) and exit nodes.
    * generate_map:
        * Creates a continuous "spine" or primary path from the inner start to the inner exit
          using a random walk limited to downward and rightward moves.
        * Expands the path by randomly adding valid neighboring cells until a specific threshold
          (38 inner cells) is met.
        * Identifies 10 random locations along the path to act as "spawners" for assets or threats.
        * Returns the final binary grid (1 for path, 0 for wall) and spawner coordinates.

    * visualize: Uses Matplotlib to render the generated map with a color-coded scheme: 
        * Grey for walls, 
        * Yellow for the path,
        * Orange for spawners,
        * Red for entry, 
        * Green for exit. 
        The resulting image is saved as 'map.png'.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class MapGenerator:
    def __init__(self, seed=30): 
        random.seed(seed)
        np.random.seed(seed)
        
        self.grid_size = 10
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Fixing few block points
        self.entry_node = (0, 1)
        self.exit_node = (9, 8)
        self.inner_start = (1, 1)
        self.inner_exit = (8, 8)
        self.path_cells = set()
        self.spawners = []

    def generate_map(self):
        # Genrating a continuous "spine" from (1,1) to (8,8)
        curr = list(self.inner_start)
        self.path_cells.add(tuple(curr))
        
        while curr != list(self.inner_exit):
            if curr[0] == 8: 
                curr[1] += 1
            elif curr[1] == 8: 
                curr[0] += 1
            else:
                if random.random() < 0.5: curr[0] += 1
                else: curr[1] += 1
            self.path_cells.add(tuple(curr))

        # Growing the path until 38 inner cells
        while len(self.path_cells) < 38:
            # Picking a random cell from our path
            r, c = random.choice(list(self.path_cells))
            
            # Checking its 4 neighbors
            neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
            
            # Checking Constraints (Must be in the 1-8 inner boundary)
            valid_neighbors = [(nr, nc) for nr, nc in neighbors if 1 <= nr <= 8 and 1 <= nc <= 8]
            
            if valid_neighbors:
                self.path_cells.add(random.choice(valid_neighbors))

        # Applying the 38 inner cells to the grid
        for r, c in self.path_cells:
            self.grid[r, c] = 1
            
        # Fixing Entry and Exit
        self.grid[self.entry_node] = 1
        self.grid[self.exit_node] = 1
        
        # Selecting 10 spnawer loaction (Excluding the entry and exit nodes)
        self.spawners = random.sample(list(self.path_cells), 10)
        return self.grid, self.spawners



    def visualize(self):
        # Map: 0=Grey (Wall), 1=Yellow (Path), 2=Orange (Spawner), 3=Red (Start), 4=Green (Exit)
        vis_grid = np.copy(self.grid)
        
        for r, c in self.spawners:
            vis_grid[r, c] = 2
            
        vis_grid[self.entry_node] = 3
        vis_grid[self.exit_node] = 4

        # Custom colors: Grey (wall), Yellow (path), Orange (Spawners), Red (Entry), Green (Exit)
        cmap = ListedColormap(['#808080', '#FFD700', "#FF9100", '#FF0000','#00FF00'])
        
        plt.figure(figsize=(6, 6))
        plt.pcolor(vis_grid[::-1], cmap=cmap, edgecolors='black', linewidths=2)
        plt.title("OT Security Patrol Route (40 Active Blocks)")
        
        # Grid
        plt.xticks(np.arange(0.5, 10.5, 1), range(10))
        plt.yticks(np.arange(0.5, 10.5, 1), reversed(range(10)))
        plt.gca().xaxis.tick_top()
        plt.savefig("/AIR/map.png", dpi=300, bbox_inches='tight')
        print("Visual map saved")



if __name__ == "__main__":
    env = MapGenerator(seed=47)  # Changeable Seed ***************************
    maze, spawners = env.generate_map()
    print(f"Total usable blocks (should be 40): {np.sum(maze)}")
    print(f"Spawner Locations: {spawners}")
    env.visualize()