"""
File: env_wrapper.py

This script acts as the interface between the Reinforcement Learning agents and the procedural
map, handling state transitions, move validation, and reward calculations.
1. EnvMap:
    * init: Initializes the environment with the grid, spawner locations,entry, and exit points. 
            It defines the discrete action space (Up, Down, Left, Right).
    * reset:    Prepares the environment for a new episode by placing the agent atthe entry node
                and clearing the "visited" bitmask and step counter.
    * getstate: Returns the current state as a tuple containing the agent's $(x, y)$coordinates
                and the integer representation of the bitmask tracking visited spawners.
    * step_scoring:
        * Translates a discrete action into grid movement.
        * Performs collision detection against boundaries and walls (with a -5 penalty).
        * Manages a bitmask to track which of the 10 spawners have been visited,
          granting a +50 reward for each new spawner discovered.
        * Calculates the final reward upon reaching the exit: a massive +500bonus if all spawner
          are visited, or a proportional penalty for missed targets.
        * Enforces a 250-step limit to prevent infinite loops during training.
"""

class EnvMap:
    def __init__(self, grid, spanwers, entry, exit):
        self.grid = grid
        self.spnawers = spanwers
        self.entry = entry
        self.exit = exit

        # Action Map 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = [0,1,2,3]
        self.action_dic = {0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

    def reset(self): # For restting the env for new training
        self.agnet_pos = list(self.entry)
        self.visited = 0 # 0000000000 in binary
        self.steps = 0
        return self.getstate()
    
    def getstate(self): # Returns state (row, col, bitmask)
        return (self.agnet_pos[0], self.agnet_pos[1], self.visited)
    
    def step_scoring (self,action): # For moving and score calc and also sate updation
        self.steps += 1
        r, c = self.agnet_pos
        dr, dc = self.action_dic[action]
        nr, nc = r + dr, c + dc # New row and col
        reward = -1 # Default step penalty
        complete = False

        # Wall Check
        if (nr < 0 or nr >= 10 or nc < 0 or nc >= 10 or self.grid[nr, nc] == 0):
            reward = -5 # Bump on wall penality
            return self.getstate(), reward, complete
        
        # Valid Move
        self.agnet_pos = [nr,nc]
        pos_tuple = tuple(self.agnet_pos)

        # Spawner Check
        if pos_tuple in self.spnawers:
            idx = self.spnawers.index(pos_tuple)

            # Verfying if it is visited before or not
            if not (self.visited & (1 << idx)):
                self.visited |= (1 << idx) # Flipping the bit to 1
                reward = 50 # Big reward for fresh data

        # At Exit
        if pos_tuple == self.exit:
            complete = True
            if self.visited == 1023: # Making sure it visits all spanwer location
                reward += 500 # Perfect Score
            else: # Proportional reward for better evaluation
                visit_count = bin(self.visited).count("1")
                missed = len(self.spnawers) - visit_count
                reward += max(0,100 - (missed * 10))

        # If stuck in loop
        if self.steps > 250:
            complete = True

        return self.getstate(), reward, complete