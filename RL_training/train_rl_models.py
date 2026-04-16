# train_rl_models.py


import time
import os
from RL_env.env_wrapper import EnvMap
from RL_env.env_gen import MapGenerator
from RL_training.rl_models import QLearning, SARSA
from RL_training.rl_utils import visulazie_comparison

class RLTraining:
    MODELS = {
        "Q-Learning": (QLearning, "/AIR/RL_training/q_brain.json",    "/AIR/RL_training/qlearning_training_plot.jpeg"),
        "SARSA":      (SARSA,     "/AIR/RL_training/sarsa_brain.json", "/AIR/RL_training/sarsa_training_plot.jpeg"),
    }

    def __init__(self, seed=47, episodes=10000):
        self.seed     = seed
        self.episodes = episodes
        # Ensure save directory exists
        os.makedirs("/AIR/RL_training/", exist_ok=True)

    def make_env(self): 
        gen = MapGenerator(seed=self.seed)
        grid, spawners = gen.generate_map()
        return EnvMap(grid, spawners, gen.entry_node, gen.exit_node)

    def run_model(self, name): 
        ModelClass, brain_path, _ = self.MODELS[name]
        env   = self.make_env() # Fixed: removed underscore
        model = ModelClass(action_space=env.action_space)
        
        print(f"\nTraining {name}........")
        rewards = []
        start_time = time.time()
        for _ in range(self.episodes):
            state  = env.reset()
            action = model.choose_action(state)
            total_reward = 0
            complete = False
            while not complete:
                next_state, reward, complete = env.step_scoring(action) 
                next_action = model.choose_action(next_state) 
                if name == "Q-Learning":
                    model.learn(state, action, reward, next_state, complete) 
                else:
                    model.learn(state, action, reward, next_state, next_action, complete) 
                state = next_state
                action = next_action
                total_reward += reward
            model.decay_ep()
            rewards.append(total_reward)
        end_time = time.time()
        total_time = end_time - start_time
        model.save_model(brain_path) 
        print(f"{name} Training finished in {total_time:.5f}s")
        return rewards, total_time

    def train_rl_models(self): 
        q_history, q_time = self.run_model("Q-Learning") # Fixed: removed underscore
        s_history, s_time = self.run_model("SARSA")      # Fixed: removed underscore
        visulazie_comparison(q_history, s_history, q_time, s_time)
        return {"Q-Learning": q_time, "SARSA": s_time}

if __name__ == "__main__":
    trainer = RLTraining(seed=47, episodes=10000)
    trainer.train_rl_models()