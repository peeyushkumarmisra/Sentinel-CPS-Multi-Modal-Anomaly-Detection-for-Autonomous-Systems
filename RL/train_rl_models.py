# train_rl_models.py

import os
import time
from Env.env_scoring import EnvMap
from Env.env_gen import MapGenerator
from RL.rl_models import QLearning, SARSA
from RL.rl_utils import visulazie_comparison, visualize_training

class RLTraining:
    def __init__(self,episodes, seed, plot_dir, model_dir):
        self.episodes = episodes
        self.seed = seed
        self.plot_dir = plot_dir
        self.model_dir = model_dir
        self.model_classes = {"q": QLearning, "sarsa": SARSA}

    def make_env(self):  # Env
        gen = MapGenerator(seed=self.seed)
        grid, spawners = gen.generate_map()
        print(f"Spawner Locations: {spawners}")
        gen.visualize()
        return EnvMap(grid, spawners, gen.entry_node, gen.exit_node)

    def run_model(self, name): 
        ModelClass = self.model_classes[name]
        env   = self.make_env()
        model = ModelClass(action_space=env.action_space, model_dir=self.model_dir)
        
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
                if name == "sarsa":
                    model.learn(state, action, reward, next_state, next_action, complete) 
                else:
                    model.learn(state, action, reward, next_state, complete) 
                state = next_state
                action = next_action
                total_reward += reward
            model.decay_ep()
            rewards.append(total_reward)
        end_time = time.time()
        total_time = end_time - start_time
        model.save_model(name)
        print(f"{name} Training finished in {total_time:.5f}s")
        plot_path = os.path.join(self.plot_dir, f"{name}_training_plot.jpeg")
        visualize_training(rewards, total_time, f"{name.upper()} Model", plot_path)
        return rewards, total_time

    def train_rl_models(self): 
        q_history, q_time = self.run_model("q")
        s_history, s_time = self.run_model("sarsa")
        plot_path = os.path.join(self.plot_dir, "rl_model_comparison.jpeg")
        visulazie_comparison(q_history, s_history, q_time, s_time, plot_path)
        return {"Q-Learning": q_time, "SARSA": s_time}
