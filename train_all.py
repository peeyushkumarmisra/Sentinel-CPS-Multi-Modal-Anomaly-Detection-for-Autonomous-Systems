"""
File: train_all.py
This script serves as the master execution hub for the entire project, orchestrating
the training pipelines for Supervised Learning (SL), Unsupervised Learning (USL),
and Reinforcement Learning (RL) sequentially.
"""

#from SL_training.train_sl_models import SLTraining
from USL_training.train_usl_models import USLTraining
#from RL_training.train_rl_models import RLTraining


if __name__ == "__main__":
    # SUPERVISED LEARNING PIPELINE
    print("SUPERVISED LEARNING MODELS TRAINING......")
    #sl_trainer = SLTraining(data_path='data/SensorStats.csv', seed=47)
    #sl_trainer.train_sl_models()
    print("SUPERVISED LEARNING MODELS TRAINING COMPLETE\n")

    # UNSUPERVISED LEARNING PIPELINE
    print("UNSUPERVISED LEARNING MODELS TRAINING......")
    usl_trainer = USLTraining(data_dir="data/fantasy_dataset", n_clusters=5, seed=47)
    usl_trainer.train_usl_models()
    print("UNSUPERVISED LEARNING MODELS TRAINING COMPLETE\n")

    # REINFORCEMENT LEARNING PIPELINE
    print("REINFORCEMENT LEARNING MODELS TRAINING......")
    #rl_trainer = RLTraining(seed=47, episodes=10000)
    #rl_trainer.train_rl_models()
    print("REINFORCEMENT LEARNING MODELS TRAINING COMPLETE")
