"""
File: train_all.py
This script serves as the master execution hub for the entire project, orchestrating
the training pipelines for Supervised Learning (SL), Unsupervised Learning (USL),
and Reinforcement Learning (RL) sequentially.
"""

from pathlib import Path
from Env.mapping import map_csv, map_imgs
from SL.train_sl_models import SLTraining
from USL.train_usl_models import USLTraining
from RL.train_rl_models import RLTraining


if __name__ == "__main__":

    # CREATING FOLDERS
    IMG_PATH = "DATA/DATASET"
    CSV_PATH = "DATA/SENSOR_STATS.csv"
    PLOT_PATH = "PLOTS/"
    MODEL_PATH = "MODELS/"
    SEED = 47
    if not Path(CSV_PATH).exists(): map_csv()
    if not Path(IMG_PATH).exists(): map_imgs()
    Path(PLOT_PATH).mkdir(parents=True, exist_ok=True) # For All Plots
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True) # For All Models

    # SUPERVISED LEARNING PIPELINE
    print("SUPERVISED LEARNING MODELS TRAINING......")
    sl_trainer = SLTraining(seed=SEED, data_path=CSV_PATH, plot_dir = PLOT_PATH, model_dir = MODEL_PATH)
    sl_trainer.train_sl_models()
    print("SUPERVISED LEARNING MODELS TRAINING COMPLETE\n")

    # UNSUPERVISED LEARNING PIPELINE
    print("UNSUPERVISED LEARNING MODELS TRAINING......")
    usl_trainer = USLTraining(seed=SEED, data_dir=IMG_PATH, plot_dir = PLOT_PATH, model_dir = MODEL_PATH)
    usl_trainer.train_usl_models()
    print("UNSUPERVISED LEARNING MODELS TRAINING COMPLETE\n")

    # REINFORCEMENT LEARNING PIPELINE
    print("REINFORCEMENT LEARNING MODELS TRAINING......")
    rl_trainer = RLTraining(episodes=5000, seed=SEED, plot_dir = PLOT_PATH, model_dir = MODEL_PATH)
    rl_trainer.train_rl_models()
    print("REINFORCEMENT LEARNING MODELS TRAINING COMPLETE")
