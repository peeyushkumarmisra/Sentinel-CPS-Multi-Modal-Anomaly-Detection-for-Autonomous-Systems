"""
File: train_all.py
This script serves as the master execution hub for the entire project, orchestrating
the training pipelines for Supervised Learning (SL), Unsupervised Learning (USL),
and Reinforcement Learning (RL) sequentially.
"""

# SUPERVISED LEARNING (SL) IMPORTS
from SL_training.rf_model import run_rf
from SL_training.svm_model import run_svm
from SL_training.sl_utils import comparative_plot_sl

# UNSUPERVISED LEARNING (USL) IMPORTS
from USL_training.train_usl_models import run_usl_pipeline, train_gmm_step, train_dec_step
from USL_training.usl_utils import evaluate_and_plot_usl

# REINFORCEMENT LEARNING (RL) IMPORTS
from RL_training.train_rl_models import run_mission, visualize_comparison


if __name__ == "__main__":
    # SUPERVISED LEARNING PIPELINE
    print("SUPERVISED LEARNING MODELS TRAINING......")
    rf_results = run_rf() # Training Random Forest Model
    svm_results = run_svm() # Training Support Vector Machine Model
    comparative_plot_sl(rf_results, svm_results) # Plotting Comparison
    print("SUPERVISED LEARNING MODELS TRAINING COMPLETE\n")

    # UNSUPERVISED LEARNING PIPELINE
    print("UNSUPERVISED LEARNING MODELS TRAINING......")
    true_labels, class_names = run_usl_pipeline() #
    gmm_preds, gmm_time = train_gmm_step(true_labels, n_clusters=5) # Training Gaussian Mixture Model
    dec_preds, dec_time = train_dec_step(true_labels, n_clusters=5) # Training Deep Embedding Clustering Model
    evaluate_and_plot_usl(true_labels, gmm_preds, dec_preds, class_names, gmm_time, dec_time) # Plotting Comparison
    print("UNSUPERVISED LEARNING MODELS TRAINING COMPLETE\n")

    # REINFORCEMENT LEARNING PIPELINE
    print("REINFORCEMENT LEARNING MODELS TRAINING......")
    q_time, q_history = run_mission(name = "Q-Learning")
    s_time, s_history = run_mission(name = "SARSA")
    visualize_comparison(q_history, s_history, q_time, s_time)
    print("REINFORCEMENT LEARNING MODELS TRAINING COMPLETE")
