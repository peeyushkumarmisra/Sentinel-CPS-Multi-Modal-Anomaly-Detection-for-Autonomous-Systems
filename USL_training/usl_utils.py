"""
File: usl_utils.py

A utility toolkit containing helper functions for file system scanning, data
visualization, and cluster-to-label alignment for unsupervised learning evaluation.
    1. plot_loss_curve: Generates and saves a standardized Matplotlib line graph showing training 
                        vs validation loss across epochs.
    2. scan_images: Recursively traverses a directory to find files with a specific extension (default .png)
                    and returns a dictionary mapping filenames to absolute paths.
    3. map_clusters_to_truth:   Uses the Hungarian Algorithm (Linear Sum Assignment) to solve the 
                                label switching problem, aligning unsupervised cluster indices with
                                the most statistically likely ground-truth labels.
    4. generate_comparison_dashboard:   Creates a comprehensive 2x2 visual report comparing GMM and DEC
                                        models across three performance metrics (ARI, NMI, Homogeneity),
                                        training speed, and confusion matrices.
    5. evaluate_and_plot_usl:   A high-level wrapper that calculates statistical metrics
                                (Adjusted Rand Index, Normalized Mutual Information, etc.) for both
                                models and triggers the dashboard generation.
"""

import os
import torch
import random 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, confusion_matrix

def plot_loss_curve(history_dict, title, ylabel, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["train_loss"], label='Training Loss',   color='#1f77b4', linewidth=2)
    plt.plot(history_dict["val_loss"],   label='Validation Loss', color='#ff7f0e', linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Loss curve saved → {plot_path}")



def scan_images(base_dir, ext='.png'):
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(ext):
                path_map[f] = os.path.join(root, f)
    return path_map



# Aligns unsupervised cluster IDs with ground truth (Using Hungarian algorithm)
def map_clusters_to_truth(true_labels, pred_labels): 
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {c: r for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in pred_labels])



def evaluate_and_plot_usl(true_labels, gmm_preds, dec_preds, class_names, gmm_time, dec_time, gmm_conf, dec_conf, plot_path):
    print("Evaluating models and generating 2x2 Performance Dashboard...")
    
    # Calculating all metrics and store them logically
    metrics = {'GMM': {}, 'DEC': {}}
    for name, preds, t in [('GMM', gmm_preds, gmm_time), ('DEC', dec_preds, dec_time)]:
        metrics[name]['ARI'] = adjusted_rand_score(true_labels, preds)
        metrics[name]['NMI'] = normalized_mutual_info_score(true_labels, preds)
        metrics[name]['Homo'] = homogeneity_score(true_labels, preds)
        metrics[name]['Time'] = t
        metrics[name]['CM'] = confusion_matrix(true_labels, preds)

    # Calculating the average confidences for each methods
    gmm_avg_conf = np.mean(gmm_conf)
    dec_avg_conf = np.mean(dec_conf)

    # 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.set_theme(style="whitegrid")
    
    # Top Left: Performance Metrics Bar Chart
    ax1 = axes[0, 0]
    x = np.arange(3)
    width = 0.35
    gmm_scores = [metrics['GMM']['ARI'], metrics['GMM']['NMI'], metrics['GMM']['Homo']]
    dec_scores = [metrics['DEC']['ARI'], metrics['DEC']['NMI'], metrics['DEC']['Homo']]
    ax1.bar(x - width/2, gmm_scores, width, label='GMM', color='#2ca02c')
    ax1.bar(x + width/2, dec_scores, width, label='DEC', color='#1f77b4')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['ARI', 'NMI', 'Homogeneity'])
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    # Add data labels
    for rect in ax1.patches:
        ax1.annotate(f'{rect.get_height():.4f}', 
                     (rect.get_x() + rect.get_width() / 2, rect.get_height()), 
                     ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Top Right: Training Time Comparison
    ax2 = axes[0, 1]
    rects = ax2.bar(['GMM', 'DEC'], [metrics['GMM']['Time'], metrics['DEC']['Time']],
                    width=0.4, color=['#2ca02c', '#1f77b4'])
    ax2.set_ylabel('Time (Seconds)')
    ax2.set_title('Training Time Comparison')
    # Add data labels
    for rect in rects:
        ax2.annotate(f'{rect.get_height():.4f} s', 
                     (rect.get_x() + rect.get_width() / 2, rect.get_height()), 
                     ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Bottom Row: Confusion Matrices
    sns.heatmap(metrics['GMM']['CM'], annot=True, fmt='d', cmap='Greens', ax=axes[1, 0],
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    axes[1, 0].set_title(f'GMM Confusion Matrix\n(Avg Conf: {gmm_avg_conf:.4%})')
    
    sns.heatmap(metrics['DEC']['CM'], annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    axes[1, 1].set_title(f'DEC Confusion Matrix\n(Avg Conf: {dec_avg_conf:.4%})')

    # Save and Cleanup
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Dashboard generated and saved to: {plot_path}")



# FOR REPRODUCIBILITY
class ReproducibilityManager:
    """Handles global seeding and enforces deterministic behavior across all libraries."""
    @staticmethod
    def reproducible(seed=42):
        print(f"Locking down randomness with seed: {seed}")
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# standard worker initialization for DataLoader multi-processing
def seed_worker(worker_id):
    """Ensures each worker in DataLoader generates deterministic batches."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
