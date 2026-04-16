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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, confusion_matrix

def plot_loss_curve(history_dict, title, ylabel, save_path):
    """
    Standardized plotting function for deep learning training curves.
    Expects a dictionary with 'train_loss' and 'val_loss' lists.
    """
    # Ensure the target directory exists before attempting to save
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["train_loss"], label='Training Loss',   color='#1f77b4', linewidth=2)
    plt.plot(history_dict["val_loss"],   label='Validation Loss', color='#ff7f0e', linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    
    # Add a light grid for easier reading of loss plateau points
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close() # Free up memory
    print(f"Loss curve saved → {save_path}")

def scan_images(base_dir, ext='.png'):
    """
    Recursively scans a root directory for image files.
    Returns a dictionary mapping the filename to its absolute file path.
    Example: {'pump_01.png': '/data/fantasy_dataset/pumps/pump_01.png'}
    """
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(ext):
                path_map[f] = os.path.join(root, f)
    return path_map

def map_clusters_to_truth(true_labels, pred_labels):
    """Aligns unsupervised cluster IDs with ground truth labels using the Hungarian algorithm."""
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {c: r for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, label) for label in pred_labels])

def generate_comparison_dashboard(metrics_dict, cm_gmm, cm_dec, class_names, save_path="graphs/model_comparison.png"):
    """Generates a 2x2 dashboard comparing GMM and DEC performance."""
    print("Generating 2x2 Performance Dashboard...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.set_theme(style="whitegrid")
    
    # Top Left: Performance Metrics
    ax1 = axes[0, 0]
    x = np.arange(3)
    width = 0.35
    ax1.bar(x - width/2, [metrics_dict['GMM']['ARI'], metrics_dict['GMM']['NMI'], metrics_dict['GMM']['Homo']], width, label='GMM', color='#2ca02c')
    ax1.bar(x + width/2, [metrics_dict['DEC']['ARI'], metrics_dict['DEC']['NMI'], metrics_dict['DEC']['Homo']], width, label='DEC', color='#1f77b4')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['ARI', 'NMI', 'Homogeneity'])
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    for rect in ax1.patches:
        ax1.annotate(f'{rect.get_height():.4f}', (rect.get_x() + rect.get_width() / 2, rect.get_height()), ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Top Right: Training Time
    ax2 = axes[0, 1]
    rects = ax2.bar(['GMM', 'DEC'], [metrics_dict['GMM']['Time'], metrics_dict['DEC']['Time']], width=0.4, color=['#2ca02c', '#1f77b4'])
    ax2.set_ylabel('Time (Seconds)')
    ax2.set_title('Training Time Comparison')
    for rect in rects:
        ax2.annotate(f'{rect.get_height():.4f} s', (rect.get_x() + rect.get_width() / 2, rect.get_height()), ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Bottom Row: Confusion Matrices
    sns.heatmap(cm_gmm, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0], xticklabels=class_names, yticklabels=class_names, cbar=True)
    axes[1, 0].set_title('GMM Confusion Matrix')
    
    sns.heatmap(cm_dec, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], xticklabels=class_names, yticklabels=class_names, cbar=True)
    axes[1, 1].set_title('DEC Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_and_plot_usl(true_labels, gmm_preds, dec_preds, class_names, gmm_time, dec_time, save_path="graphs/model_comparison.png"):
    """
    Calculates ARI, NMI, Homogeneity, and Confusion Matrices for both models,
    then automatically triggers the 2x2 dashboard generation.
    """
    metrics = {'GMM': {}, 'DEC': {}}
    
    for name, preds, t in [('GMM', gmm_preds, gmm_time), ('DEC', dec_preds, dec_time)]:
        metrics[name]['ARI'] = adjusted_rand_score(true_labels, preds)
        metrics[name]['NMI'] = normalized_mutual_info_score(true_labels, preds)
        metrics[name]['Homo'] = homogeneity_score(true_labels, preds)
        metrics[name]['Time'] = t

    cm_gmm = confusion_matrix(true_labels, gmm_preds)
    cm_dec = confusion_matrix(true_labels, dec_preds)

    # Assumes generate_comparison_dashboard is already in usl_utils.py
    generate_comparison_dashboard(metrics, cm_gmm, cm_dec, class_names, save_path)