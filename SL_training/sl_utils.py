#sl_utils.py

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate_plot(y_test, rf_pred, svm_pred, rf_time, svm_time):
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, average='macro', zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, average='macro', zero_division=0)
    rf_cm = confusion_matrix(y_test, rf_pred)
    
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_prec = precision_score(y_test, svm_pred, average='macro', zero_division=0)
    svm_rec = recall_score(y_test, svm_pred, average='macro', zero_division=0)
    svm_cm = confusion_matrix(y_test, svm_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metric Comparison
    ax_metrics = axes[0, 0]
    labels = ['Accuracy', 'Precision', 'Recall']
    rf_scores = [rf_acc, rf_prec, rf_rec]
    svm_scores = [svm_acc, svm_prec, svm_rec]
    x = np.arange(len(labels))
    width = 0.35
    
    ax_metrics.bar(x - width/2, rf_scores, width, label='Random Forest', color='#2ca02c')
    ax_metrics.bar(x + width/2, svm_scores, width, label='SVM', color='#1f77b4')
    ax_metrics.set_ylabel('Score')
    ax_metrics.set_title('Performance Metrics Comparison')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(labels)
    ax_metrics.set_ylim(0, 1.15)
    ax_metrics.legend()
    
    for i, (rf_val, svm_val) in enumerate(zip(rf_scores, svm_scores)):
        ax_metrics.text(i - width/2, rf_val + 0.02, f'{rf_val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax_metrics.text(i + width/2, svm_val + 0.02, f'{svm_val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Training Time Comparison
    ax_time = axes[0, 1]
    models = ['Random Forest', 'SVM']
    times = [rf_time, svm_time]
    
    bars = ax_time.bar(models, times, color=['#2ca02c', '#1f77b4'], width=0.4)
    ax_time.set_ylabel('Time (Seconds)')
    ax_time.set_title('Training Time Comparison')
    
    for bar in bars:
        yval = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.02), f'{yval:.5f} s', ha='center', va='bottom', fontweight='bold')

    # Confusion Matrices
    ax_rf_cm = axes[1, 0]
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=ax_rf_cm)
    ax_rf_cm.set_title('Random Forest Confusion Matrix')
    ax_rf_cm.set_xlabel('Predicted Label')
    ax_rf_cm.set_ylabel('True Label')
    
    ax_svm_cm = axes[1, 1]
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', ax=ax_svm_cm)
    ax_svm_cm.set_title('SVM Confusion Matrix')
    ax_svm_cm.set_xlabel('Predicted Label')
    ax_svm_cm.set_ylabel('True Label')
    
    plt.tight_layout()
    
    # Ensure directory exists before saving
    os.makedirs('SL_training', exist_ok=True)
    plt.savefig('SL_training/sl_model_comparative_summary.jpeg', dpi=300)
    plt.close()