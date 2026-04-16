"""
File: sl_utils.py

A utility script for the Supervised Learning (SL) pipeline that handles data preprocessing,
feature scaling, metric calculation, and visualization.
  1. categorize_detailed: Logic-based labeling function that creates granular asset classes
                        (e.g.,splitting "robot_arm" into "Override" vs "Non Override") based
                        on sensor flags.
  2. load_csv_and_scale: 
    * Reads the 'SensorStats.csv' dataset and selects specific technical features.
    * Applies the detailed categorization logic to create the target variable.
    * Performs an 80/20 train/test split with stratification to maintain class balance.
    * Implements StandardScaler to normalize features, fitting on training data and transforming the test set.
  3. get_metrics: Calculates Accuracy, Macro-averaged Precision, Macro-averaged Recall, and the Confusion Matrix.
  4. comparative_plot_sl:
    * Generates a 2x2 summary dashboard saved as a PNG.
    * Includes a bar chart for metric comparison (RF vs SVM), a training time benchmarking plot,
      and individual heatmaps for the confusion matrices of both models.
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def categorize_detailed(row):
    asset = row["Asset Class"]
    if asset == "robot_arm":
        return "Robotic Arm (Override)" if row["Remote Override Flag"] == 1 else "Robotic Arm (Non Override)"
    elif asset == "agv_units":
        return "AGV Unit (Fly)" if row["Airborne Status"] == 1 else "AGV Unit (No Fly)"
    elif asset == "cnc_machine":
        return "CNC Machine"
    elif asset == "plc_controller":
        return "PLC Controller"
    elif asset == "drone":
        return "Drone"
    return "Unknown"


def load_csv_and_scale():
    # Loading Data
    df = pd.read_csv('data/SensorStats.csv')
    
    # Data Prep
    features = [
        'VOC Emissions', 'Acoustic / Vibration', 'CPU Utilization', 'Mechanical Load',
        'Axis Elevation / Altitude', 'Actuator Torque', 'Thermal Load', 'Compromise Risk Index'
    ] # Features used in training
    df['Detailed_Class'] = df.apply(categorize_detailed, axis=1) # Creatred a further classfication
    x = df[features]
    y = df['Detailed_Class']
    x = x.fillna(0) # Replacing missing sensor packets with 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=47, stratify=y)

    # Scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    return x_train_scaled, x_test_scaled, y_train, y_test


def get_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, cm


def comparative_plot_sl(rf_metrics, svm_metrics):
    # Unpacking the data
    rf_acc, rf_prec, rf_rec, rf_cm, rf_time = rf_metrics
    svm_acc, svm_prec, svm_rec, svm_cm, svm_time = svm_metrics
    
    # Plot
    fig, axes = pt.subplots(2, 2, figsize=(12, 10)) # Creating 2x2 grid
    
    # Metric Comparison Bar Chart
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
    # Adding exact values on top of bars
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
    
    # Add exact time
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
    
    pt.tight_layout()
    pt.savefig('SL_training/sl_model_comparative_summary.png', dpi=300)
    pt.close()