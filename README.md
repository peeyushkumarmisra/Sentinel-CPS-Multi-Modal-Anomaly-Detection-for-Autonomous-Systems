# Cyber-Physical Security in Autonomous Systems

## Industrial (OT) Security Analysis Pipeline

This repository contains the main pipeline for training and testing supervised learning, unsupervised learning, and reinforcement learning models on industrial sensor and image data.

---

## Running the Full Pipeline

The entire workflow is controlled by one master script.

### 1. Prepare the Dataset

Place the dataset inside:

```bash
data/fantasy_dataset/
```

### 2. Run the Main Script

From the root folder of the project, run:

```bash
python train_all.py
```

### 3. Output Files

After training is complete:

* Trained model files (`.pt` and `.pkl`) are saved in:

```bash
models/
```

* Performance graphs and comparison dashboards are saved in:

```bash
graphs/
```

---

# Unsupervised Learning (USL) Module

The unsupervised learning branch combines deep learning with traditional machine learning to group industrial assets into clusters with high confidence.

---

## File Overview

### `train_usl_model.py`

Main control file for the unsupervised pipeline.

**Functions:**

* `train_cae_step()`
* `extract_features_step()`
* `train_gmm_step()`
* `train_dec_step()`
* `run_usl_pipeline()`

**Purpose:**
Runs each stage of the USL pipeline in order.

---

### `feature_pipeline.py`

Handles feature extraction and feature reduction.

**Classes / Functions:**

* `FeaturePipeline`
* `DimensionalityReducer`

**Purpose:**

* Loads images
* Extracts deep learning and traditional features
* Applies PCA for dimensionality reduction
* Saves fitted scalers and PCA models for later use during inference

---

### `cae_model.py`

Defines and trains the convolutional autoencoder.

**Classes / Functions:**

* `CAEmodel`
* `CAETrainer`
* `_ImageDataset`

**Purpose:**
Compresses raw `80x80` images into a smaller `128-dimensional` latent feature vector while keeping important visual information.

---

### `cv_features.py`

Extracts traditional computer vision features from images.

**Classes / Functions:**

* `CVFeatureExtractor`
* `cfd`
* `hum`
* `hsv`
* `lbp`
* `extract_batch`

**Purpose:**
Extracts:

* Shape features
* Color features
* Texture features

Methods include:

* Contour Fourier Descriptors
* Hu Moments
* HSV Histograms
* Local Binary Patterns

---

### `train_gmm.py`

Trains the Gaussian Mixture Model.

**Classes / Functions:**

* `GMMTrainer`

**Purpose:**
Uses PCA-reduced features to create probabilistic clusters.

---

### `train_dec.py`

Trains the Deep Embedded Clustering model.

**Classes / Functions:**

* `ClusteringNN`
* `DECTrainer`

**Purpose:**

* Generates initial pseudo-labels using BIRCH clustering
* Refines clusters using a neural network
* Combines Cross-Entropy loss and KL-Divergence loss during training

---

### `inference.py`

Provides fast predictions during deployment or reinforcement learning.

**Classes / Functions:**

* `SecurityEvaluator`
* `ClusteringNN` (inference mode)

**Purpose:**
Loads saved `.pt` and `.pkl` files once into memory and provides fast predictions without repeatedly reading from disk.

---

### `usl_utils.py`

Contains helper functions used across the USL pipeline.

**Functions:**

* `scan_images()`
* `plot_loss_curve()`
* `map_clusters_to_truth()`
* `generate_comparison_dashboard()`
* `evaluate_and_plot_usl()`

**Purpose:**
Handles:

* Image scanning
* Plotting training loss curves
* Matching clusters to ground truth labels
* Creating evaluation dashboards
* Generating final comparison plots