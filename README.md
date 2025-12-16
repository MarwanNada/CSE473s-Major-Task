# **Course:** CSE473s: Computational Intelligence (Fall 2025)
# Project Overview

This project implements a modular Deep Learning library from scratch using only Python and NumPy. The goal is to demystify the internal mechanics of neural networks by implementing forward propagation, backpropagation (Gradient Descent), and various activation/loss functions manually.

Beyond the core library, the project demonstrates advanced applications including unsupervised learning (Autoencoders) and transfer learning (Latent Space Classification), benchmarking the custom implementation against TensorFlow/Keras.

# Features

**Modular Design:** Layers, Activations, Losses, and Optimizers are decoupled classes.

**Layers:** Fully Connected (Dense) layers with configurable input/output sizes.

**Activations:** Sigmoid, Tanh, ReLU, Softmax.

**Losses:** Mean Squared Error (MSE).

**Optimization:** Stochastic Gradient Descent (SGD) with support for Mini-Batch training.

# Repository Structure

> .
> 
> ├── lib/                    # Source code for the library
> 
> │   ├── __init__.py
> 
> │   ├── layers.py           # Dense layer implementation (Forward/Backward)
> 
> │   ├── activations.py      # Sigmoid, Tanh, ReLU, etc.
> 
> │   ├── losses.py           # MSE loss logic
> 
> │   ├── optimizer.py        # SGD optimizer (Weight updates)
> 
> │   └── network.py          # Network container class
> 
> ├── notebooks/
> 
> │   └── project_demo.ipynb  # Main Demo: XOR, Autoencoder, SVM, & TF Comparison
> 
> ├── report/
> 
> │   └── project_report.pdf  # Final PDF Report
> 
> ├── .gitignore
> 
> └── README.md


# Getting Started

# Prerequisites

1. **Python 3.x**

2. **NumPy**

3. **Matplotlib (for visualization)**

4. **TensorFlow/Keras (for benchmarking comparison only)**

5. **Scikit-Learn (for SVM classification)**

# Installation

**Clone the repository:**

`git clone [https://github.com/MarwanNada/CSE473s-Major-Task](https://github.com/MarwanNada/CSE473s-Major-Task.git)`

# Usage

**To run the complete project demo:**

1. Navigate to the notebooks folder.

2. Open project_demo.ipynb.

3. Run all cells sequentially.

# Results Summary

1. **XOR Problem (Validation)**

**Objective:** Solve a non-linearly separable problem.

**Result:** Successfully converged to near-zero loss (< 0.01) using a 2-4-1 architecture.

**Performance:** The custom library was significantly faster (~1s) than Keras (~300s) for this small-scale task due to lower overhead.

2. **Autoencoder (Unsupervised Learning)**

**Objective:** Compress MNIST images (784 pixels) into a 32-dimensional latent space and reconstruct them.

**Result:** Reconstructed images successfully captured digit structures.

**Technique:** Utilized Mini-Batch SGD and Data Shuffling to ensure convergence.

3. **Latent Space Classification (Transfer Learning)**

**Objective:** Use the trained Encoder as a feature extractor for an SVM classifier.

**Result:** Achieved >90% accuracy on the test set using only the 32 compressed features.

**Conclusion:** Proves the custom library learned semantically meaningful features.

4. **Gradient Checking**

Validated numerical vs. analytical gradients with an error margin < 1e-7, confirming the correctness of the backpropagation calculus.