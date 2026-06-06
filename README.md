# AutoEncoders from Scratch: Information Bottlenecks and Feature Emergence

## Overview
This repository contains the code and research report exploring the trade-offs between dimensionality reduction and feature preservation in deterministic Autoencoders (AEs). Authored by Naman Goel, this project investigates how varying latent space capacities (4, 32, and 128 dimensions) influence a model's ability to transition from simple pixel-level reconstruction to meaningful semantic representation. 

## Files in this Repository
* **`AutoEncoders_from_Scratch.pdf`**: The comprehensive research paper titled "Information Bottlenecks and Feature Emergence." It covers the theoretical framework (such as the Manifold Hypothesis) and provides an in-depth analysis of the experimental results.
* **`AutoEncoders_from_Scratch.ipynb`**: A Jupyter Notebook containing the complete PyTorch implementation of the experiments. It includes data loading, the `Autoencoder` neural network class, training loops for different bottleneck sizes, and visual evaluations.

## Dataset
The experiments are conducted using the **Fashion-MNIST** dataset. Fashion-MNIST was selected because its classes often share global silhouettes (e.g., Coats, Pullovers, and Shirts), which is notably more complex than standard MNIST and requires the model to prioritize discriminative features over non-discriminative ones.

## Methodology & Key Findings
The project evaluates representation learning by training Autoencoders with three distinct latent bottleneck sizes:
* **Small / Undercomplete (Latent Size 4)**: Forces severe compression.
* **Medium / Balanced (Latent Size 32)**: Acts as an optimal denoiser and successfully maps the data manifold.
* **Large / Overcomplete (Latent Size 128)**: Minimizes Mean Squared Error (MSE) but fails to learn a generalized manifold, demonstrating the dangers of overfitting and memorization.

The "upstream" utility of these learned representations is also evaluated through a "downstream" linear probing task, revealing the geometric separability of the learned latent spaces. 

## Requirements
To run the Jupyter Notebook, you will need a Python 3 environment with the following libraries:
* `torch` (PyTorch)
* `torchvision`
* `matplotlib`
* `numpy`
* `scikit-learn` (for PCA and t-SNE visualizations)

## Execution
Open `AutoEncoders_from_Scratch.ipynb` in Jupyter Notebook or JupyterLab. The notebook is structured to run sequentially: it will download the dataset, visualize samples, define the model architecture, and train the Small, Medium, and Large models before plotting the reconstruction visualizations.
