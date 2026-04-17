# Tomographic Image Reconstruction

This repository implements a full tomography pipeline that combines:

- **MATLAB** for data generation and classical reconstruction methods.
- **Python (PyTorch + neuraloperator)** for learning a Fourier Neural Operator (FNO) that approximates the tomographic forward operator.

The typical workflow is:

1. Use **MATLAB** to generate and inspect tomographic data (sinograms, reconstructions) and to create training data
2. Use **Python** to train an FNO on these training
3. Use **Python** to reconstruct and visualize results, and to compute image quality metrics.

---

## Repository Overview

A typical structure looks like:

MATLAB
- view_sinogram.m
- backslash.m
- iterative.m
- iterative_nesterov.m
- CreateTrainingData.m   # (use it for data generation)

Python
- train.py
- reconstruct.py
- visualize.py
- tomo_data_processor.py
- neuralop/              # local copy of neuraloperator (if needed)

---

## Libraries Needed
Core numerical and scientific computing
- numpy
- scipy

Deep learning
- torch
- torchvision

Fourier Neural Operator / Neural Operator library
- neuraloperator

Image I/O and processing
- Pillow
- scikit-image

Plotting and visualization
- matplotlib

---

## Instructions

This section explains a quick overview on how to reproduce the full pipeline:

1. Install the required Python libraries ('requirements.txt')
2. Generate training data in MATLAB (`CreateTrainingData.m`)
- Open MATLAB & navigate to the MATLAB folder
- Make sure you have Airtools II
- Open (`CreateTrainingData.m`)
- Set N, rootFiler, maxSamples, & output folders
- **Training data should be found in the box folder (MATLAB Code - MATs, PNGs)
3. Train the Fourier Neural Operator in Python (`train.py`).
4. Reconstruct with the trained model (`reconstruct.py`).
5. Visualize and evaluate results (`visualize.py`).

└─ README.md
