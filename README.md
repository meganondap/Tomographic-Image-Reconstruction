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
train_tomo_fno.py
reconstruct_tomo_fno.py
visualize_tomo_fno.py
tomo_data_processor.py
neuralop/              # local copy of neuraloperator (if needed)
requirements.txt

## Instruction



└─ README.md
