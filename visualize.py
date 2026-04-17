"""
visualize_tomo_recon.py

Visualize and evaluate tomography reconstructions.

For each sample index XXXXX, expects:
    - original: sample_XXXXX.mat
    - prediction: sample_XXXXX_pred.mat

original.mat should contain:
    - Y_mat        : [H,W] (true unscaled)
prediction.mat should contain:
    - Y_mat_pred   : [H,W] (reconstructed)

This script:
    - displays original Y_mat, reconstructed Y_mat_pred, and their difference
    - computes MSE, PSNR, and SSIM for each sample
    - optionally plots a loss curve if a loss file is provided
"""

import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

# ---------------- USER SETTINGS ----------------

BASE_DIR = r"C:\Users\megan\Box\MeganOndap\MATLAB Code"
MAT_DIR  = os.path.join(BASE_DIR, "MATs")  # folder with sample_*.mat and sample_*.mat_pred

# How many samples to visualize
N_VIS_SAMPLES = 2

# Optional: path to a numpy or text file with loss values per epoch
CHECKPOINT_ROOT = "checkpoints"  # or your actual folder
TRAIN_LOSS_FILE = os.path.join(CHECKPOINT_ROOT, "train_loss_history.npy")
VAL_LOSS_FILE   = os.path.join(CHECKPOINT_ROOT, "val_loss_history.npy")

if os.path.exists(TRAIN_LOSS_FILE):
    train_losses = np.load(TRAIN_LOSS_FILE)
else:
    train_losses = np.array([])

if os.path.exists(VAL_LOSS_FILE):
    val_losses = np.load(VAL_LOSS_FILE)
else:
    val_losses = np.array([])
# ---------------- HELPER FUNCTIONS ----------------

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_psnr(y_true, y_pred, data_range=None):
    """
    PSNR in dB. data_range is max - min of true data.
    """
    mse = compute_mse(y_true, y_pred)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = float(y_true.max() - y_true.min())
        if data_range == 0:
            data_range = 1.0
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

def compute_ssim(y_true, y_pred):
    """
    SSIM using skimage.metrics.ssim.
    Assumes y_true and y_pred are 2D arrays.
    """
    # data_range is required; use range of true image
    dr = float(y_true.max() - y_true.min())
    if dr == 0:
        dr = 1.0
    return ssim(y_true, y_pred, data_range=dr)

# ---------------- LOAD FILE LIST ----------------

orig_files = sorted(glob.glob(os.path.join(MAT_DIR, "sample_*.mat")))
if len(orig_files) == 0:
    raise RuntimeError(f"No sample_*.mat files found in {MAT_DIR}")

# Only keep those that have corresponding _pred.mat
pairs = []
for orig_path in orig_files:
    base = orig_path[:-4]  # strip ".mat"
    pred_path = base + "_pred.mat"
    if os.path.exists(pred_path):
        pairs.append((orig_path, pred_path))

if len(pairs) == 0:
    raise RuntimeError("No sample_XXXXX.mat with corresponding sample_XXXXX_pred.mat found.")

pairs = pairs[:N_VIS_SAMPLES]
print(f"Visualizing {len(pairs)} samples.")

# ---------------- VISUALIZATION LOOP ----------------

mse_list = []
psnr_list = []
ssim_list = []

for orig_path, pred_path in pairs:
    orig_mat = sio.loadmat(orig_path)
    pred_mat = sio.loadmat(pred_path)

    if "Y_mat" not in orig_mat:
        print(f"Skipping {orig_path}: no Y_mat.")
        continue
    if "Y_mat_pred" not in pred_mat:
        print(f"Skipping {pred_path}: no Y_mat_pred.")
        continue

    Y_true = orig_mat["Y_mat"].astype(np.float32)
    Y_pred = pred_mat["Y_mat_pred"].astype(np.float32)

    # Compute metrics
    mse_val = compute_mse(Y_true, Y_pred)
    psnr_val = compute_psnr(Y_true, Y_pred)
    ssim_val = compute_ssim(Y_true, Y_pred)

    mse_list.append(mse_val)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    print(f"{os.path.basename(orig_path)}:")
    print(f"  MSE  = {mse_val:.3e}")
    print(f"  PSNR = {psnr_val:.3f} dB")
    print(f"  SSIM = {ssim_val:.4f}")

    # Plot images
    diff = Y_pred - Y_true

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(Y_true, cmap="gray")
    axes[0].set_title("Original Y_mat")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(Y_pred, cmap="gray")
    axes[1].set_title("Reconstructed Y_mat_pred")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap="bwr")
    axes[2].set_title("Difference (pred - true)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(os.path.basename(orig_path))
    plt.tight_layout()
    plt.show()

# ---------------- SUMMARY METRICS ----------------

if mse_list:
    print("\nSummary over visualized samples:")
    print(f"  Mean MSE  = {np.mean(mse_list):.3e}")
    print(f"  Mean PSNR = {np.mean(psnr_list):.3f} dB")
    print(f"  Mean SSIM = {np.mean(ssim_list):.4f}")

# ---------------- OPTIONAL: LOSS CURVE ----------------

# ---------------- TRAINING & VALIDATION LOSS CURVES ----------------

if train_losses.size == 0:
    print("No training loss history found. Skipping loss plot.")
else:
    if val_losses.size == 0:
        print("No validation loss history found. Plotting training loss only.")
        n_epochs = len(train_losses)
        epochs = np.arange(1, n_epochs + 1)

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_losses, marker="o", label="Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Epoch")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, max(train_losses) * 1.1)
        plt.tight_layout()
        plt.show()
    else:
        n_epochs = min(len(train_losses), len(val_losses))
        epochs = np.arange(1, n_epochs + 1)

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_losses[:n_epochs], marker="o", label="Training loss")
        plt.plot(epochs, val_losses[:n_epochs], marker="s", label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss vs Epoch")
        plt.grid(True)
        plt.legend()
        ymax = max(train_losses.max(), val_losses.max()) * 1.1
        plt.ylim(0, ymax)
        plt.tight_layout()
        plt.show()