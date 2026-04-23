"""
reconstruct_tomo_fno.py

Use a trained FNO model to:
    - predict Z_tilde from X_true
    - undo MATLAB scaling to reconstruct Z_mat (A^T A x_true)
    - save predictions to new *_pred.mat files

Each original.mat file (sample_XXXXX.mat) contains:
    - X_true_mat   : [H,W]
    - Z_mat        : [H,W]          (true unscaled)
    - Z_tilde_mat  : [H,W]          (true scaled)
    - global_min   : scalar
    - global_max   : scalar
"""

import os
import glob
import torch
import numpy as np
import scipy.io as sio

from neuralop.models.fno import FNO
from tomo_data_processor import ImageDataProcessor

# ---------------- USER SETTINGS ----------------

BASE_DIR = r"C:\Users\megan\Box\MeganOndap\MATLAB Code"
MAT_DIR  = os.path.join(BASE_DIR, "MATs")  # folder with sample_*.mat

CHECKPOINT_ROOT = "checkpoints"
MODEL_LOAD_PATH = os.path.join(CHECKPOINT_ROOT, "fno_tomo_phase_mat_ztilde.pt")

# Must match training
FNO_N_MODES         = (64, 64)
FNO_IN_CHANNELS     = 1
FNO_OUT_CHANNELS    = 1
FNO_HIDDEN_CHANNELS = 64
FNO_N_LAYERS        = 6
FNO_DOMAIN_PADDING  = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# How many samples to reconstruct (starting from the first)
N_RECON_SAMPLES = 2   # change this to any number you want

# ---------------- LOAD MODEL ----------------

model = FNO(
    n_modes=FNO_N_MODES,
    in_channels=FNO_IN_CHANNELS,
    out_channels=FNO_OUT_CHANNELS,
    hidden_channels=FNO_HIDDEN_CHANNELS,
    n_layers=FNO_N_LAYERS,
    domain_padding=FNO_DOMAIN_PADDING,
).to(DEVICE)

# Load state_dict (saved with torch.save(model.state_dict(),...))
state_dict = torch.load(MODEL_LOAD_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state_dict)
model.eval()

data_processor = ImageDataProcessor(device=DEVICE)
data_processor.wrap(model)

print(f"Loaded model from {MODEL_LOAD_PATH}")
print("Running inference on.mat files in:", MAT_DIR)

# ---------------- INFERENCE LOOP ----------------

mat_files = sorted(glob.glob(os.path.join(MAT_DIR, "sample_*.mat")))
# Exclude any prediction files like sample_XXXXX_pred.mat
mat_files = [f for f in mat_files if not f.endswith("_pred.mat")]

if len(mat_files) == 0:
    raise RuntimeError(f"No sample_*.mat files found in {MAT_DIR}")

# DEBUG: print all found MAT files
print("Found MAT files in MAT_DIR:")
for f in mat_files:
    print("  ", os.path.basename(f))

# Limit to first N_RECON_SAMPLES
mat_files = mat_files[:N_RECON_SAMPLES]
print(f"Reconstructing {len(mat_files)} samples.")

# DEBUG: print which files we will reconstruct
print(f"\nReconstructing {len(mat_files)} samples:")
for f in mat_files:
    print("  ->", os.path.basename(f))

for mat_path in mat_files:
    mat = sio.loadmat(mat_path)

    # Check required fields
    if "X_true_mat" not in mat:
        print(f"Skipping {mat_path}: no X_true_mat.")
        continue
    if "global_min" not in mat or "global_max" not in mat:
        print(f"Skipping {mat_path}: missing global_min/global_max.")
        continue

    X_true = mat["X_true_mat"].astype(np.float32)   # [H,W]
    global_min = float(mat["global_min"].squeeze())
    global_max = float(mat["global_max"].squeeze())

    # Prepare input tensor [B=1, C=1, H, W]
    x = torch.from_numpy(X_true).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # model(x) returns Z_tilde_pred with shape [1,1,H,W]
        z_tilde_pred = model(x)
        z_tilde_pred = z_tilde_pred.squeeze(0).squeeze(0).cpu().numpy()  # [H,W]

    # Undo MATLAB scaling:
    # Z_tilde = (Z_mat - global_min) / (global_max - global_min)
    # => Z_mat = Z_tilde * (global_max - global_min) + global_min
    Z_mat_pred = z_tilde_pred * (global_max - global_min) + global_min

    # Optionally compare with ground truth if present
    if "Z_mat" in mat:
        Z_mat_true = mat["Z_mat"].astype(np.float32)
        mse = np.mean((Z_mat_pred - Z_mat_true) ** 2)
        print(f"{os.path.basename(mat_path)}: MSE vs true Z_mat = {mse:.3e}")
    else:
        Z_mat_true = None
        print(f"{os.path.basename(mat_path)}: reconstructed Z_mat_pred (no ground truth to compare).")

    # Save predictions to new.mat file
    out_path = mat_path.replace(".mat", "_pred.mat")
    out_dict = {
        "X_true_mat": X_true,
        "Z_tilde_pred": z_tilde_pred,
        "Z_mat_pred": Z_mat_pred,
        "global_min": global_min,
        "global_max": global_max,
    }
    if Z_mat_true is not None:
        out_dict["Z_mat_true"] = Z_mat_true

    sio.savemat(out_path, out_dict)
    print(f"Saved predictions to {out_path}")