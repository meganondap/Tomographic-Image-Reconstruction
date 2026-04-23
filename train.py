"""
train.py

Train FNO on tomography pairs (X_true -> Z_tilde), using MATLAB.mat files.

Each.mat file (sample_XXXXX.mat) contains:
    - X_true_mat   : [H,W] double
    - Z_mat        : [H,W] double
    - Z_tilde_mat  : [H,W] double
    - global_min   : scalar
    - global_max   : scalar

In this script:
    - We load x = X_true_mat
    - We load z = Z_tilde_mat
    - We DO NOT apply any additional global scaling in Python.
    - The model learns to predict Z_tilde directly.
"""

import os
import sys
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split, Dataset

from neuralop.training import AdamW
from neuralop.losses.data_losses import LpLoss, H1Loss
from neuralop.training.trainer import Trainer
from neuralop.models.fno import FNO

from tomo_data_processor import ImageDataProcessor  # your existing DataProcessor

# ============================================================
# 0. TOP-LEVEL CONFIG / HYPERPARAMETERS
# ============================================================

PHASE_ID = "phase_mat_ztilde"

# ---- Path to folder containing sample_*.mat ----
BASE_DIR = r"C:\Users\megan\Box\MeganOndap\MATLAB Code"
MAT_DIR  = os.path.join(BASE_DIR, "MATs")  # folder with sample_*.mat

# create checkpoint folder if it doesnt exist
CHECKPOINT_ROOT = "checkpoints"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

# paths for training log and model saving
TRAIN_LOG_PATH  = os.path.join(CHECKPOINT_ROOT, f"train_log_{PHASE_ID}.txt")
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_ROOT, f"fno_tomo_{PHASE_ID}.pt")

# ---- Data / splitting ----
MAX_SAMPLES    = 1      # cap dataset at N pairs

TRAIN_FRACTION = 1
TRAIN_BATCH_SIZE = 1

VAL_FRACTION   = 0
VAL_BATCH_SIZE = 1

TEST_FRACTION  = 0
TEST_BATCH_SIZE = 1

# ---- Model hyperparameters ----
FNO_N_MODES         = (64, 64)
FNO_IN_CHANNELS     = 1
FNO_OUT_CHANNELS    = 1
FNO_HIDDEN_CHANNELS = 64
FNO_N_LAYERS        = 6
FNO_DOMAIN_PADDING  = None

# ---- Optimization / training hyperparameters ----
N_EPOCHS      = 500
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 0
#WEIGHT_DECAY  = 1e-3

SCHEDULER_TYPE = "StepLR"   # "StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"
STEP_SIZE      = 250
GAMMA          = 0.5
SCHEDULER_PATIENCE = 10
SCHEDULER_T_MAX    = 100

# ---- Loss configuration ----
L2_WEIGHT  = 1.0   # weight for L2 loss
H1_WEIGHT  = 0.01   # weight for H1 loss

# ---- WandB / logging flags ----
WANDB_LOG    = False
WANDB_NAME   = None
WANDB_GROUP  = None
WANDB_PROJ   = "tomo-fno"
WANDB_ENTITY = None
WANDB_SWEEP  = False

# ============================================================
# 1. Logging to file + console
# ============================================================

log_file = open(TRAIN_LOG_PATH, "w")

class Tee(object):
    """Duplicate prints to both console and a log file."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

sys.stdout = Tee(sys.__stdout__, log_file)

print(f"=== Starting {PHASE_ID} ===")
print(f"Training log: {TRAIN_LOG_PATH}")
print(f"Model path:   {MODEL_SAVE_PATH}")
print(f"MAT_DIR:      {MAT_DIR}")

# ============================================================
# 2. Dataset class for sample_*.mat
# ============================================================

class TomographyMatDatasetZtilde(Dataset):
    """
    Dataset for (X_true, Z_tilde) pairs stored in MATLAB.mat files.

    Each original.mat file is expected to have:
        - X_true_mat   : [H,W] double
        - Z_tilde_mat  : [H,W] double
        - Z_mat        : [H,W] double (not used here)
        - global_min   : scalar
        - global_max   : scalar

    This class will:
        - load only files named sample_XXXXX.mat
        - ignore any files ending with '_pred.mat' (e.g. sample_XXXXX_pred.mat)
    """

    def __init__(self, mat_dir):
        super().__init__()
        self.mat_dir = mat_dir

        # Find all sample_*.mat files
        pattern = os.path.join(mat_dir, "sample_*.mat")
        all_files = sorted(glob.glob(pattern))

        # Exclude prediction files like sample_00016_pred.mat
        self.mat_files = [f for f in all_files if not f.endswith("_pred.mat")]

        if len(self.mat_files) == 0:
            raise RuntimeError(f"No original sample_*.mat files found in {mat_dir}")

        print(f"Found {len(self.mat_files)} original sample_*.mat files for training.")

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat_path = self.mat_files[idx]

        import scipy.io as sio
        mat = sio.loadmat(mat_path)

        if "X_true_mat" not in mat:
            raise KeyError(f"'X_true_mat' not found in {mat_path}")
        if "Z_tilde_mat" not in mat:
            raise KeyError(f"'Z_tilde_mat' not found in {mat_path}")

        X_true = mat["X_true_mat"].astype(np.float32)   # [H,W]
        Z_tilde = mat["Z_tilde_mat"].astype(np.float32) # [H,W]

        x = torch.from_numpy(X_true).unsqueeze(0)   # [1,H,W]
        z = torch.from_numpy(Z_tilde).unsqueeze(0)  # [1,H,W]

        return {"x": x, "y": z}

# ============================================================
# 3. Device setup
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
is_logger = True

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Using device:", device)

# ============================================================
# 4. Build dataset (no scaling)
# ============================================================

base_dataset = TomographyMatDatasetZtilde(mat_dir=MAT_DIR)

# Limit to MAX_SAMPLES (for overfitting or small experiments)
max_samples = min(MAX_SAMPLES, len(base_dataset))
base_dataset = Subset(base_dataset, list(range(max_samples)))

n_total = len(base_dataset)
print(f"Total.mat samples available (after MAX_SAMPLES cap): {n_total}")

if n_total == 0:
    raise RuntimeError("No samples available after applying MAX_SAMPLES.")

# ============================================================
# 5. Train/val/test split and DataLoaders (robust)
# ============================================================

def split_dataset(dataset, train_frac, val_frac, test_frac, generator=None):
    """
    Robustly split a dataset into train/val/test subsets.

    - If the dataset is tiny (n <= 2), use all samples for all splits.
    - Otherwise, use the provided fractions, but always ensure:
        * each split has at least 0 samples
        * total sizes sum to n
    """
    n = len(dataset)

    # Tiny dataset: use everything for all splits
    if n <= 2:
        print(f"Tiny dataset (n_total = {n}). Using all samples for train/val/test.")
        return dataset, dataset, dataset, n, n, n

    # Normal case: use fractions
    # Clamp fractions to [0,1] and renormalize if needed
    fracs = np.array([train_frac, val_frac, test_frac], dtype=float)
    if fracs.sum() == 0:
        # default to all training if everything is zero
        fracs = np.array([1.0, 0.0, 0.0])
    else:
        fracs = fracs / fracs.sum()

    # Compute split sizes
    n_train = int(round(fracs[0] * n))
    n_val   = int(round(fracs[1] * n))
    n_test  = n - n_train - n_val

    # Fix any negative or zero-length issues
    if n_train < 1:
        n_train = 1
    if n_val < 0:
        n_val = 0
    if n_test < 0:
        n_test = 0

    # Re-adjust if we overshoot
    while n_train + n_val + n_test > n:
        # reduce the largest split by 1
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_train and n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break

    # If we undershoot, add leftover to train
    while n_train + n_val + n_test < n:
        n_train += 1

    # Now actually split
    splits = [n_train, n_val, n_test]
    subsets = random_split(dataset, splits, generator=generator)

    train_ds, val_ds, test_ds = subsets
    return train_ds, val_ds, test_ds, n_train, n_val, n_test


# Use the robust splitter
train_dataset, val_dataset, test_dataset, n_train, n_val, n_test = split_dataset(
    base_dataset,
    TRAIN_FRACTION,
    VAL_FRACTION,
    TEST_FRACTION,
    generator=torch.Generator().manual_seed(0),
)

print(f"Total samples used: {n_total} (train={n_train}, val={n_val}, test={n_test})")

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=VAL_BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=TEST_BATCH_SIZE, shuffle=False)
print(f"Total samples used: {n_total} (train={n_train}, val={n_val}, test={n_test})")

# ============================================================
# 6. Model, optimizer, scheduler
# ============================================================

model = FNO(
    n_modes=FNO_N_MODES,
    in_channels=FNO_IN_CHANNELS,
    out_channels=FNO_OUT_CHANNELS,
    hidden_channels=FNO_HIDDEN_CHANNELS,
    n_layers=FNO_N_LAYERS,
    domain_padding=FNO_DOMAIN_PADDING,
).to(device)

print(model)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

if SCHEDULER_TYPE == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=GAMMA,
        patience=SCHEDULER_PATIENCE,
        mode="min",
    )
elif SCHEDULER_TYPE == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX
    )
elif SCHEDULER_TYPE == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )
else:
    scheduler = None

# ============================================================
# 7. Losses and DataProcessor
# ============================================================

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

def combined_train_loss(z_pred, y, **kwargs):
    """
    Weighted sum of L2 and H1 losses.

    z_pred : model output (Z_tilde_pred)
    y      : ground truth (Z_tilde_mat)
    """
    l2 = l2loss(z_pred, y)
    h1 = h1loss(z_pred, y)
    total = L2_WEIGHT * l2 + H1_WEIGHT * h1
    return total

eval_losses = {
    "l2": l2loss,
    "h1": h1loss,
}

data_processor = ImageDataProcessor(device=device)
data_processor.wrap(model)

# ============================================================
# 8. Trainer and training loop
# ============================================================

trainer = Trainer(
    model=model,
    n_epochs=N_EPOCHS,
    data_processor=data_processor,
    device=device,
    wandb_log=WANDB_LOG,
    verbose=is_logger,
)

test_loaders = {
    "val":  val_loader,
    "test": test_loader,
}

train_loss_history = []
val_loss_history   = []

# ---- Wrap train_one_epoch to log training loss ----
orig_train_one_epoch = trainer.train_one_epoch

def train_one_epoch_with_logging(*args, **kwargs):
    """
    Wrap Trainer.train_one_epoch to log the training loss per epoch.
    Accepts arbitrary args/kwargs to match the original signature.
    """
    result = orig_train_one_epoch(*args, **kwargs)

    # result is usually (train_loss, train_metrics,...) – take the first element
    if isinstance(result, tuple) and len(result) > 0:
        epoch_loss = result[0]
    else:
        epoch_loss = result

    try:
        train_loss_history.append(float(epoch_loss))
    except Exception:
        pass

    return result

trainer.train_one_epoch = train_one_epoch_with_logging

# ---- Wrap evaluate_all to log validation loss ----
orig_evaluate_all = trainer.evaluate_all

def evaluate_all_with_logging(*args, **kwargs):
    """
    Wrap Trainer.evaluate_all to log a single scalar validation loss per epoch.
    We will log the 'val' loader's 'l2' metric if available.
    """
    eval_metrics = orig_evaluate_all(*args, **kwargs)

    # eval_metrics is usually a dict like:
    #   {'val': {'l2': value, 'h1': value,...},
    #    'test': {...}}
    if isinstance(eval_metrics, dict) and "val" in eval_metrics:
        val_metrics = eval_metrics["val"]
        if isinstance(val_metrics, dict) and len(val_metrics) > 0:
            if "l2" in val_metrics:
                val_loss = val_metrics["l2"]
            else:
                val_loss = next(iter(val_metrics.values()))
            try:
                val_loss_history.append(float(val_loss))
            except Exception:
                pass

    return eval_metrics

trainer.evaluate_all = evaluate_all_with_logging

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=combined_train_loss,
    eval_losses=eval_losses,
    regularizer=None,
)

# ============================================================
# 9. Save trained model
# ============================================================

# Save trained model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saved trained model to {MODEL_SAVE_PATH}")

# Save loss histories
train_loss_history = np.array(train_loss_history, dtype=float)
val_loss_history   = np.array(val_loss_history, dtype=float)

train_loss_file = os.path.join(CHECKPOINT_ROOT, "train_loss_history.npy")
val_loss_file   = os.path.join(CHECKPOINT_ROOT, "val_loss_history.npy")

np.save(train_loss_file, train_loss_history)
np.save(val_loss_file,   val_loss_history)

print(f"Saved training loss history to {train_loss_file}")
print(f"Saved validation loss history to {val_loss_file}")