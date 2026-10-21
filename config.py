import os
import torch

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "datasets"
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
CIF_DIR = os.path.join(DATA_DIR, "cif")

FIG_DIR = "figures"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
LR = 5e-4
EPOCHS = 500
TEST_SIZE = 0.2

# Data Augmentation Settings
AUG_METHOD = "both"  # Options: None, "rattle", "symmetry", "both"
AUG_FACTOR = 2       # Number of augmented samples per original training sample