import torch

# Global Hyperparameters & Paths
LR = 3e-5  # try a lower learning rate if higher rate leads to instability
BATCH_SIZE = 32  # increase the batch size for more robust updates
NUM_EPISODES = 1000
CHECKPOINT_DIR = "model_checkpoints"
RUNS_DIR = "runs"
LOG_DIR = "logs"
REFRESH_INTERVAL = 5
NUM_WORKERS = 6

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"