import torch

# Global Hyperparameters & Paths
LR = 5e-5  # try a lower learning rate if higher rate leads to instability
BATCH_SIZE = 64  # Larger batch size for better GPU utilization
NUM_EPISODES = 1000
CHECKPOINT_DIR = "model_checkpoints"
RUNS_DIR = "runs"
LOG_DIR = "logs"
REFRESH_INTERVAL = 5
NUM_WORKERS = 1

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add these constants
MAX_REPLAY_BUFFER_SIZE = 5000  # Reduce buffer size if memory is an issue
CLEANUP_INTERVAL = 100
TENSOR_PRECISION = 'float16' if torch.cuda.is_available() else 'float32'  # Use float16 for half precision if memory is critical

# Add to agent/config.py
GRADIENT_CLIP = 0.5
USE_MIXED_PRECISION = True
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate larger batches

# Add to agent/config.py
TRAINING_INTERVAL = 10  # Train less frequently
USE_JIT = True  # Enable JIT compilation for performance


USE_BEST_CHECKPOINT = True  # Use the best checkpoint for evaluation