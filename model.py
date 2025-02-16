import os
import random
import numpy as np
import psutil
from datetime import datetime
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs found:", gpus)
    # Optionally, enable memory growth for each GPU.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found!")

# Define missing constants and variables
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Directory containing training files (adjust this path as needed)
train_data_dir = "./train_data/"

# Get list of all files and split into training and validation lists (90/10 split)
all_files = os.listdir(train_data_dir)
random.shuffle(all_files)
split_idx = int(0.9 * len(all_files))
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

def print_memory_usage():
    """Monitor memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def create_model():
    """Create model using Functional API with proper input layer"""
    inputs = Input(shape=(176, 200, 3))
    
    # First conv block
    x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)
    
    # Second conv block
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)
    
    # Third conv block
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)
    
    # Dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def data_generator(files, batch_size=32):
    """Generator for memory-efficient data loading"""
    while True:
        random.shuffle(files)
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_data = list(process_batch(batch_files, 0, len(batch_files)))
            if batch_data:
                x = np.concatenate([x for x, _ in batch_data], axis=0)
                y = np.concatenate([y for _, y in batch_data], axis=0)
                yield x, y

def process_batch(files, start, end):
    """Process data in smaller batches with memory tracking"""
    batch_data = []
    total_samples = 0
    
    print(f"Processing files {start} to {end}")
    for file in files[start:end]:
        try:
            data = np.load(os.path.join(train_data_dir, file))

            # Get labels and frames from the dictionary
            labels = data['labels']
            frames = data['frames']

            # Process each frame and label pair
            for label, frame in zip(labels, frames):
                if frame.shape == (176, 200, 3):
                    # Frame is already normalized in observer.py
                    sample = (label, frame)
                    batch_data.append(sample)
                    total_samples += 1
                    
                if total_samples >= BATCH_SIZE:
                    x = np.array([i[1] for i in batch_data])
                    y = np.array([i[0] for i in batch_data])
                    yield x, y
                    batch_data = []
                    total_samples = 0
                    gc.collect()
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if batch_data:
        x = np.array([i[1] for i in batch_data])
        y = np.array([i[0] for i in batch_data])
        yield x, y

# Build model and compile
model = create_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def train_model_in_batches():
    """Train model in smaller batches with early stopping."""
    print("Starting training...")
    
    best_val_loss = float("inf")
    patience = 2  # Number of epochs to wait without improvement
    epochs_no_improvement = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Process training and validation data
        train_generator = process_batch(train_files, 0, len(train_files))
        val_generator = process_batch(val_files, 0, len(val_files))
        
        train_steps = 0
        train_loss = 0
        train_acc = 0
        
        # Train on batches
        for x_batch, y_batch in train_generator:
            history = model.train_on_batch(x_batch, y_batch)
            train_loss += history[0]
            train_acc += history[1]
            train_steps += 1
            
            if train_steps % 10 == 0:
                print(f"Step {train_steps}: loss = {train_loss/train_steps:.4f}, acc = {train_acc/train_steps:.4f}")
                
            del x_batch, y_batch
            gc.collect()
            
        # Validate
        val_steps = 0
        val_loss = 0
        val_acc = 0
        
        for x_val, y_val in val_generator:
            val_history = model.test_on_batch(x_val, y_val)
            val_loss += val_history[0]
            val_acc += val_history[1]
            val_steps += 1
            del x_val, y_val
            gc.collect()
            
        # Compute average losses & accuracies
        avg_train_loss = train_loss / train_steps if train_steps else 0
        avg_train_acc = train_acc / train_steps if train_steps else 0
        avg_val_loss = val_loss / val_steps if val_steps else 0
        avg_val_acc = val_acc / val_steps if val_steps else 0
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train Accuracy: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {avg_val_acc:.4f}")
        
        # Save checkpoint every epoch
        model.save(f"model_checkpoints/checkpoint-epoch-{epoch+1}.keras")
        
        # Early stopping check
        if avg_val_loss < (best_val_loss + 0.005):
            best_val_loss = avg_val_loss
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= patience:
                print(f"Early stopping triggered: no improvement for {patience} epochs.")
                break
            
try:
    train_model_in_batches()
    model.save(f"BasicCNN-Final-{EPOCHS}-epochs-{LEARNING_RATE}-LR.keras")
    print("Training completed successfully")
except Exception as e:
    print(f"Training failed: {e}")
finally:
    print_memory_usage()
    gc.collect()
    tf.keras.backend.clear_session()