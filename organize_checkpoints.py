import os
import re
import shutil

CHECKPOINT_DIR = "model_checkpoints"

# Regex to match checkpoint filenames.
# This expects filenames like: dueling-SLICKNAME-episode-XXX.pth
pattern = re.compile(r"dueling-([^-]+(?:_[^-]+)*)-episode-(\d+)\.pth")

for fname in os.listdir(CHECKPOINT_DIR):
    match = pattern.match(fname)
    if match:
        model_name = match.group(1)  # e.g. "Slick_Cheetah_2"
        target_folder = os.path.join(CHECKPOINT_DIR, model_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        old_path = os.path.join(CHECKPOINT_DIR, fname)
        new_path = os.path.join(target_folder, fname)
        shutil.move(old_path, new_path)
        print(f"Moved {fname} to {target_folder}")