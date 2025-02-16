import numpy as np
import cv2  # Using OpenCV for image processing
import os
import platform
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv
load_dotenv()

from sc2.main import run_replay
from sc2.observer_ai import ObserverAI

# Assume train_data_dir is same as in model.py
train_data_dir = "./train_data/"

class ObserverBot(ObserverAI):
    def __init__(self):
        super().__init__()
        self.data = []
        self.frame_count = 0

    async def on_step(self, iteration: int):
        # Capture the game screen/state.
        frame = self.get_current_frame()  # You need to implement this to capture the frame.
        # Process the frame to the correct shape and scale
        processed_frame = cv2.resize(frame, (200, 176))  # Adjust if necessary
        processed_frame = processed_frame.astype(np.float32) / 255.0
        
        # Generate a label based on the replay context (e.g. a one-hot vector)
        # For example purposes, we use a dummy label. In practice, extract action data.
        label = np.array([1, 0, 0, 0])
        
        self.data.append((label, processed_frame))
        self.frame_count += 1
        
        # Optionally, save every N frames.
        if self.frame_count % 100 == 0:
            file_path = os.path.join(train_data_dir, f"replay_data_{iteration}.npy")
            np.save(file_path, self.data, allow_pickle=True)
            self.data = []
            
    def get_current_frame(self):
        # Dummy implementation: capture a screenshot or use game API data
        # Replace with actual capture logic
        return np.zeros((240, 320, 3), dtype=np.uint8)
    
if __name__ == "__main__":
    my_observer_ai = ObserverBot()
    replay_name = "maruVsSerral.SC2Replay"
    home_replay_folder = Path.home() / "Documents" / "StarCraft II" / "Replays"
    replay_path = home_replay_folder / replay_name

    logger.info(f"Checking replay path: {replay_path}")

    if not replay_path.is_file():
        logger.error(f"Replay file does not exist: {replay_path}")
        raise FileNotFoundError(f"Replay file does not exist: {replay_path}")

    logger.info(f"Replay file found: {replay_path}")
    try:
        # Convert the replay path to a POSIX string to avoid issues with spaces.
        replay_path_str = replay_path.as_posix()
        logger.info(f"Using replay path string: {replay_path_str}")
        run_replay(ai=my_observer_ai, replay_path=replay_path_str, realtime=True)
        logger.info("Replay started successfully")
    except Exception as e:
        logger.error(f"Failed to start replay: {e}")