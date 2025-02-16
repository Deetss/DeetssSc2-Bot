import numpy as np
import cv2
import os
import platform
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv
load_dotenv()

from sc2.main import run_replay
from sc2.observer_ai import ObserverAI
from sc2.ids.unit_typeid import UnitTypeId

train_data_dir = "./train_data/"

class ObserverBot(ObserverAI):
    def __init__(self):
        super().__init__()
        self.buffer_size = 3_000
        self.buffer_position = 0
        
        # Pre-allocate arrays with fixed sizes
        self.label_buffer = np.zeros((self.buffer_size, 4), dtype=np.float32)
        self.frame_buffer = np.zeros((self.buffer_size, 176, 200, 3), dtype=np.float32)
        
        self.iteration = 0

    def compute_reward(self):
        reward = 0.0
        
        # Economy reward: reward collecting minerals and vespene above a baseline.
        resource_score = 1 #(self.minerals + self.vespene) / 3000.0
        reward += resource_score

        # Penalize being supply capped
        if self.supply_left <= 0:
            reward -= 1.0

        # Reward for military production and readiness: e.g., if you have more army units relative to supply, reward.
        army_units = len(self.units({UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.HYDRALISK}))
        if self.supply_cap - self.supply_left > 0:
            reward += (army_units / (self.supply_cap - self.supply_left))

        # Reward for successful expansion (could be based on timing, number of townhalls, etc.)
        if self.townhalls and self.townhalls.amount >= 2:
            reward += 0.5

        # Additional domain-specific improvements can be added here once you track outcomes of attacks etc.
        
        return reward

    async def on_step(self, iteration):
        # Capture and process frame
        frame = self.get_current_frame()
        processed_frame = cv2.resize(frame, (200, 176))
        processed_frame = processed_frame.astype(np.float32) / 255.0

        # Compute reward from the current game state in case you need it,
        # but for training a classification model, you need a one-hot action label.
        # Here we assume you have a method that decides an action (0, 1, 2, or 3)
        # For example, you could use compute_reward to guide this decision,
        # but here we simply call a placeholder method "determine_action"
        action = int(np.argmax(self.label_buffer[self.buffer_position]))
        # Create one-hot encoded label for 4 classes:
        label = np.eye(4, dtype=np.float32)[action]

        # Store in buffers
        self.label_buffer[self.buffer_position] = label
        self.frame_buffer[self.buffer_position] = processed_frame
        self.buffer_position += 1

        # Save when buffer is full
        if self.buffer_position >= self.buffer_size:
            save_dict = {
                'labels': self.label_buffer,
                'frames': self.frame_buffer
            }
            file_path = os.path.join(train_data_dir, f"replay_data_{iteration}.npz")
            np.savez_compressed(file_path, **save_dict)
            logger.info(f"Saved {self.buffer_size} frames to {file_path}")
            # Reset the buffer position
            self.buffer_position = 0
            
    def get_current_frame(self):
        # Dummy implementation: capture a screenshot or use game API data
        return np.zeros((240, 320, 3), dtype=np.uint8)

if __name__ == "__main__":
    my_observer_ai = ObserverBot()
    replay_name = "raynor.SC2Replay"
    home_replay_folder = Path.home() / "OneDrive" / "Documents" / "StarCraft II" / "Replays"
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
        run_replay(ai=my_observer_ai, replay_path=replay_path_str, observed_id=2)
        logger.info("Replay started successfully")
    except Exception as e:
        logger.error(f"Failed to start replay: {e}")