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

from reward_mixin import RewardMixin

train_data_dir = "./train_data/"

class ObserverBot(RewardMixin, ObserverAI):
    def __init__(self):
        super().__init__()
        self.buffer_size = 1_000
        self.buffer_position = 0

        # Frame + label buffers
        self.label_buffer = np.zeros((self.buffer_size, 4), dtype=np.float32)
        self.frame_buffer = np.zeros((self.buffer_size, 176, 200, 3), dtype=np.float32)
        
        # Event log buffer: each entry holds the events in that frame
        self.events_buffer = [list() for _ in range(self.buffer_size)]
        self.current_events = []

    async def on_step(self, iteration):
        # 1. Get the current frame
        frame = self.get_current_frame()
        processed_frame = cv2.resize(frame, (200, 176)).astype(np.float32) / 255.0
        
        # 2. Decide or retrieve an action index here
        action_index = 3 # default to wait
        
        # Check for economy-related events
        if any(event["type"] == "unit_created" and event["unit_type"] in ["DRONE", "OVERLORD"] for event in self.current_events):
            action_index = 0  # economy

        # Check for army-related events
        elif any(event["type"] == "unit_created" and event["unit_type"] in ["ZERGLING", "ROACH", "HYDRALISK"] for event in self.current_events):
            action_index = 1  # build army

        # Building construction
        elif any(event["type"] == "building_started" for event in self.current_events):
            action_index = 0 # economy

        # You'll need to expand this logic to cover all possible actions
        label = np.eye(4, dtype=np.float32)[action_index]

        # 3. Store the frame, label, and any gathered events
        self.label_buffer[self.buffer_position] = label
        self.frame_buffer[self.buffer_position] = processed_frame
        self.events_buffer[self.buffer_position] = self.current_events[:]

        self.buffer_position += 1
        self.current_events.clear()

        # 4. Save if buffer is full
        if self.buffer_position >= self.buffer_size:
            path = os.path.join(train_data_dir, f"replay_data_{iteration}.npz")
            np.savez_compressed(
                path,
                labels=self.label_buffer,
                frames=self.frame_buffer,
                # Convert events to a NumPy array with dtype=object
                events=np.array(self.events_buffer, dtype=object)
            )
            self.buffer_position = 0

    async def on_unit_created(self, unit):
        # Log the event
        self.current_events.append({"type": "unit_created", "unit_type": unit.type_id.name})

    async def on_building_construction_started(self, structure):
        self.current_events.append({"type": "building_started", "structure": structure.type_id.name})

    async def on_building_construction_complete(self, structure):
        self.current_events.append({"type": "building_complete", "structure": structure.type_id.name})

    async def on_unit_destroyed(self, unit_tag):
        self.current_events.append({"type": "unit_destroyed", "tag": unit_tag})

    async def on_upgrade_complete(self, upgrade):
        self.current_events.append({"type": "upgrade_complete", "upgrade_id": upgrade.name})

    def get_current_frame(self):
        # Replace with your real frame capture
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