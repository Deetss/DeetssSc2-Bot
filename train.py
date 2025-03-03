from pysc2.env import sc2_env
from pysc2.lib import features, actions
import random
import os
import multiprocessing
import signal
from absl import flags, app
from agent.zerg_agent import ZergAgent
import sys
import numpy as np
import gc
import psutil
import torch
import shutil

from agent.config import NUM_WORKERS, NUM_EPISODES
from memoryUtils import monitor_memory

_original_unpack_rgb_image = features.Feature.unpack_rgb_image

def patched_unpack_rgb_image(plane):
    if plane.bits_per_pixel != 24:
        # Get dimensions from the plane's shape if available
        try:
            height, width = plane.size
        except (AttributeError, ValueError):
            # Default dimensions if we can't get them from plane
            height, width = 192, 256  # Match your interface dimensions
        
        # Return a dummy RGB image
        return np.zeros((height, width, 3), dtype=np.uint8)  # Changed from 4 to 3 channels
    return _original_unpack_rgb_image(plane)

features.Feature.unpack_rgb_image = patched_unpack_rgb_image

def train_agent(worker_id):
    flags.FLAGS(sys.argv)
    cool_adjectives = ["Swift", "Mighty", "Slick"]
    animals = ["Cheetah", "Panther", "Eagle"]
    agent_name = f"{random.choice(cool_adjectives)}_{random.choice(animals)}_{worker_id}"
    agent = ZergAgent(worker_id, agent_name=agent_name)
    
    cleanup_interval = 100  # Adjust based on your needs
    
    with sc2_env.SC2Env(
            map_name="AbyssalReef",
            players=[
                sc2_env.Agent(sc2_env.Race.zerg),
                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.harder)
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.FEATURES,
                feature_dimensions=features.Dimensions(screen=(256, 192), minimap=(128, 128)),
                rgb_dimensions=features.Dimensions(screen=(256, 192), minimap=(128, 128)),
                use_feature_units=True,
                use_camera_position=True,
                use_unit_counts=True,
                use_raw_units=True
            ),
            step_mul=16,
            game_steps_per_episode=0,
            visualize=True) as env:
        agent.setup(env.observation_spec(), env.action_spec())
        for episode in range(NUM_EPISODES):
            timesteps = env.reset()
            agent.reset()
            while True:
                step_actions = [agent.step(timesteps[0])]
                if timesteps[0].last():
                    agent.wins += int(timesteps[0].reward > 0)
                    agent.losses += int(timesteps[0].reward <= 0)
                    break
                timesteps = env.step(step_actions)
                
                # Periodic cleanup
                if episode % cleanup_interval == 0:
                    agent.periodic_cleanup()
            
            if (episode + 1) % agent.REFRESH_INTERVAL == 0:
                agent.refresh_model()
                agent.periodic_cleanup()  # Extra cleanup after model refresh

            if episode % 10 == 0:
                memory_usage = monitor_memory()
                if memory_usage > 3500:  # 4GB threshold
                    gc.collect()
                    torch.cuda.empty_cache()

            print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            agent.writer.add_scalar("Memory/Usage", memory_usage, episode)

def main(unused_argv):
    # Clear all previous runs at the start of training
    runs_dir = "runs"
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
    os.makedirs(runs_dir)
    
    processes = []
    
    def handle_sigint(signum, frame):
        print("SIGINT received, terminating workers...")
        for p in processes:
            p.terminate()
        # Force garbage collection after terminating processes.
        gc.collect()
        exit(0)
    
    signal.signal(signal.SIGINT, handle_sigint)
    for worker_id in range(NUM_WORKERS):
        print(f"Starting worker {worker_id}")
        p = multiprocessing.Process(target=train_agent, args=(worker_id,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join(28000)
        if p.is_alive():
            print("A process is taking too long; terminating it.")
            p.terminate()
    # Ensure all processes are joined and initiate extra garbage collection.
    for p in processes:
        p.join()
    gc.collect()

if __name__ == "__main__":
    app.run(main)