from pysc2.agents import base_agent
from pysc2.lib import actions, features
import numpy as np
from absl import app
import random
import time
import os
import gc
from torch.utils.tensorboard import SummaryWriter

import re
from agent.loss import CustomLoss
from agent.a2c import a2c_train_step, create_a2c_model
from agent.config import DEVICE as device
from agent.reward_mixin import RewardMixin

import torch
import torch.optim as optim
import torch.nn.functional as F

from agent.config import REFRESH_INTERVAL, LR, MAX_REPLAY_BUFFER_SIZE

import cProfile
import pstats
from torch.cuda.amp import GradScaler, autocast
from collections import deque
import shutil

USE_PRETRAINED = True  # Load a pretrained model if available.
PRETRAINED_FILENAME = "dueling-Swift_Eagle_3-episode-535.pth"
#RETRAINED_FILENAME = "observer_dqn_checkpoint.pth"

class ZergAgent(RewardMixin, base_agent.BaseAgent):
    def __init__(self, worker_id, agent_name=None, use_pretrained=USE_PRETRAINED, pretrained_filename=PRETRAINED_FILENAME):
        super(ZergAgent, self).__init__()
        self.use_pretrained = use_pretrained
        self.worker_id = worker_id
        self.REFRESH_INTERVAL = REFRESH_INTERVAL
        # Generate a random name if none provided
        self.agent_name = agent_name if agent_name else f"agent_{worker_id}"
        # Create a dedicated checkpoint folder for this agent.
        self.checkpoint_dir = os.path.join("model_checkpoints", self.agent_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint = None
        self.epsilon = 1.0  # Fully random exploration initially.
        self.num_actions = len(actions.FUNCTIONS)
        #self.model = create_dueling_model(num_actions=len(actions.FUNCTIONS), structured_size=31)
        self.model = create_a2c_model(num_actions=len(actions.FUNCTIONS), structured_size=31,action_coord_sizes=None)
        self.episode_count = 0  # New counter for episodes
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.batch_size = 16
        self.replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_SIZE)
        
        # Variables for logging / performance.
        self.total_reward = 0
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.action_count = 0
        self.start_time = time.time()
        # Initialize TensorBoard SummaryWriter.
        self.writer = SummaryWriter(log_dir=os.path.join("runs", self.agent_name))
        
        # Modified checkpoint loading logic
        if self.use_pretrained:
            if pretrained_filename and os.path.exists(pretrained_filename):
                print(f"Loading specified checkpoint from {pretrained_filename}")
                self._load_checkpoint(pretrained_filename)
            else:
                # Look for latest checkpoint with this agent's name
                latest_checkpoint = self._find_latest_checkpoint()
                if latest_checkpoint:
                    print(f"Loading latest checkpoint: {latest_checkpoint}")
                    self._load_checkpoint(latest_checkpoint)
                    # Extract episode number from filename
                    match = re.search(r'episode-(\d+)', latest_checkpoint)
                    if match:
                        self.episode_count = int(match.group(1))
                    self.epsilon = 0.1  # Reduced exploration for pretrained model
                else:
                    print("No checkpoints found. Starting with a random model.")
        else:
            print("use_pretrained=False. Starting with a random model.")

        self.prev_worker_count = 0
        self.prev_army_count = 0
        self.prev_base_count = 0  # We'll still need to track this separately
        self.episode_step_count = 0  # Initialize step count

    def extract_full_structured_obs(self, obs):
        """Extracts a wider set of features from the observation dictionary."""
        keys_to_use = [
            "player",
            "control_groups",
            "single_select",
            "multi_select",
            "cargo",
            "build_queue",
            "production_queue",
            "last_actions",
            "cargo_slots_available",
            "home_race_requested",
            "away_race_requested",
        ]
        obs_list = []
        for key in keys_to_use:
            value = obs.observation.get(key, None)
            if value is None:
                continue
            try:
                flat = np.array(value).flatten()
                obs_list.append(flat)
            except Exception as e:
                print(f"Skipping {key}: {e}")
        if not obs_list:
            return np.zeros((31,), dtype=np.float32)  # or another default vector
        combined = np.concatenate(obs_list).astype(np.float32)
        
        # Adjust to the fixed size your model expects.
        desired_size = 31  # update this to match your model's input dimensions
        if combined.size < desired_size:
            padded = np.zeros((desired_size,), dtype=np.float32)
            padded[:combined.size] = combined
            combined = padded
        elif combined.size > desired_size:
            combined = combined[:desired_size]
        
        return combined

    def refresh_model(self, checkpoint_dir="model_checkpoints"):
        """Reloads the most recent checkpoint from the given directory."""
        if not os.path.exists(checkpoint_dir):
            return
        checkpoints = [os.path.join(checkpoint_dir, f)
                       for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if not checkpoints:
            return
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        try:
            # Removed weights_only parameter from torch.load.
            state_dict = torch.load(latest_checkpoint, map_location=device, encoding='latin1')
        except Exception as e:
            print(f"Failed to load checkpoint {latest_checkpoint} with encoding 'latin1', falling back:", e)
            try:
                state_dict = torch.load(latest_checkpoint, map_location=device)
            except Exception as e2:
                print("Loading fallback failed. Skipping refresh.")
                return

        # Filter out mismatched keys and update the current model's state.
        current_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in current_state_dict:
                if current_state_dict[key].size() == value.size():
                    filtered_state_dict[key] = value
                else:
                    print(f"Skipping {key} due to size mismatch: checkpoint {value.size()} vs current model {current_state_dict[key].size()}.")
            else:
                print(f"Skipping {key} as it is not present in the current model.")
        current_state_dict.update(filtered_state_dict)
        self.model.load_state_dict(current_state_dict)
        print(f"Refreshed model from {latest_checkpoint}")

    def preprocess_state(self, obs):
        import numpy as np  # ensure numpy is imported
        # Process screen image.
        if "rgb_screen" in obs.observation:
            screen = obs.observation["rgb_screen"]
            if screen.ndim == 3:
                # If channels are last, move to channels-first.
                if screen.shape[2] in (1, 3):
                    screen = np.transpose(screen, (2, 0, 1))
            elif screen.ndim == 2:
                screen = np.expand_dims(screen, 0)
            # If grayscale, replicate channels.
            if screen.shape[0] == 1:
                screen = np.repeat(screen, 3, axis=0)
        else:
            screen = obs.observation["feature_screen"][features.SCREEN_FEATURES.player_id.index]
            if len(screen.shape) == 1:
                screen = np.tile(screen, (64, 1))
            elif len(screen.shape) != 2:
                try:
                    screen = screen.reshape((64, 64))
                except Exception:
                    screen = np.tile(screen, (64, 1))
            # Convert to 3 channels.
            screen = np.repeat(screen[np.newaxis, ...], 3, axis=0)

        # Process minimap image.
        if "rgb_minimap" in obs.observation:
            minimap = obs.observation["rgb_minimap"]
            if minimap.ndim == 3:
                if minimap.shape[2] in (1, 3):
                    minimap = np.transpose(minimap, (2, 0, 1))
            elif minimap.ndim == 2:
                minimap = np.expand_dims(minimap, 0)
            if minimap.shape[0] == 1:
                minimap = np.repeat(minimap, 3, axis=0)
        else:
            minimap = obs.observation["feature_minimap"][features.MINIMAP_FEATURES.player_id.index]
            minimap = np.repeat(minimap[np.newaxis, ...], 3, axis=0)

        screen_tensor = torch.tensor(screen, dtype=torch.float32).unsqueeze(0).to(device)
        minimap_tensor = torch.tensor(minimap, dtype=torch.float32).unsqueeze(0).to(device)
        return screen_tensor, minimap_tensor
    
    def _to_rgb(self, img_tensor):
        # Remove batch dimension if present.
        if img_tensor.dim() == 3:
            # Already CHW, assume it's what the model sees.
            pass
        elif img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)
            
        # If the tensor is 2D (H, W), add a channel dimension.
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)
            
        # If it has 1 channel, replicate it to 3 channels.
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        return img_tensor

    def step(self, obs):
        """Simplified step function focused on A2C training."""
        super(ZergAgent, self).step(obs)
        
        try:
            with torch.no_grad():
                # Get structured observations only
                obs_state = self.extract_full_structured_obs(obs)
                obs_state_tensor = torch.tensor(obs_state, dtype=torch.float32, device=device)
                
                # Forward pass
                screen_tensor, minimap_tensor = self.preprocess_state(obs)
                action_logits, _, arg_out = self.model(
                    screen_tensor,
                    minimap_tensor,
                    obs_state_tensor.unsqueeze(0)
                )
                
                # Action selection
                action_probs = F.softmax(action_logits, dim=1)[0].cpu().numpy()
                chosen_action_id = np.random.choice(len(action_probs), p=action_probs)
                
                if chosen_action_id not in obs.observation["available_actions"]:
                    chosen_action_id = np.random.choice(obs.observation["available_actions"])
                
                # Get counts from player information
                player_info = obs.observation.player
                current_worker_count = player_info[7]  # idle worker count
                current_army_count = player_info[8]    # army count
                current_base_count = player_info[6]     # base count
                
                # Create a current state representation
                current_state = {
                    'worker_count': current_worker_count,
                    'army_count': current_army_count,
                    'base_count': current_base_count,
                }
                
                # Process action arguments and get reward
                args, reward = self._process_action_args(chosen_action_id, arg_out, obs)
                reward = self.compute_reward(obs, current_state, chosen_action_id)  # Call compute_reward
                
                # Log the reward for debugging
                #print(f"Chosen action: {chosen_action_id}, Reward: {reward}")
                
                # Ensure reward is a valid number
                if reward is None:
                    reward = 0  # Default to 0 if reward is None
                
                # Update buffer with simplified state representation
                self.replay_buffer.append((
                    screen_tensor,
                    minimap_tensor,
                    obs_state_tensor.unsqueeze(0),
                    chosen_action_id,
                    reward
                ))
                
                self.total_reward += reward
                
                # Log metrics to TensorBoard
                self.writer.add_scalar("Action/Chosen", chosen_action_id, self.episode_count)
                self.writer.add_scalar("Action/Count", self.action_count, self.episode_count)
                
                return actions.FunctionCall(chosen_action_id, args)
                
        except Exception as e:
            print(f"Error in step: {e}")
            return actions.FunctionCall(0, [])

    def _perform_training(self):
        """Separated training logic with optional profiling"""
        # Initialize GradScaler
        scaler = GradScaler()
        
        # Only profile every 100 training steps
        should_profile = (self.training_steps % 100 == 0)
        profiler = cProfile.Profile() if should_profile else None
        
        try:
            if should_profile:
                profiler.enable()
            
            batch_size = min(2000, len(self.replay_buffer))
            training_data = list(self.replay_buffer)[-batch_size:]
            
            # Use GradScaler for mixed precision training
            with torch.amp.autocast('cuda'):
                loss, policy_loss, value_loss, entropy = a2c_train_step(
                    self.model,
                    self.optimizer,
                    rollout=training_data,
                    gamma=0.99,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    writer=self.writer
                )
            
            self._log_training_metrics(loss, policy_loss, value_loss, entropy)
            
            while len(self.replay_buffer) > MAX_REPLAY_BUFFER_SIZE // 2:
                self.replay_buffer.popleft()
            
            self.training_steps += 1
            
            self.writer.add_scalar("Loss/Total", float(loss), self.training_steps)
            self.writer.add_scalar("Loss/Policy", float(policy_loss), self.training_steps)
            self.writer.add_scalar("Loss/Value", float(value_loss), self.training_steps)
            self.writer.add_scalar("Loss/Entropy", float(entropy), self.training_steps)
            
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            if should_profile:
                profiler.disable()
                stats = pstats.Stats(profiler).sort_stats('cumulative')
                stats.print_stats()
            
            torch.cuda.empty_cache()

    def add_to_replay_buffer(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > MAX_REPLAY_BUFFER_SIZE:
            self.replay_buffer.pop(0)  # Remove oldest transition

    def reset(self):
        """Simplified reset function."""
        super(ZergAgent, self).reset()
        self.episode_rewards.append(self.total_reward)
        self.writer.add_scalar("Reward/Total", float(self.total_reward), self.episode_count)
        self.writer.add_scalar("Episode/Count", float(self.episode_count), self.episode_count)
        print(f"Episode {self.episode_count} reset. Total reward: {self.total_reward:.2f}")

        # Perform end-of-episode training
        if len(self.replay_buffer) > 32:
            try:
                training_data = list(self.replay_buffer)[-2000:]
                loss, policy_loss, value_loss, entropy = a2c_train_step(
                    self.model,
                    self.optimizer,
                    rollout=training_data,
                    gamma=0.99,
                    ent_coef=0.01,
                    vf_coef=0.5
                )
                
                # Log metrics - ensure we're dealing with scalar values
                if loss != 0.0:  # Only log if we actually performed training
                    self.writer.add_scalar("Loss/Total", float(loss), self.episode_count)
                    self.writer.add_scalar("Loss/Policy", float(policy_loss), self.episode_count)
                    self.writer.add_scalar("Loss/Value", float(value_loss), self.episode_count)
                    self.writer.add_scalar("Loss/Entropy", float(entropy), self.episode_count)
                
            except Exception as e:
                print(f"Training error in reset: {e}")
                # Don't clear the replay buffer on error, just continue
                pass

        # Save checkpoint periodically
        if self.episode_count % REFRESH_INTERVAL == 0:
            self._save_checkpoint()

        self.total_reward = 0
        self.episode_count += 1
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def log_results(self):
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        elapsed_time = time.time() - self.start_time if self.start_time else 1
        apm = (self.action_count / elapsed_time) * 60
        self.writer.add_scalar("Avg Reward", avg_reward, self.episode_count)
        self.writer.add_scalar("APM", apm, self.episode_count)
        self.writer.add_scalar("Wins", self.wins, self.episode_count)
        self.writer.add_scalar("Losses", self.losses, self.episode_count)
        self.writer.close()
        print(f"Agent {self.agent_name} completed {self.episode_count} episodes.")
        print(f"Average reward: {avg_reward:.2f}, APM: {apm:.2f}")
        print(f"Wins: {self.wins}, Losses: {self.losses}")
        self.writer.flush()
        self.plot_rewards()

    def plot_rewards(self):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.title(f"{self.agent_name} - Episode Rewards")
            plt.plot(self.episode_rewards, marker='o')
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.grid(True)
            plt.tight_layout()
            plot_filename = f"{self.agent_name}_reward_plot.png"
            plt.savefig(plot_filename)
            print(f"Saved reward plot to {plot_filename}")
            plt.show()
        except ImportError:
            print("Matplotlib is required for plotting rewards. Install it with 'pip install matplotlib'.")

    def _load_checkpoint(self, checkpoint_path):
        try:
            loaded_state_dict = torch.load(checkpoint_path, map_location=device, encoding='latin1', weights_only=True)
            current_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            for key, value in loaded_state_dict.items():
                if key in current_state_dict and current_state_dict[key].size() == value.size():
                    filtered_state_dict[key] = value
                else:
                    print(f"Skipping {key} due to size mismatch.")
            current_state_dict.update(filtered_state_dict)
            self.model.load_state_dict(current_state_dict)
            self.checkpoint = checkpoint_path
            print(f"Checkpoint loaded from {checkpoint_path}")
        except Exception as e:
            print("Failed to load checkpoint:", e)

    def periodic_cleanup(self):
        # Clear unnecessary memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Trim replay buffer if too large
        if len(self.replay_buffer) > MAX_REPLAY_BUFFER_SIZE:
            self.replay_buffer = self.replay_buffer[-MAX_REPLAY_BUFFER_SIZE:]

    def _process_action_args(self, chosen_action_id, arg_out, obs):
        """Process and construct arguments for the chosen action."""
        args = []
        reward = 0  # Initialize reward

        # Extract coordinate prediction if available
        if isinstance(arg_out, dict) and "coord_all" in arg_out:
            chosen_coord = arg_out["coord_all"][str(chosen_action_id)]
            screen_xy = chosen_coord.cpu().detach().numpy()[0]
        else:
            screen_xy = None
        
        # Extract queued prediction if available
        chosen_queued = None
        if isinstance(arg_out, dict) and "queued" in arg_out:
            queued_probs = torch.softmax(arg_out["queued"], dim=1).cpu().detach().numpy()[0]
            chosen_queued = int(np.random.choice(2, p=queued_probs))
        
        # Extract minimap prediction if available
        minimap_xy = None
        if isinstance(arg_out, dict) and "minimap" in arg_out:
            minimap_coord = arg_out["minimap"][str(chosen_action_id)]
            minimap_xy = minimap_coord.cpu().detach().numpy()[0]
        
        # Construct arguments for each parameter the action requires
        for arg in self.action_spec[0].functions[chosen_action_id].args:
            if arg.name == "queued" and chosen_queued is not None:
                args.append([chosen_queued])
            elif arg.name == "screen" and screen_xy is not None:
                x = int(min(max(round(screen_xy[0] * arg.sizes[0]), 0), arg.sizes[0]-1))
                y = int(min(max(round(screen_xy[1] * arg.sizes[1]), 0), arg.sizes[1]-1))
                args.append([x, y])
            elif arg.name == "minimap" and minimap_xy is not None:
                x = int(min(max(minimap_xy[0] * arg.sizes[0], 0), arg.sizes[0]-1))
                y = int(min(max(minimap_xy[1] * arg.sizes[1], 0), arg.sizes[1]-1))
                args.append([x, y])
            else:
                # For any other arguments, use random values within the allowed range
                rand_args = [np.random.randint(0, size) for size in arg.sizes]
                args.append(rand_args)
        
        # Log the chosen action and the constructed arguments
        #print(f"Chosen action ID: {chosen_action_id}, Args: {args}")

        # Here you should implement the logic to calculate the reward based on the action taken
        # For example, you might want to check the state of the environment after the action
        # and assign a reward based on that.
        # reward = calculate_reward_based_on_action(chosen_action_id, obs)

        return args, reward

    def log_images(self, screen_tensor, minimap_tensor):
        """Log screen and minimap images to TensorBoard."""
        try:
            # Squeeze out the batch dimension if present
            screen_image = screen_tensor.squeeze(0) if screen_tensor.dim() == 4 else screen_tensor
            minimap_image = minimap_tensor.squeeze(0) if minimap_tensor.dim() == 4 else minimap_tensor

            # Ensure both images are in RGB format (3 x H x W)
            screen_image_vis = self._to_rgb(screen_image)
            minimap_image_vis = self._to_rgb(minimap_image)

            # Convert to bytes format for TensorBoard
            screen_vis = (screen_image_vis * 255).byte()
            minimap_vis = (minimap_image_vis * 255).byte()

            # Log images
            self.writer.add_image('Screen', screen_vis, self.episode_count, dataformats='CHW')
            self.writer.add_image('Minimap', minimap_vis, self.episode_count, dataformats='CHW')
            
        except Exception as e:
            print(f"Warning: Failed to log images: {e}")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint for this agent."""
        if not os.path.exists(self.checkpoint_dir):
            return None
            
        pattern = re.compile(rf".*{self.agent_name}.*episode-(\d+)\.pth")
        checkpoints = []
        for fname in os.listdir(self.checkpoint_dir):
            match = pattern.match(fname)
            if match:
                episode_num = int(match.group(1))
                checkpoints.append((episode_num, os.path.join(self.checkpoint_dir, fname)))
                
        return checkpoints[-1][1] if checkpoints else None

    def _save_checkpoint(self):
        """Save a checkpoint of the model."""
        try:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"dueling-{self.agent_name}-episode-{self.episode_count}.pth"
            )
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")