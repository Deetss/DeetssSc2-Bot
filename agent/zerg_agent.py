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
from model import create_dueling_model  # Use the new dueling model
from agent.a2c import create_a2c_model
from agent.config import DEVICE as device
from agent.reward_mixin import RewardMixin

import torch
import torch.optim as optim

from agent.config import REFRESH_INTERVAL, LR

USE_PRETRAINED = True  # Load a pretrained model if available.
#PRETRAINED_FILENAME = "dueling-Slick_Eagle_0-episode-245.pth"
PRETRAINED_FILENAME = "observer_dqn_checkpoint.pth"

class ZergAgent(RewardMixin, base_agent.BaseAgent):
    def __init__(self, worker_id, agent_name=None, use_pretrained=USE_PRETRAINED, pretrained_filename=PRETRAINED_FILENAME):
        super(ZergAgent, self).__init__()
        self.use_pretrained = use_pretrained
        self.worker_id = worker_id
        self.REFRESH_INTERVAL = REFRESH_INTERVAL
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
        self.replay_buffer = []  # to store (state, action, reward, next_state)
        
        # Variables for logging / performance.
        self.total_reward = 0
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.action_count = 0
        self.start_time = time.time()
        # Initialize TensorBoard SummaryWriter.
        self.writer = SummaryWriter(log_dir=os.path.join("runs", self.agent_name))
        
        # Attempt to load the latest checkpoint for this agent.
        if self.use_pretrained:
            if pretrained_filename and os.path.exists(pretrained_filename):
                print(f"Loading specified checkpoint from {pretrained_filename}")
                self._load_checkpoint(pretrained_filename)
            elif os.path.exists(self.checkpoint_dir):
                pattern = re.compile(rf"dueling-{self.agent_name}-episode-(\d+)\.pth")
                checkpoints = []
                for fname in os.listdir(self.checkpoint_dir):
                    match = pattern.match(fname)
                    if match:
                        episode_num = int(match.group(1))
                        checkpoints.append((episode_num, os.path.join(self.checkpoint_dir, fname)))
                if checkpoints:
                    latest_episode, latest_checkpoint_path = max(checkpoints, key=lambda x: x[0])
                    print(f"Loading latest checkpoint {latest_checkpoint_path}")
                    self._load_checkpoint(latest_checkpoint_path)
                    self.epsilon = 0.1
                    self.episode_count = latest_episode
                else:
                    print("No checkpoints found. Starting with a random model.")
            else:
                print("No checkpoint directory found. Starting with a random model.")
        else:
            print("use_pretrained=False. Starting with a random model.")

    def extract_structured_obs(self, obs):
        """Gather more observation data (control groups, selected units, etc.)."""
        player_data = obs.observation.get("player", [])
        control_groups = obs.observation.get("control_groups", [])
        single_select = obs.observation.get("single_select", [])
        multi_select = obs.observation.get("multi_select", [])
        cargo = obs.observation.get("cargo", [])
        build_queue = obs.observation.get("build_queue", [])

        # Convert each to a flat array. Adjust dimensions as needed.
        cg_flat = np.array(control_groups).flatten() if len(control_groups) else []
        ss_flat = np.array(single_select).flatten() if len(single_select) else []
        ms_flat = np.array(multi_select).flatten() if len(multi_select) else []
        cr_flat = np.array(cargo).flatten() if len(cargo) else []
        bq_flat = np.array(build_queue).flatten() if len(build_queue) else []

        combined = np.concatenate([
            np.array(player_data).flatten(),
            cg_flat, ss_flat, ms_flat, cr_flat, bq_flat
        ]).astype(np.float32)

        desired_size = 31
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
        super(ZergAgent, self).step(obs)
        
        screen_tensor, minimap_tensor = self.preprocess_state(obs)
        obs_state = self.extract_structured_obs(obs)
        obs_state_tensor = torch.tensor(obs_state, dtype=torch.float32).unsqueeze(0).to(device)
        
        if obs.observation.game_loop % 500 == 0 or obs.first():
            # Squeeze out the batch dimension.
            screen_image = screen_tensor.squeeze(0)
            minimap_image = minimap_tensor.squeeze(0)

            # Ensure both images are in RGB format (3 x H x W).
            screen_image_vis = self._to_rgb(screen_image)
            minimap_image_vis = self._to_rgb(minimap_image)

            # Log images exactly as seen by the model.
            self.writer.add_image("Screen", (screen_image_vis * 255).byte(), self.episode_count, dataformats='CHW')
            self.writer.add_image("Minimap", (minimap_image_vis * 255).byte(), self.episode_count, dataformats='CHW')
        
        # Forward pass without chosen action to get Q-values and all coordinate predictions.
        action_logits, _, arg_out = self.model(
            screen_tensor,
            minimap_tensor,
            obs_state_tensor
        )
        action_probs = torch.softmax(action_logits, dim=1).cpu().detach().numpy()[0]
        chosen_action_id = np.random.choice(len(action_probs), p=action_probs)
        
        # If the chosen action is not available, pick one from the pool.
        available_actions = obs.observation["available_actions"]
        if chosen_action_id not in available_actions:
            chosen_action_id = np.random.choice(available_actions)
        
        # Extract the coordinate prediction from the chosen action’s head.
        if isinstance(arg_out, dict) and "coord_all" in arg_out:
            chosen_coord = arg_out["coord_all"][str(chosen_action_id)]
            screen_xy = chosen_coord.cpu().detach().numpy()[0]
        else:
            screen_xy = None
        
        # Pre-compute other arguments from network output if available.
        chosen_queued = None
        if isinstance(arg_out, dict) and "queued" in arg_out:
            queued_probs = torch.softmax(arg_out["queued"], dim=1).cpu().detach().numpy()[0]
            chosen_queued = int(np.random.choice(2, p=queued_probs))
        
        minimap_xy = None
        if isinstance(arg_out, dict) and "minimap" in arg_out:
            minimap_xy = torch.sigmoid(arg_out["minimap"]).cpu().detach().numpy()[0]
        
        # Construct target_coords for replay buffer.
        if screen_xy is not None:
            target_coords = torch.tensor(screen_xy, dtype=torch.float32, device=device)
        else:
            target_coords = torch.zeros(2, dtype=torch.float32, device=device)
        
        # Construct the function call arguments.
        args = []
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
                rand_args = [np.random.randint(0, size) for size in arg.sizes]
                args.append(rand_args)

        reward = self.compute_reward(obs, obs_state)
        if obs.last() and obs.reward <= 0:
            reward = self.episode_reward_accum

        self.total_reward += reward
        next_state_tensor = screen_tensor.clone().detach()
        self.replay_buffer.append((screen_tensor, chosen_action_id, reward, next_state_tensor, target_coords))
        
        return actions.FunctionCall(chosen_action_id, args)

    # def train_online(self):
    #     batch = random.sample(self.replay_buffer, self.batch_size)
    #     states, actions_batch, rewards, next_states, target_coords = zip(*batch)
    #     states = torch.stack(states)
    #     actions_batch = torch.tensor(actions_batch, dtype=torch.int64, device=device)
    #     rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    #     next_states = torch.stack(next_states)
    #     target_coords = torch.stack(target_coords)

    #     noise_std = 0.01
    #     target_coords = target_coords + torch.randn_like(target_coords) * noise_std

    #     dummy_minimap = torch.zeros(states.size(0), 11, 64, 64, dtype=torch.float32, device=device)
    #     dummy_obs_state = torch.zeros((states.size(0), 31), dtype=torch.float32, device=device)

    #     # Forward pass without chosen action to get all coordinate predictions.
    #     q_vals, coord_preds = self.model(states, dummy_minimap, dummy_obs_state, chosen_action=None)
    #     current_q = q_vals.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

    #     gamma = 0.99
    #     with torch.no_grad():
    #         next_q_vals, _ = self.model(next_states, dummy_minimap, dummy_obs_state, chosen_action=None)
    #         max_next_q, _ = next_q_vals.max(dim=1)
    #         target_q = rewards + gamma * max_next_q

    #     raw_reward_weights = (rewards - 0.0) / (rewards.max() + 1e-6)
    #     reward_weights = torch.clamp(raw_reward_weights, 0, 1)
    #     min_weight = torch.tensor(0.1, device=device)
    #     reward_weights = torch.max(reward_weights, min_weight)
        
    #     # For training, select for each sample the coordinate output corresponding to the stored action.
    #     if isinstance(coord_preds, dict) and "coord_all" in coord_preds:
    #         pred_coords_list = []
    #         for i, a in enumerate(actions_batch):
    #             # Each head's prediction is batched so pick index i.
    #             pred = coord_preds["coord_all"][str(a.item())][i]
    #             pred_coords_list.append(pred)
    #         pred_coords = torch.stack(pred_coords_list)
    #     else:
    #         pred_coords = coord_preds  # fallback

    #     loss_fn = CustomLoss()
    #     loss, q_loss_value, coord_loss_value = loss_fn(current_q, target_q, pred_coords, target_coords, reward_weights)

    #     # Add auxiliary losses if needed (for unused heads; update as appropriate).
    #     if isinstance(coord_preds, dict) and "coord_all" in coord_preds:
    #         for key, head_out in coord_preds["coord_all"].items():
    #             # For example, add a dummy loss to all heads (or only for heads not corresponding to actions in batch)
    #             dummy_target = torch.zeros_like(head_out)
    #             aux_loss = 0.01 * torch.nn.functional.mse_loss(head_out, dummy_target)
    #             loss += aux_loss

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and param.grad is None:
    #             print(f"WARNING: No gradient for {name}")
    #         elif param.requires_grad:
    #             grad_norm = param.grad.data.norm()

    #     self.optimizer.step()
    #     self.replay_buffer = self.replay_buffer[-1000:]
    #     self.writer.add_scalar("Loss/Total", loss.item(), self.episode_count)
    #     self.writer.add_scalar("Loss/Q", q_loss_value.item(), self.episode_count)
    #     self.writer.add_scalar("Loss/Coord", coord_loss_value.item(), self.episode_count)

    #     # Log histograms of weights and gradients
    #     for name, param in self.model.named_parameters():
    #         self.writer.add_histogram(f"Weights/{name}", param, self.episode_count)
    #         if param.grad is not None:
    #             self.writer.add_histogram(f"Gradients/{name}", param.grad, self.episode_count)
    #     gc.collect()

    def reset(self):
        super(ZergAgent, self).reset()
        self.episode_rewards.append(self.total_reward)
        self.writer.add_scalar("Reward/Total", self.total_reward, self.episode_count)
        self.total_reward = 0
        self.action_count = 0

        if len(self.replay_buffer) >= self.batch_size:
            # Optionally remove or adjust the call to train_online:
            # self.train_online()
            pass

        self.episode_count += 1
        if self.episode_count % REFRESH_INTERVAL == 0:
            ckpt_path = os.path.join(self.checkpoint_dir, f"dueling-{self.agent_name}-episode-{self.episode_count}.pth")
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Saved agent checkpoint: {ckpt_path}")

        min_epsilon = 0.1
        decay_rate = 0.99
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def log_results(self):
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        elapsed_time = time.time() - self.start_time if self.start_time else 1
        apm = (self.action_count / elapsed_time) * 60
        self.writer.add_scalar("Avg Reward", avg_reward, self.episode_count)
        self.writer.add_scalar("APM", apm, self.episode_count)
        
        # Existing plotting code...
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