import os
import torch
import torch.optim as optim
import numpy as np

from pysc2.agents import base_agent
from pysc2 import run_configs
from pysc2.lib import actions, features
from s2clientprotocol import sc2api_pb2 as sc_pb
from agent.reward_mixin import RewardMixin
from agent.a2c import create_a2c_model
from agent.config import LR
class ObserverAgent(RewardMixin, base_agent.BaseAgent):
    """Minimal agent to train an A2C model from replay data."""

    def __init__(self, map_name):
        super(ObserverAgent, self).__init__()
        self.map_name = map_name
        self._initialize_model()
        self.reset()

    def _initialize_model(self):
        self.num_actions = len(actions.FUNCTIONS)
        self.model = create_a2c_model(num_actions=self.num_actions, structured_size=31)
        self._load_checkpoint()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.gamma = 0.99
        self.transitions = []  # Stores (screen, action, reward)

    def _load_checkpoint(self):
        self.checkpoint_path = "observer_dqn_checkpoint.pth"
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=torch.device("cpu"))
                self.model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded checkpoint from {self.checkpoint_path}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

    def reset(self):
        super(ObserverAgent, self).reset()
        self.reward_accum = 0
        self.transitions = []

    def preprocess_obs(self, obs):
        # Use 'rgb_screen' and 'rgb_minimap' if available; otherwise, default to zeros.
        if hasattr(obs.observation, "rgb_screen"):
            screen = torch.tensor(np.array(getattr(obs.observation, "rgb_screen")), dtype=torch.float32)
            screen = screen.permute(2, 0, 1).unsqueeze(0)
        else:
            screen = torch.zeros(1, 3, 192, 256)
        if hasattr(obs.observation, "rgb_minimap"):
            minimap = torch.tensor(np.array(getattr(obs.observation, "rgb_minimap")), dtype=torch.float32)
            minimap = minimap.permute(2, 0, 1).unsqueeze(0)
        else:
            minimap = torch.zeros(1, 3, 128, 128)
        # Use the 'player' info as the structured input—pad/truncate to length 31.
        if hasattr(obs.observation, "player"):
            player = getattr(obs.observation, "player")
        else:
            player = []
        structured = torch.tensor(player, dtype=torch.float32).unsqueeze(0)
        if structured.size(1) < 31:
            pad = torch.zeros(1, 31 - structured.size(1))
            structured = torch.cat([structured, pad], dim=1)
        elif structured.size(1) > 31:
            structured = structured[:, :31]
        return screen, minimap, structured

    def step(self, obs):
        screen, minimap, structured = self.preprocess_obs(obs)
        with torch.no_grad():
            logits, value, _ = self.model(screen, minimap, structured)
            probs = torch.softmax(logits, dim=1)
        action = int(torch.multinomial(probs, 1))
        
        # Ensure current_state is defined properly
        current_state = {
            'worker_count': obs.observation.player[7],  # idle worker count
            'army_count': obs.observation.player[8],    # army count
            'base_count': obs.observation.player[6],     # base count
        }
        
        reward = self.compute_reward(obs, current_state, action)
        self.reward_accum += reward
        self.transitions.append((screen, action, reward))
        return actions.FunctionCall(action, [])

    def finish_episode(self):
        R = 0
        returns = []
        for (_, _, reward) in reversed(self.transitions):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Prepare batch of screen observations and actions.
        screens = torch.cat([t[0] for t in self.transitions], dim=0)
        actions_batch = torch.tensor([t[1] for t in self.transitions], dtype=torch.long)
        
        # Dummy inputs for minimap and structured data.
        dummy_minimap = torch.zeros(len(self.transitions), 3, 128, 128)
        dummy_structured = torch.zeros(len(self.transitions), 31)
        
        logits, values, _ = self.model(screens, dummy_minimap, dummy_structured)
        values = values.squeeze(1)
        advantages = returns - values.detach()
        log_probs = torch.log_softmax(logits, dim=1)
        chosen_log_probs = log_probs.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
        
        policy_loss = - (chosen_log_probs * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Log metrics to TensorBoard
        self.writer.add_scalar("Loss/Policy", policy_loss.item(), self.episode_count)
        self.writer.add_scalar("Loss/Value", value_loss.item(), self.episode_count)
        self.writer.add_scalar("Reward/Total", self.reward_accum, self.episode_count)
        
        torch.save(self.model.state_dict(), self.checkpoint_path)
        print(f"Episode complete. Total reward: {self.reward_accum:.2f}, Loss: {loss.item():.2f}")

    def learn_from_replay(self, replay_path, player_id):
        # Load replay data and (if available) map data.
        run_config = run_configs.get()
        replay_data = run_config.replay_data(replay_path)
        map_data = None
        try:
            info = run_config.replay_info(replay_data)
            if info.local_map_path:
                map_data = run_config.map_data(info.local_map_path, len(info.player_info))
        except Exception as e:
            print("Error retrieving replay info:", e)

        # Use minimal interface options.
        interface = sc_pb.InterfaceOptions(
            raw=True,
            raw_affects_selection=True,
            raw_crop_to_playable_area=True,
            score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=24)
        )
        replay_request = sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id,
            disable_fog=True
        )

        with run_config.start() as controller:
            controller.start_replay(replay_request)
            self.reset()
            while True:
                controller.step()
                obs = controller.observe()
                if obs is None:
                    break
                # Process each observation.
                self.step(obs)
                if hasattr(obs, "player_result") and obs.player_result:
                    break
            self.finish_episode()