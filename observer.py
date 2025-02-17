from pathlib import Path
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np
from absl import app

from qlearn import QLearningTable
from reward_utils import compute_reward

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = 1
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_OVERLORD = units.Zerg.Overlord
_HATCHERY = units.Zerg.Hatchery

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

REWARD_PER_MINERAL = 0.0001
REWARD_PER_GAS = 0.0001

class ObserverAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ObserverAgent, self).__init__()
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_action = None
        self.previous_state = None
        self.previous_minerals = None
        self.previous_gas = None
        # Initialize Q-learning model once with available smart actions.
        self.q_table = QLearningTable(actions=list(range(len(actions.FUNCTIONS))))

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def step(self, obs):
        super(ObserverAgent, self).step(obs)

        player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        smart_actions = actions.FUNCTIONS  # for action lookup; q_table is already set.
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        depot_y, depot_x = (unit_type == _OVERLORD).nonzero()
        supply_depot_count = 1 if depot_y.any() else 0

        hatchery_y, hatchery_x = (unit_type == _HATCHERY).nonzero()
        hatchery_count = 1 if hatchery_y.any() else 0

        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        current_minerals = obs.observation['player'][1]
        current_gas = obs.observation['player'][2]

        current_state = [
            supply_depot_count,
            hatchery_count,
            supply_limit,
            army_supply,
            killed_unit_score,
            killed_building_score,
            current_minerals,
            current_gas,
        ]
        
        reward = compute_reward(self, obs, current_state)

        # Award a victory bonus or defeat penalty if the game is over.
        if obs.last() and obs.reward > 0:
            VICTORY_BONUS = 10  
            reward += VICTORY_BONUS
        elif obs.last() and obs.reward < 0:
            DEFEAT_PENALTY = -10
            reward += DEFEAT_PENALTY

        print(f"Reward: {reward}")

        # Update Q-learning model with the transition from the previous step.
        if self.previous_action is not None:
            self.q_table.learn(str(self.previous_state),
                               self.previous_action,
                               reward,
                               str(current_state))

        # Instead of taking an action, always record and return no_op.
        no_op = actions.FUNCTIONS.no_op.id
        print(f"Recorded state: {current_state} with reward: {reward}. Using no_op.")

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_minerals = current_minerals
        self.previous_gas = current_gas
        self.previous_state = current_state
        self.previous_action = no_op

        # In replay mode the returned action is essentially ignored.
        return actions.FunctionCall(no_op, [])

def main(unused_argv):
    # Set the path to your replay file here.
    replay_name = "raynor.SC2Replay"
    home_replay_folder = Path.home() / "OneDrive" / "Documents" / "StarCraft II" / "Replays"
    replay_path = home_replay_folder / replay_name
    
    agent = ObserverAgent()
    try:
        with sc2_env.SC2Env(
            replay_path=replay_path,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True),
            step_mul=16,
            game_steps_per_episode=0,
            visualize=True) as env:

            agent.setup(env.observation_spec(), env.action_spec())
            timesteps = env.reset()
            agent.reset()

            while True:
                # Call step to update the Q-table from the replay data.
                agent.step(timesteps[0])
                if timesteps[0].last():
                    break
                timesteps = env.step([])
    except KeyboardInterrupt:
        pass
    finally:
        # Save Q-learning model state on session exit.
        agent.q_table.save_model("q_table.csv")

if __name__ == "__main__":
    app.run(main)