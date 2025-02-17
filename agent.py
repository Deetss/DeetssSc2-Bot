from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np
import multiprocessing
from absl import app
from collections import deque
import signal
import time  # Add this import

from qlearn import QLearningTable
from reward_mixin import RewardMixin
from reward_utils import compute_reward

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = 1
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_OVERLORD = units.Zerg.Overlord
_HATCHERY = units.Zerg.Hatchery

# Number of episodes per worker.
NUM_EPISODES = 3

class ZergAgent(RewardMixin, base_agent.BaseAgent):
    def __init__(self, worker_id):
        super(ZergAgent, self).__init__()
        self.worker_id = worker_id

        self.history_length = 50  # Define history length
        self.state_history = deque(maxlen=self.history_length)  # Initialize deque

        self.total_reward = 0  # Initialize total reward
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.action_count = 0  # Initialize action count
        self.start_time = None  # Initialize start time
        # Initialize Q-learning model once with available smart actions.
        self.q_table = QLearningTable(
            actions=list(range(len(actions.FUNCTIONS))),
            history_length=self.history_length,
            db_filename=f"q_table_worker_{worker_id}.db",
            db_dir="./db"  # make sure this matches your saved directory
        )
        
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def step(self, obs):
        super(ZergAgent, self).step(obs)
        self.action_count += 1  # Increment action count

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
        worker_count = len([unit for unit in obs.observation.feature_units
                            if unit.unit_type == units.Zerg.Drone])
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        current_minerals = obs.observation['player'][1]
        current_gas = obs.observation['player'][2]

        structure_count = len([unit for unit in obs.observation.feature_units
                               if unit.unit_type in [21, 22, 26, 105, 110, 111, 150, 151]])  # Structure unit types

        # Additional features
        idle_worker_count = obs.observation['player'][7]
        collected_minerals = obs.observation['score_cumulative'][7]
        collected_vespene = obs.observation['score_cumulative'][8]
        spent_minerals = obs.observation['score_cumulative'][11]
        spent_vespene = obs.observation['score_cumulative'][12]

        current_state = [
            supply_depot_count,
            hatchery_count,
            supply_limit,
            army_supply,
            killed_unit_score,
            killed_building_score,
            current_minerals,
            current_gas,
            worker_count,
            structure_count,
            idle_worker_count,
            collected_minerals,
            collected_vespene,
            spent_minerals,
            spent_vespene
        ]

        # Add current state to history
        self.state_history.append(current_state)

        # Pad history if it's shorter than history_length
        while len(self.state_history) < self.history_length:
            self.state_history.append(current_state)  # Duplicate current state

        # Concatenate the state history into a single feature vector
        historical_state = np.concatenate(list(self.state_history)).tolist()

        reward = compute_reward(self, obs, current_state)  # or historical state?
        self.total_reward += reward

        if self.previous_action is not None:
            self.q_table.learn(str(self.previous_state),  # str(self.previous_historical_state)
                               self.previous_action,
                               reward,
                               str(historical_state))  # str(current_state)

        rl_action = self.q_table.choose_action(historical_state)  # current_state

        available_actions = obs.observation["available_actions"]
        if rl_action not in available_actions:
            rl_action = np.random.choice(available_actions)
        if self.action_spec[0].functions[rl_action]:
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec[0].functions[rl_action].args]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_minerals = current_minerals
        self.previous_gas = current_gas
        self.previous_state = historical_state  # current_state
        self.previous_action = rl_action
        self.previous_army_supply = army_supply
        self.previous_structure_count = structure_count

        return actions.FunctionCall(rl_action, args)

    def reset(self):
        super(ZergAgent, self).reset()
        self.state_history.clear()  # Clear history at the start of each episode
        self.episode_rewards.append(self.total_reward)
        print(f"Worker {self.worker_id} - Total reward: {self.total_reward}")
        self.total_reward = 0
        self.action_count = 0  # Reset action count
        self.start_time = time.time()  # Reset start time

    def log_results(self):
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        elapsed_time = time.time() - self.start_time if self.start_time else 1
        apm = (self.action_count / elapsed_time) * 60  # Calculate APM
        print(f"Worker {self.worker_id} - Wins: {self.wins}, Losses: {self.losses}, Average Reward: {avg_reward}, APM: {apm:.2f}")

def train_agent(worker_id):
    import sys
    from absl import flags
    flags.FLAGS(sys.argv)

    agent = ZergAgent(worker_id)
    with sc2_env.SC2Env(
            map_name="AbyssalReef",
            players=[
                sc2_env.Agent(sc2_env.Race.zerg),
                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.harder)
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True,
            ),
            step_mul=16,
            game_steps_per_episode=0,
            visualize=False) as env:

        agent.setup(env.observation_spec(), env.action_spec())

        for episode in range(NUM_EPISODES):
            timesteps = env.reset()
            agent.reset()
            while True:
                step_actions = [agent.step(timesteps[0])]
                if timesteps[0].last():
                    if timesteps[0].reward > 0:
                        agent.wins += 1
                    else:
                        agent.losses += 1
                    break
                timesteps = env.step(step_actions)
        # Instead of calling a heavy save_model here using VACUUM,
        # commit any outstanding transactions and close the connection.
        agent.q_table.conn.commit()
        agent.q_table.conn.close()
        agent.log_results()
    return

def main(unused_argv):
    num_workers = 3  # Adjust as needed.
    processes = []
    db_dir = "./db"  # Ensure this matches the directory used by QLearningTable

    def handle_sigint(signum, frame):
        print("SIGINT received, merging Q-tables...")
        QLearningTable.merge_q_tables(num_workers, "q_table_shared.db", db_dir=db_dir)
        for p in processes:
            p.terminate()
        exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    for worker_id in range(num_workers):
        print(f"Starting worker {worker_id}")
        p = multiprocessing.Process(target=train_agent, args=(worker_id,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join(timeout=300)  # wait up to 5 minutes per process
        if p.is_alive():
            print("A process is taking too long to finish; terminating it.")
            p.terminate()

    print("All processes finished, starting merge...")
    QLearningTable.merge_q_tables(num_workers, "q_table_shared.db", db_dir=db_dir)
    print("Q-table merge complete. Exiting program.")

if __name__ == "__main__":
    app.run(main)