from pysc2.lib.features import ScoreCumulative
import numpy as np
import gc

class RewardMixin:
    def __init__(self, *args, **kwargs):
        self.previous_score = 0
        self.previous_idle_production_time = 0
        self.previous_idle_worker_time = 0
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_collected_minerals = 0
        self.previous_collected_vespene = 0
        self.previous_spent_minerals = 0
        self.previous_spent_vespene = 0
        self.previous_army_count = 0
        # Accumulator for intermediate differences.
        self.episode_reward_accum = 0
        self.episode_step_count = 0
        self.previous_worker_count = 0
        self.previous_base_count = 0
        self.max_reward_seen = 1.0  # For reward normalization
        self.state_action_history = {}
        self.exploration_bonus_decay = 0.99  # Decay factor for exploration bonus
        self.exploration_threshold = 0.1  # Minimum exploration bonus
        self.exploration_bonus = 1.0  # Initial exploration bonus
        self.current_state = None
        self.max_history_size = 30000  # Limit the size of the history to 30,000
        self.ending_reward = 0  # Track ending reward
        super().__init__(*args, **kwargs)

    def compute_reward(self, obs, current_state, action):
        # Convert current_state to integers for indexing
        current_state = current_state

        self.episode_step_count += 1
        self.current_state = current_state
        
        # Convert current_state to a tuple to make it hashable
        current_state_tuple = tuple(current_state)  # Convert to int if necessary

        # Get current unit counts for sub-goal rewards
        current_worker_count = len([unit for unit in obs.observation.raw_units 
                                  if unit.alliance == 1 and unit.unit_type in [45, 84]])  # SCV, Probe, Drone
        current_base_count = len([unit for unit in obs.observation.raw_units 
                                if unit.alliance == 1 and unit.unit_type in [18, 59, 86]])  # CC, Nexus, Hatchery
        current_army_count = len([unit for unit in obs.observation.raw_units
                                if unit.alliance == 1 and unit.unit_type not in [45, 84, 18, 59, 86]])  # Exclude workers and bases
        
        # Gather current score values with error handling
        try:
            current_score = obs.observation['score_cumulative'][ScoreCumulative.score]
            idle_prod_time = obs.observation['score_cumulative'][ScoreCumulative.idle_production_time]
            idle_worker_time = obs.observation['score_cumulative'][ScoreCumulative.idle_worker_time]
            killed_units = obs.observation['score_cumulative'][ScoreCumulative.killed_value_units]
            killed_structs = obs.observation['score_cumulative'][ScoreCumulative.killed_value_structures]
            collected_minerals = obs.observation['score_cumulative'][ScoreCumulative.collected_minerals]
            collected_vespene = obs.observation['score_cumulative'][ScoreCumulative.collected_vespene]
            spent_minerals = obs.observation['score_cumulative'][ScoreCumulative.spent_minerals]
            spent_vespene = obs.observation['score_cumulative'][ScoreCumulative.spent_vespene]
        except KeyError as e:
            print(f"KeyError: {e} - Check if the keys exist in obs.observation['score_cumulative']")
            return 0  # or handle the error as needed

        # Compute differences.
        diff_score = current_score - self.previous_score
        diff_idle_prod = idle_prod_time - self.previous_idle_production_time
        diff_idle_worker = idle_worker_time - self.previous_idle_worker_time
        diff_killed_units = killed_units - self.previous_killed_unit_score
        diff_killed_structs = killed_structs - self.previous_killed_building_score
        diff_spent_m = spent_minerals - self.previous_spent_minerals
        diff_spent_v = spent_vespene - self.previous_spent_vespene
        diff_collected_m = collected_minerals - self.previous_collected_minerals
        diff_collected_v = collected_vespene - self.previous_collected_vespene

        # Update previous values.
        self.previous_score = current_score
        self.previous_idle_production_time = idle_prod_time
        self.previous_idle_worker_time = idle_worker_time
        self.previous_killed_unit_score = killed_units
        self.previous_killed_building_score = killed_structs
        self.previous_spent_minerals = spent_minerals
        self.previous_spent_vespene = spent_vespene
        self.previous_collected_minerals = collected_minerals
        self.previous_collected_vespene = collected_vespene
        self.previous_army_count = current_army_count

        # Compute sub-goal rewards
        worker_reward = 0.5 if current_worker_count > self.previous_worker_count else 0
        expansion_reward = 2.0 if current_base_count > self.previous_base_count else 0
        army_reward = 1.0 if current_army_count > self.previous_army_count else 0
        
        # Update previous counts
        self.previous_worker_count = current_worker_count
        self.previous_base_count = current_base_count

        # Dynamic exploration bonus that decreases over time
        self.exploration_bonus *= self.exploration_bonus_decay
        self.exploration_bonus = max(self.exploration_bonus, self.exploration_threshold)

        # Initialize step_reward to 0 at the start of the computation
        step_reward = 0.0

        # Check if the action has been taken for the current state
        if current_state_tuple not in self.state_action_history:
            self.state_action_history[current_state_tuple] = set()

        if action in self.state_action_history[current_state_tuple]:
            # Allow for some repetition without a penalty
            step_reward = 0.1  # Small bonus for repeating a known action
        else:
            # Proceed with the action and update history
            self.state_action_history[current_state_tuple].add(action)


        # Compute main reward components
        step_reward += ( diff_score * (0.001 * (self.episode_step_count / 3000))
                        - diff_idle_prod * 0.001
                        - diff_idle_worker * 0.001
                        + diff_collected_m * 0.001
                        + diff_collected_v * 0.0025
                        + diff_killed_units * 2.5
                        + diff_killed_structs * 5
                        + diff_spent_m * 0.1
                        + diff_spent_v * 0.15)

        # Add exploration bonus
        step_reward += self.exploration_bonus

        # Add delayed rewards if applicable
        if self.is_delayed_reward_condition_met():
            step_reward += self.calculate_delayed_reward()

        # Add sub-goal rewards
        step_reward += worker_reward + expansion_reward + army_reward

        time_penalty = 0.01 * (1 + (self.episode_step_count / 3000))
        step_reward -= time_penalty

        # Clip rewards to reasonable ranges
        step_reward = max(min(step_reward, 5.0), -5.0)

        # Update max reward for normalization
        self.max_reward_seen = max(self.max_reward_seen, abs(step_reward))

        # Normalize reward
        normalized_reward = step_reward / self.max_reward_seen

        # Accumulate the intermediate reward
        self.episode_reward_accum += normalized_reward

        # Log the current reward
        #print(f"Current reward: {normalized_reward}")

        # Clear old entries in state_action_history if it exceeds the limit
        if len(self.state_action_history) > self.max_history_size:
            keys_to_remove = list(self.state_action_history.keys())[:len(self.state_action_history) - self.max_history_size]
            for key in keys_to_remove:
                del self.state_action_history[key]
            
            # Trigger garbage collection
            
        gc.collect()
        return normalized_reward

    def is_delayed_reward_condition_met(self):
        # Define your condition for delayed rewards here
        # For example, you might check if a certain number of steps have passed
        return self.episode_step_count > 1200  # Example condition
    
    def calculate_delayed_reward(self):
        # Calculate the delayed reward based on the current state of the game
        # This could include rewards for achieving certain milestones, such as
        # building a certain structure, reaching a certain army size, or collecting
        # a certain amount of resources.
        delayed_reward = 0.0

        if self.current_state['army_count'] > 10:
            delayed_reward += 2.0  # Reward for having more than 10 army units

        # if self.previous_collected_minerals + self.previous_collected_vespene > 1000:
        #     delayed_reward += 3.0  # Reward for collecting more than 1000 resources

        return delayed_reward

    def reset(self):
        # Reset episode variables to prevent memory accumulation
        self.episode_reward_accum = 0
        self.ending_reward = 0  # Reset ending reward
        self.state_action_history.clear()  # Clear history at the end of each episode
        # Reset previous values to avoid carryover from previous episodes
        self.previous_score = 0
        self.previous_worker_count = 0
        self.previous_base_count = 0
        self.previous_army_count = 0
        # ... reset other variables as needed ...
        print("Resetting RewardMixin state.")

    def log_episode(self):
        # Log the ending reward at the end of the episode
        print(f"Ending reward for the episode: {self.ending_reward}")
        # Log to TensorBoard or any other logging mechanism you are using
        self.writer.add_scalar("Reward/Ending", self.ending_reward, self.episode_count)

    def finish_episode(self, reward):
        self.ending_reward = reward  # Store the ending reward
        self.log_episode()  # Log the ending reward