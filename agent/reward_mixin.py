from pysc2.lib.features import ScoreCumulative

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
        # Accumulator for intermediate differences.
        self.episode_reward_accum = 0
        super().__init__(*args, **kwargs)

    def compute_reward(self, obs, current_state):
        # Gather current score values.
        current_score = obs.observation['score_cumulative'][ScoreCumulative.score]
        idle_prod_time = obs.observation['score_cumulative'][ScoreCumulative.idle_production_time]
        idle_worker_time = obs.observation['score_cumulative'][ScoreCumulative.idle_worker_time]
        killed_units = obs.observation['score_cumulative'][ScoreCumulative.killed_value_units]
        killed_structs = obs.observation['score_cumulative'][ScoreCumulative.killed_value_structures]
        collected_minerals = obs.observation['score_cumulative'][ScoreCumulative.collected_minerals]
        collected_vespene = obs.observation['score_cumulative'][ScoreCumulative.collected_vespene]
        spent_minerals = obs.observation['score_cumulative'][ScoreCumulative.spent_minerals]
        spent_vespene = obs.observation['score_cumulative'][ScoreCumulative.spent_vespene]

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

        # Sum up the differences. You can adjust these factors as needed.
        step_reward = (diff_score * 0.001
                    - diff_idle_prod * 0.1
                    - diff_idle_worker * 0.1
                    + diff_collected_m * 0.1
                    + diff_collected_v * 0.1
                    + diff_killed_units * 1
                    + diff_killed_structs * 2
                    + diff_spent_m * 0.1
                    + diff_spent_v * 0.1 )

        # Accumulate the intermediate reward.
        self.episode_reward_accum += step_reward

        # For most steps, return a small exploration bonus.
        exploration_bonus = 0.01

        if obs.last():
            # At the end, consider the accumulated reward plus a bonus/penalty for win/loss.
            final_reward = self.episode_reward_accum
            if obs.reward > 0:
                final_reward += 10  # victory bonus
            else:
                final_reward -= 15  # defeat penalty
                # Penalize if the agent was stationary for too long
                if self.previous_idle_production_time > 0 or self.previous_idle_worker_time > 0:
                    final_reward -= 5
            # Reset accumulator for the next episode.
            self.episode_reward_accum = 0
            return final_reward

        return exploration_bonus