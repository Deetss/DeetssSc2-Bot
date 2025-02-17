from pysc2.lib.features import ScoreCumulative

def compute_reward(self, obs, current_state):
    # Gather current score values
    current_score = obs.observation['score_cumulative'][ScoreCumulative.score]
    idle_prod_time = obs.observation['score_cumulative'][ScoreCumulative.idle_production_time]
    idle_worker_time = obs.observation['score_cumulative'][ScoreCumulative.idle_worker_time]
    killed_units = obs.observation['score_cumulative'][ScoreCumulative.killed_value_units]
    killed_structures = obs.observation['score_cumulative'][ScoreCumulative.killed_value_structures]
    collected_minerals = obs.observation['score_cumulative'][ScoreCumulative.collected_minerals]
    collected_vespene = obs.observation['score_cumulative'][ScoreCumulative.collected_vespene]
    spent_minerals = obs.observation['score_cumulative'][ScoreCumulative.spent_minerals]
    spent_vespene = obs.observation['score_cumulative'][ScoreCumulative.spent_vespene]

    # Compute differences
    diff_score = current_score - self.previous_score
    diff_idle_prod = idle_prod_time - self.previous_idle_production_time
    diff_idle_worker = idle_worker_time - self.previous_idle_worker_time
    diff_killed_units = killed_units - self.previous_killed_unit_score
    diff_killed_structs = killed_structures - self.previous_killed_building_score
    diff_collected_m = collected_minerals - self.previous_collected_minerals
    diff_collected_v = collected_vespene - self.previous_collected_vespene
    diff_spent_m = spent_minerals - self.previous_spent_minerals
    diff_spent_v = spent_vespene - self.previous_spent_vespene

    # Assign rewards (positive or negative)
    reward = 0
    reward += diff_score * 0.001
    reward -= diff_idle_prod * 0.05
    reward -= diff_idle_worker * 0.05
    reward += diff_killed_units * 0.04
    reward += diff_killed_structs * 0.1
    reward += diff_collected_m * 0.0001
    reward += diff_collected_v * 0.0001
    reward += diff_spent_m * 0.0001
    reward += diff_spent_v * 0.0001

    # Big reward or penalty for game end
    if obs.last():
        if obs.reward > 0:
            reward += 10
        else:
            reward -= 10

    # Update previous values
    self.previous_score = current_score
    self.previous_idle_production_time = idle_prod_time
    self.previous_idle_worker_time = idle_worker_time
    self.previous_killed_unit_score = killed_units
    self.previous_killed_building_score = killed_structures
    self.previous_collected_minerals = collected_minerals
    self.previous_collected_vespene = collected_vespene
    self.previous_spent_minerals = spent_minerals
    self.previous_spent_vespene = spent_vespene

    return reward