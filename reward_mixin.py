from sc2.ids.unit_typeid import UnitTypeId

class RewardMixin:
    def __init__(self, *args, **kwargs):
        # Ensure previous counts exist regardless of inherited base
        self.previous_army_count = 0
        self.previous_worker_count = 0
        self.previous_enemy_army = 0
        self.previous_enemy_structures = 0
        super().__init__(*args, **kwargs)

    def compute_reward(self):
        reward = 0.0
        resource_score = 1
        reward += resource_score

        if self.supply_left <= 0:
            reward -= 1.0
        else:
            reward += 0.2

        if self.supply_workers < 20:
            reward -= 1.0

        if self.supply_army < 10:
            reward -= 1.0

        if (self.supply_cap - self.supply_left) > 0:
            # Compute army count from all_units filtering for army units
            army_units = self.all_units.filter(lambda u: u.can_attack).amount
            reward += (army_units / (self.supply_cap - self.supply_left))

        if self.townhalls and self.townhalls.amount >= 2:
            reward += 0.5

        # Idle workers
        idle_workers = self.workers.idle.amount
        reward -= 0.05 * idle_workers

        # Lost army
        current_army = self.all_units.filter(lambda u: u.can_attack).amount
        lost_army = max(0, self.previous_army_count - current_army)
        reward -= 0.5 * lost_army

        # Lost workers
        current_worker_count = self.workers.amount
        lost_workers = max(0, self.previous_worker_count - current_worker_count)
        reward -= 1.0 * lost_workers

        # Enemy kills
        current_enemy_army = self.enemy_units.filter(lambda u: u.can_attack).amount
        killed_enemy_army = max(0, self.previous_enemy_army - current_enemy_army)
        reward += 0.5 * killed_enemy_army

        current_enemy_structures = self.enemy_structures.amount
        destroyed_structures = max(0, self.previous_enemy_structures - current_enemy_structures)
        reward += 1.0 * destroyed_structures

        return reward