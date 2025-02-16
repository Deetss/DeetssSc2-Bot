import random
from typing import Optional
import cv2
import numpy as np
import time
from collections import OrderedDict

import os
from dotenv import load_dotenv
load_dotenv()

import sc2
from sc2.main import run_game
from sc2 import maps, position
from sc2.data import Race, Difficulty, Result
from sc2.player import Bot, Computer, Human
from sc2.constants import *
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2, Point3
from sc2.ids.upgrade_id import UpgradeId
from sc2.data import race_gas, race_worker, race_townhalls, ActionResult, Attribute, Race
from sc2.bot_ai import BotAI

from keras.models import load_model

HEADLESS = True

class DeetssBot(BotAI):
    def __init__(self, use_model: bool = False):
        super().__init__()
        self.use_model = use_model
        if self.use_model:
            self.model = load_model("BasicCNN-Final-10-epochs-0.001-LR.keras")
        else:
            self.model = None
        self.buildStuffInverval = 2
        self.ITERATIONS_PER_MINUTE = 200
        self.MAX_WORKERS = 55
        self.last_known_enemy_count = 0
        self.train_data = []
        self.do_something_after = 0
        self.exactExpansionLocations = []
        self.townhalls = None
        # exploration factor for random actions:
        self.epsilon = 0.1  # 10% random actions

    def compute_reward(self):
        reward = 0.0
        
        # Economy reward: reward collecting minerals and vespene above a baseline.
        resource_score = (self.minerals + self.vespene) / 3000.0
        reward += resource_score

        # Penalize being supply capped
        if self.supply_left <= 0:
            reward -= 1.0

        # Reward for military production and readiness: e.g., if you have more army units relative to supply, reward.
        army_units = len(self.units({UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.HYDRALISK}))
        if self.supply_cap - self.supply_left > 0:
            reward += (army_units / (self.supply_cap - self.supply_left))

        # Reward for successful expansion (could be based on timing, number of townhalls, etc.)
        if self.townhalls and self.townhalls.amount >= 2:
            reward += 0.5

        # Additional domain-specific improvements can be added here once you track outcomes of attacks etc.
        
        return reward

    async def on_step(self, iteration):
        larvae: Units = self.larva
        forces: Units = self.units.of_type({UnitTypeId.ZERGLING, UnitTypeId.HYDRALISK})
        self.iteration = iteration

        self.currentDroneCountIncludingPending = self.workers.amount + self.already_pending(
            UnitTypeId.DRONE) + self.units(UnitTypeId.EXTRACTOR).ready.filter(lambda x: x.vespene_contents > 0).amount

        if iteration == 0:
            print(self._game_info.map_ramps)
            if self.enemy_start_locations[0]:
                self.units(UnitTypeId.OVERLORD).random.move(self.enemy_start_locations[0]) 
            await self.chat_send("(glhf)")
            await self.findExactExpansionLocations()
        
        if iteration % 5 == 0:
            await self.intel()
            self.log_state(iteration)  # log state image for inspection
        
        # Build complete state representation.
        state = self.get_state_representation()

        # Use random actions if not using a model or within the first 100 iterations.
        if not self.use_model or iteration < 100:
            action = random.randint(0, 3)
            print(f"Iteration {iteration}, Random action (forced): {action}")
        else:
            if random.random() < self.epsilon:
                action = random.randint(0, 3)
                print(f"Iteration {iteration}, Randomly chosen action: {action}")
            else:
                predictions = self.model.predict(state)[0]
                action = int(np.argmax(predictions))
                print(f"Iteration {iteration}, Predictions: {predictions}, Chosen action: {action}")

        # Compute reward and record
        reward = self.compute_reward()
        self.train_data.append((state, action, reward))

        await self.handle_action(action)

    async def findExactExpansionLocations(self):
      # execute this on start, finds all expansions where creep tumors should not be build near
      self.exactExpansionLocations = []
      for loc in self.expansion_locations.keys():
          # TODO: change mindistancetoresource so that a hatch still has room to be built
          self.exactExpansionLocations.append(await self.find_placement(UnitTypeId.HATCHERY, loc, placement_step=1))

    async def intel(self):
        struct_dict = {
            UnitTypeId.HATCHERY: [4, (0, 255, 0)],
            UnitTypeId.EXTRACTOR: [2, (55, 200, 0)],
            UnitTypeId.SPAWNINGPOOL: [2, (200, 100, 0)],
            UnitTypeId.EVOLUTIONCHAMBER: [2, (150, 150, 0)],
            UnitTypeId.ROACHWARREN: [2, (255, 0, 0)],
            UnitTypeId.HYDRALISKDEN: [2, (255, 100, 0)],
        }

        unit_dict = {
            UnitTypeId.OVERLORD: [2, (20, 235, 0)],
            UnitTypeId.DRONE: [1, (55, 200, 0)],
            UnitTypeId.HYDRALISK: [1, (255, 100, 10)],
            UnitTypeId.ROACH: [1, (255, 10, 10)],
            UnitTypeId.ZERGLING: [1, (255, 0, 0)],
        }


        # for game_info: https://github.com/Dentosal/python-sc2/blob/master/sc2/game_info.py#L162
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0],3), np.uint8)

        for unit_type in unit_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position

                cv2.circle(game_data, (int(pos[0]), int(
                    pos[1])), unit_dict[unit_type][0], unit_dict[unit_type][1], -1)  # BGR
            
        for struct_type in struct_dict:
            for struct in self.structures(struct_type).ready:
                pos = struct.position

                cv2.circle(game_data, (int(pos[0]), int(
                    pos[1])), struct_dict[struct_type][0], struct_dict[struct_type][1], -1)  # BGR

        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(
                    pos[1])), 1, (200, 50, 255), -1)
            else:
                cv2.circle(game_data, (int(pos[0]), int(
                    pos[1])), 4, (0, 0, 255), -1)




        for enemy_unit in self.enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]

                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1 , (55, 0, 155), -1) 
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1,  (55, 0, 155), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units({UnitTypeId.HYDRALISK, UnitTypeId.ROACH, UnitTypeId.ZERGLING})) / \
            (self.supply_cap-self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        
        cv2.line(game_data, (0, 19), (int(line_max*military_weight),
                                      19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15),
                 (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11),
                 (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7),
                 (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3),
                 (0, 255, 25), 3)  # minerals minerals/1500
        

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)

        # At the end of intel(), log key info to verify what is being drawn
        num_units = sum([len(self.units(u).ready) for u in [UnitTypeId.DRONE, UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.HYDRALISK]])
        num_structs = sum([len(self.structures(s).ready) for s in [UnitTypeId.HATCHERY, UnitTypeId.SPAWNINGPOOL, UnitTypeId.EVOLUTIONCHAMBER]])
        print(f"Intel update: {num_units} units and {num_structs} structures drawn.")

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    def log_state(self, iteration):
      debug_dir = "bot_vision/debug"
      filename = f"{debug_dir}/debug_state_{iteration}.png"
      cv2.imwrite(filename, self.flipped)

      # List debug images and remove older ones if there are more than 50
      files = [f for f in os.listdir(debug_dir) if f.startswith("debug_state_") and f.endswith(".png")]
      if len(files) > 50:
          def extract_iter(f):
              try:
                  return int(f.replace("debug_state_", "").replace(".png", ""))
              except ValueError:
                  return 0
          files.sort(key=extract_iter)
          files_to_remove = files[:-50]  # keep only last 50
          for old_file in files_to_remove:
              os.remove(os.path.join(debug_dir, old_file))

    def get_state_representation(self):
        """
        Build a state image that matches the model’s expected dimensions.
        Model expects input shape: (1, 176, 200, 3)
        After processing through conv layers, the flattened vector is 64512.
        """
        # If the flipped image doesn't exist, initialize it with zeros.
        if not hasattr(self, "flipped") or self.flipped is None:
            # Default image shape set to 176x200; cv2.resize uses (width, height)
            self.flipped = np.zeros((176, 200, 3), dtype=np.uint8)
        else:
            # Ensure self.flipped is resized to (176,200,3) if needed.
            if self.flipped.shape[:2] != (176, 200):
                self.flipped = cv2.resize(self.flipped, (200, 176))
        
        # Normalize pixel values to [0, 1]
        state = self.flipped.astype(np.float32) / 255.0
        # Add batch dimension
        state = state.reshape((1, 176, 200, 3))
        return state
    
    async def handle_action(self, action):
        """
        Map the RL action to an in-game macro.
        For instance:
          0: focus on economy (build workers, overlords or expand)
          1: build an army
          2: attack enemy units/structures
          3: wait/do nothing
        """
        if action == 0:
            await self.handle_economy()
        elif action == 1:
            await self.build_army()
        elif action == 2:
            await self.attack()
        elif action == 3:
            # wait: let the bot collect resources
            pass

    async def handle_economy(self):
        """
        Instead of always building overlords when supply is low,
        choose our economy actions based on the RL decision.
        You might even subdivide:
           - if resources are low: build workers
           - if supply is low: build overlords
           - if expansion criteria met: expand
        """
        # Example decision-making (you might later integrate another prediction or threshold):
        if self.supply_left <= 2 and self.can_afford(UnitTypeId.OVERLORD):
            self.larva.random.train(UnitTypeId.OVERLORD)
        elif self.supply_workers < self.structures(UnitTypeId.HATCHERY).amount * 16:
            await self.build_workers()
        elif self.can_afford(UnitTypeId.HATCHERY):
            await self.expand()

    async def build_workers(self):
        if self.supply_workers < self.structures(UnitTypeId.HATCHERY).amount * 16 and self.larva and self.can_afford(UnitTypeId.DRONE) and self.minerals < 1000:
                self.larva.random.train(UnitTypeId.DRONE)

    async def build_army(self):
        if self.larva:
            # make some hydras if den is ready
            if self.structures(UnitTypeId.HYDRALISKDEN).ready and self.can_afford(UnitTypeId.HYDRALISK):
                self.larva.random.train(UnitTypeId.HYDRALISK)
            
            # make some roaches if warren is ready
            if self.structures(UnitTypeId.ROACHWARREN).ready and self.can_afford(UnitTypeId.ROACH):
                if not self.units(UnitTypeId.HYDRALISK).idle:
                    self.larva.random.train(UnitTypeId.ROACH)
                elif self.units(UnitTypeId.ROACH).amount < self.units(UnitTypeId.HYDRALISK).amount * 0.75:
                    self.larva.random.train(UnitTypeId.ROACH)

            # spawn some zerglings if spawning pool is ready
            if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.can_afford(UnitTypeId.ZERGLING):
                if not self.units(UnitTypeId.ROACH).idle:
                    self.larva.random.train(UnitTypeId.ZERGLING)
                elif self.units(UnitTypeId.ZERGLING).amount < self.units(UnitTypeId.ROACH).amount / 2:
                    self.larva.random.train(UnitTypeId.ZERGLING)                
            
        # queens @ pool
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready and (self.units(UnitTypeId.QUEEN).amount + self.already_pending(UnitTypeId.QUEEN) < self.townhalls.ready.idle.amount * self.time / 90) and not self.units(UnitTypeId.QUEEN).amount >= self.townhalls.ready.idle.amount * 5 and self.can_afford(UnitTypeId.QUEEN):
            hatcheries = self.townhalls.ready.idle
            for hatch in hatcheries:
                hatch.train(UnitTypeId.QUEEN)
                print("Queen incoming!")

    async def attack(self):
      if not self.use_model:
          # Choose a random action if no model is used
          model_choice = random.randint(0, 3)
          print(f"Random attack selection: {model_choice}")
      else:
          state_input = self.flipped.reshape((1, self.flipped.shape[0], self.flipped.shape[1], 3))
          predictions = self.model.predict(state_input)[0]
          model_choice = int(np.argmax(predictions))
          print(f"Model predictions: {predictions}, Chosen attack action: {model_choice}")

      target = None

      if model_choice == 0:
          # No attack: wait/do nothing
          wait = random.randrange(20, 165)
          self.do_something_after = self.iteration + wait
      elif model_choice == 1:
          # Attack enemy unit closest to a random townhall
          if self.enemy_units and self.townhalls:
              target = self.enemy_units.closest_to(random.choice(self.townhalls))
      elif model_choice == 2:
          # Attack enemy structure
          if self.enemy_structures:
              target = random.choice(self.enemy_structures)
      elif model_choice == 3:
          # Attack enemy start location
          target = self.enemy_start_locations[0] if self.enemy_start_locations else None

      # Order attack if a valid target is found
      if target:
          aggressive_units = {UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.HYDRALISK}
          for UNIT in aggressive_units:
              for unit in self.units(UNIT).idle:
                  unit.attack(target)

    async def expand(self):
        if self.supply_used >= 17 and self.can_afford(UnitTypeId.HATCHERY) and not self.already_pending(UnitTypeId.HATCHERY) and self.townhalls.amount < 2:
            await self.expand_now()
        elif self.supply_workers >= 25 and not self.already_pending(UnitTypeId.HATCHERY) and self.townhalls.amount < 3 and self.can_afford(UnitTypeId.HATCHERY):
            await self.expand_now()
        elif (self.supply_workers >= 37 and self.townhalls.amount < 4 and not self.already_pending(UnitTypeId.HATCHERY) and self.can_afford(UnitTypeId.HATCHERY)):
            await self.expand_now()
        elif self.supply_workers >= 55 and self.townhalls.amount < (self.time / 60) / 2 and self.can_afford(UnitTypeId.HATCHERY):
            await self.expand_now()

while True:
    run_game(maps.get("AbyssalReefAIE"), [
        Bot(Race.Zerg, DeetssBot(use_model=False)),
        Computer(Race.Protoss, Difficulty.Hard)
    ], realtime=False)