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
    def __init__(self):
        super().__init__()
        self.model = load_model("BasicCNN-Final-10-epochs-0.001-LR.keras")
        self.buildStuffInverval = 2
        self.ITERATIONS_PER_MINUTE = 200
        self.MAX_WORKERS = 55
        self.last_known_enemy_count = 0
        self.train_data = []
        self.do_something_after = 0
        self.exactExpansionLocations = []
        self.townhalls = None

    async def on_step(self, iteration):
        larvae: Units = self.larva
        forces: Units = self.units.of_type({UnitTypeId.ZERGLING, UnitTypeId.HYDRALISK})

        self.iteration = iteration

        self.currentDroneCountIncludingPending = self.workers.amount + self.already_pending(
            UnitTypeId.DRONE) + self.units(UnitTypeId.EXTRACTOR).ready.filter(lambda x: x.vespene_contents > 0).amount

        if iteration == 0:
            print(self._game_info.map_ramps)
            if self.enemy_start_locations[0]:
                self.units(UnitTypeId.OVERLORD).random.move(self.enemy_start_locations[0]) # send out first ovey

            await self.chat_send("(glhf)")
            await self.findExactExpansionLocations()
        
        # await self.scout()
        # await self.distribute_workers()

        # if iteration % self.buildStuffInverval == 0:
        #     # If supply is low, train overlords
        #     await self.build_overseers()
        #     # If we need more workers build some
        #     await self.build_workers()
        #     # expand
        #     await self.expand()
        #     # Gas 
        #     await self.build_extractors()
        #     # Offensive building
        #     await self.offensive_buildings()
        #     #await self.distribute_overlords()
        #     # Research Upgrades
        #     await self.research_upgrades()

        #     # Make descisions
        #     await self.intel()
        #     # Manage army
        #     await self.attack()
        #     # Queen micro
        #     await self.queen_micro()

        # if iteration % 5 == 0: #This keeps the bot from spending all its money on units
        #     # Build offensive units
        #     await self.build_army()

        if iteration % 5 == 0:
            await self.intel()
            self.log_state(iteration)  # log the intel state image for inspection

        # Build a complete state representation
        state = self.get_state_representation()
        predictions = self.model.predict(state)[0]
        action = np.argmax(predictions)
        print(f"Iteration {iteration}, Predictions: {predictions}, Chosen action: {action}")

        await self.handle_action(action)

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
        if self.supply_workers < self.structures(UnitTypeId.HATCHERY).amount * 16:
            await self.build_workers()
        elif self.supply_left <= 2 and self.can_afford(UnitTypeId.OVERLORD):
            self.larva.random.train(UnitTypeId.OVERLORD)
        elif self.can_afford(UnitTypeId.HATCHERY):
            await self.expand()

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

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    async def scout(self):
        if len(self.units(UnitTypeId.OVERLORD)) > 0 and self.supply_cap > 100:
            scout = self.units(UnitTypeId.OVERLORD)[0]
            if scout.is_idle:
                enemy_location = random.choice(self.exactExpansionLocations)
                move_to = self.random_location_variance(enemy_location)
                
                scout.move(move_to)

    async def findExactExpansionLocations(self):
        # execute this on start, finds all expansions where creep tumors should not be build near
        self.exactExpansionLocations = []
        for loc in self.expansion_locations.keys():
            # TODO: change mindistancetoresource so that a hatch still has room to be built
            self.exactExpansionLocations.append(await self.find_placement(UnitTypeId.HATCHERY, loc, placement_step=1))

    async def find_target(self, state):
        if len(self.enemy_units.visible) > 0:
            return random.choice(self.enemy_units.visible)
        elif len(self.enemy_structures.visible) > 0:
            return random.choice(self.enemy_structures.visible)
        else:
            return self.enemy_start_locations[0]
        
    async def build_workers(self):
        if self.supply_workers < self.structures(UnitTypeId.HATCHERY).amount * 16 and self.larva and self.can_afford(UnitTypeId.DRONE) and self.minerals < 1000:
                self.larva.random.train(UnitTypeId.DRONE)

    async def build_overseers(self):
        if self.supply_left <= 2 and self.larva and self.can_afford(UnitTypeId.OVERLORD) and not self.already_pending(UnitTypeId.OVERLORD):
            self.larva.random.train(UnitTypeId.OVERLORD)

    async def expand(self):
        if self.supply_used >= 17 and self.can_afford(UnitTypeId.HATCHERY) and not self.already_pending(UnitTypeId.HATCHERY) and self.townhalls.amount < 2:
            await self.expand_now()
        elif self.supply_workers >= 25 and not self.already_pending(UnitTypeId.HATCHERY) and self.townhalls.amount < 3 and self.can_afford(UnitTypeId.HATCHERY):
            await self.expand_now()
        elif (self.supply_workers >= 37 and self.townhalls.amount < 4 and not self.already_pending(UnitTypeId.HATCHERY) and self.can_afford(UnitTypeId.HATCHERY)):
            await self.expand_now()
        elif self.supply_workers >= 55 and self.townhalls.amount < (self.time / 60) / 2 and self.can_afford(UnitTypeId.HATCHERY):
            await self.expand_now()

    async def build_extractors(self):
        if (self.townhalls.amount >= 2 and not self.already_pending(UnitTypeId.EXTRACTOR) and self.gas_buildings.amount < 1) or (self.time > 270 and self.currentDroneCountIncludingPending > 35 and self.structures(UnitTypeId.EXTRACTOR).amount < self.townhalls.amount * 1.5):
            worker = self.select_build_worker(self.townhalls.first)
            if self.can_afford(UnitTypeId.EXTRACTOR):
                worker.build_gas(
                    self.vespene_geyser.closest_to(worker))
    #async def distribute_overlords(self):

    async def offensive_buildings(self):
        # Spawning pool @ 17
        if self.currentDroneCountIncludingPending >= 17 and not self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.already_pending(UnitTypeId.SPAWNINGPOOL) < 1 and self.townhalls.amount >= 2:
            if self.can_afford(UnitTypeId.SPAWNINGPOOL):
                #pos = await self.find_placement(UnitTypeId.SPAWNINGPOOL, townhallLocationFurthestFromOpponent, min_distance=6)
                pos = await self.find_placement(UnitTypeId.SPAWNINGPOOL, self.townhalls.ready.random.position.to2)
                if pos is not None:
                    drone = self.workers.closest_to(pos)
                    if self.can_afford(UnitTypeId.SPAWNINGPOOL):
                        print("Building Pool now! @ " + self.time_formatted)
                        drone.build(UnitTypeId.SPAWNINGPOOL, pos)

        # warren if spawing pool
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.can_afford(UnitTypeId.ROACHWARREN) and not self.structures(UnitTypeId.ROACHWARREN).exists and self.already_pending(UnitTypeId.ROACHWARREN) < 1:
            pos = await self.find_placement(UnitTypeId.ROACHWARREN, self.townhalls.ready.random.position.to2)
            if pos is not None:
                print("Incoming Roach Warren! @ " + self.time_formatted)
                drone = self.workers.closest_to(pos)
                drone.build(UnitTypeId.ROACHWARREN, pos)

        # start some evo chambers if pool is done
        if self.supply_workers > 20 and self.can_afford(UnitTypeId.EVOLUTIONCHAMBER) and not self.already_pending(UnitTypeId.EVOLUTIONCHAMBER) + self.structures(UnitTypeId.EVOLUTIONCHAMBER).amount >= 2:
            if self.can_afford(UnitTypeId.EVOLUTIONCHAMBER):
                #pos = await self.find_placement(UnitTypeId.SPAWNINGPOOL, townhallLocationFurthestFromOpponent, min_distance=6)
                pos = await self.find_placement(UnitTypeId.EVOLUTIONCHAMBER, self.townhalls.ready.random.position.to2)
                if pos is not None:
                    drone = self.workers.closest_to(pos)
                    if self.can_afford(UnitTypeId.EVOLUTIONCHAMBER):
                        print("Building evo chamber now! @ " + self.time_formatted)
                        drone.build(UnitTypeId.EVOLUTIONCHAMBER, pos)
        
        # start hydra den if lair
        if self.structures(UnitTypeId.LAIR).ready.amount > 0 and not self.structures(UnitTypeId.HYDRALISKDEN).exists:
            if self.can_afford(UnitTypeId.HYDRALISKDEN):
                #pos = await self.find_placement(SPAWNINGPOOL, townhallLocationFurthestFromOpponent, min_distance=6)
                pos = await self.find_placement(UnitTypeId.HYDRALISKDEN, self.townhalls.ready.random.position.to2)
                if pos is not None:
                    drone = self.workers.closest_to(pos)
                    if self.can_afford(UnitTypeId.HYDRALISKDEN):
                        print("Building hydra den now! @ " +
                              self.time_formatted)
                        drone.build(UnitTypeId.HYDRALISKDEN, pos)


    async def research_upgrades(self):
        hq: Optional[Unit] = self.townhalls.first if self.townhalls else None

        # speedlings at 100 gas
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.vespene >= 100:
            self.structures(UnitTypeId.SPAWNINGPOOL).first.research(
                UpgradeId.ZERGLINGMOVEMENTSPEED)

        # start upgrades if evos are done
        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready.idle.exists:
            for evo in self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready.idle:
                abilities = await self.get_available_abilities(evo)
                targetAbilities = [AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL1, AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL2, AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL3,
                                    AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL1, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL2, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL3]
                for ability in targetAbilities:
                    if ability in abilities:
                        if self.can_afford(ability):
                            print("Researching " + ability.name)
                            evo(ability)

        # UnitTypeId.lair @ 35 workers
        if self.currentDroneCountIncludingPending > 20 and self.structures(UnitTypeId.LAIR).amount <= 1 and self.can_afford(UnitTypeId.LAIR) and self.townhalls.ready.idle.amount > 0:
            print("Upgrading to Lair @ " + self.time_formatted)
            if hq.is_idle and not self.townhalls(UnitTypeId.LAIR):
                if self.can_afford(UnitTypeId.LAIR):
                    hq.build(UnitTypeId.LAIR)

        # start upgrades if den are done
        if self.structures(UnitTypeId.HYDRALISKDEN).ready.idle.exists:
            for den in self.structures(UnitTypeId.HYDRALISKDEN).ready.idle:
                abilities = await self.get_available_abilities(den)
                targetAbilities = [AbilityId.RESEARCH_GROOVEDSPINES, AbilityId.RESEARCH_MUSCULARAUGMENTS]
                for ability in targetAbilities:
                    if ability in abilities:
                        if self.can_afford(ability):
                            print("Researching " + ability.name)
                            den(ability)

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

    async def attack(self):
        state_input = self.flipped.reshape((1, self.flipped.shape[0], self.flipped.shape[1], 3)) 
        predictions = self.model.predict(state_input)[0]
        model_choice = np.argmax(predictions)
        # Use model_choice instead of overriding with a random number
        target = None

        if model_choice == 0:
            # no attack: wait/do nothing
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

        # Order attack if valid target is found
        if target:
            aggressive_units = {UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.HYDRALISK}
            for UNIT in aggressive_units:
                for unit in self.units(UNIT).idle:
                    unit.attack(target)
        

    async def queen_micro(self):
        # inject hatch if queen has energy
        if self.units(UnitTypeId.QUEEN).amount > 0:
            for hatch in self.townhalls:
                queen = self.units(UnitTypeId.QUEEN).closest_to(hatch.position)
                if queen.energy >= 25 and queen.is_idle and not hatch.has_buff(BuffId.QUEENSPAWNLARVATIMER):
                    queen(AbilityId.EFFECT_INJECTLARVA, hatch)
                # elif queen.energy >= 25 and queen.is_idle and hatch.has_buff(QUEENSPAWNLARVATIMER):
                #     pos = await self.find_placement(UnitTypeId.CREEPTUMOR, queen.position)
                #     queen(BUILD_CREEPTUMOR_QUEEN, pos)
    
    def on_end(self, game_result):
        print("Game and ended")
        print(game_result)

        np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data, dtype=object))
            

while True:
    run_game(maps.get("AbyssalReefAIE"), [
        Bot(Race.Zerg, DeetssBot()),
        Computer(Race.Protoss, Difficulty.Hard)
       # Human(Race.Zerg, 'player', fullscreen=False)
    ], realtime=False)
