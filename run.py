import random
import cv2
import numpy as np
import time
from collections import OrderedDict
import multiprocessing

import sc2
from sc2 import run_game, maps, Race, Difficulty, Result, position
from sc2.player import Bot, Computer
from sc2.constants import *
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2, Point3
from sc2.data import race_gas, race_worker, race_townhalls, ActionResult, Attribute, Race

HEADLESS = False

USE_MODEL = True

class DeetssBot(sc2.BotAI):
    def __init__(self):
        self.buildStuffInverval = 2
        self.MAX_WORKERS = 55
        self.do_something_after = 0
        self.exactExpansionLocations = []
        
        self.train_data = []
        self.use_model = USE_MODEL

        if self.use_model:
            import tensorflow as tf

        try:
            if self.use_model:       
                print("\n\n\nUSING MODEL!!\n\n\n")
                self.model = tf.keras.models.load_model(
                    "1x512-CNN.model")
        except OSError:
            print(OSError)
            print("\nMaybe you need to train a model first")

    async def on_step(self, iteration):
        # do this every step
        self.currentDroneCountIncludingPending = self.units(DRONE).amount + self.already_pending(
            DRONE) + self.units(EXTRACTOR).ready.filter(lambda x: x.vespene_contents > 0).amount

        if iteration == 0:
            #print(self._game_info.map_ramps)
            self.units(OVERLORD).random.move(self.enemy_start_locations[0]) # send out first ovey
            await self.chat_send("(glhf)")
            await self.findExactExpansionLocations()

        
        await self.scout()
        await self.distribute_workers()

        if iteration % self.buildStuffInverval == 0:
            # If supply is low, train overlords
            await self.build_overseers()
            # If we need more workers build some
            await self.build_workers()
            # expand
            await self.expand()
            # Gas 
            await self.build_extractors()
            # Offensive building
            await self.offensive_buildings()
            #await self.distribute_overlords()
            # Research Upgrades
            await self.research_upgrades()

            # Make descisions
            await self.intel()
            # Manage army
            await self.attack()
            # Queen micro
            await self.queen_micro()

        if iteration % 5 == 0: #This keeps the bot from spending all its money on units
            # Build offensive units
            await self.build_army()


    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * self.game_info.map_size[0]
        y += ((random.randrange(-20, 20))/100) * self.game_info.map_size[1]

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
        if len(self.units(OVERLORD)) > 0 and self.supply_cap > 100:
            scout = self.units(OVERLORD)[0]
            if scout.is_idle:
                enemy_location = random.choice(self.exactExpansionLocations)
                move_to = self.random_location_variance(enemy_location)
                
                scout.move(move_to)

    async def findExactExpansionLocations(self):
        # execute this on start, finds all expansions where creep tumors should not be build near
        self.exactExpansionLocations = []
        for loc in self.expansion_locations.keys():
            # TODO: change mindistancetoresource so that a hatch still has room to be built
            self.exactExpansionLocations.append(await self.find_placement(HATCHERY, loc, placement_step=1))

    async def find_target(self, state):
        if len(self.enemy_units.visible) > 0:
            return random.choice(self.enemy_units.visible)
        elif len(self.enemy_structures.visible) > 0:
            return random.choice(self.enemy_structures.visible)
        else:
            return self.enemy_start_locations[0]
        
    async def build_workers(self):
        if self.supply_workers < self.structures(HATCHERY).amount * 16 and self.larva and self.can_afford(DRONE) and self.minerals < 1000:
                self.larva.random.train(DRONE)

    async def build_overseers(self):
        if self.supply_left <= 2 and self.larva and self.can_afford(OVERLORD) and not self.supply_cap == 200 and self.already_pending(OVERLORD) < 2:
            self.larva.random.train(OVERLORD)

    async def expand(self):
        try:
            if self.supply_used >= 17 and self.can_afford(HATCHERY) and not self.already_pending(HATCHERY) and self.townhalls.amount < 2:
                await self.expand_now()
            elif self.supply_workers >= 25 and not self.already_pending(HATCHERY) and self.townhalls.amount < 3 and self.can_afford(HATCHERY):
                await self.expand_now()
            elif (self.supply_workers >= 37 and self.townhalls.amount < 4 and not self.already_pending(HATCHERY) and self.can_afford(HATCHERY)):
                await self.expand_now()
            elif self.supply_workers >= 55 and self.townhalls.amount < (self.time / 60) / 2 and self.can_afford(HATCHERY):
                await self.expand_now()
        except Exception as e:
            print(str(e))

    async def build_extractors(self):
        if (self.townhalls.amount >= 2 and not self.already_pending(EXTRACTOR) and self.gas_buildings.amount < 1 and self.structures(SPAWNINGPOOL).exists) or (self.time > 270 and self.currentDroneCountIncludingPending > 35 and self.structures(EXTRACTOR).amount < self.townhalls.amount * 1.5):
            worker = self.select_build_worker(self.townhalls.first)
            if self.can_afford(EXTRACTOR):
                worker.build_gas(
                    self.vespene_geyser.closest_to(worker))
    #async def distribute_overlords(self):

    async def offensive_buildings(self):
        # Spawning pool @ 17
        if self.currentDroneCountIncludingPending >= 17 and not self.structures(SPAWNINGPOOL).exists and self.already_pending(SPAWNINGPOOL) < 1 and self.townhalls.amount >= 2:
            if self.can_afford(SPAWNINGPOOL):
                #pos = await self.find_placement(SPAWNINGPOOL, townhallLocationFurthestFromOpponent, min_distance=6)
                pos = await self.find_placement(SPAWNINGPOOL, self.townhalls.ready.random.position)
                if pos is not None:
                    drone = self.workers.closest_to(pos)
                    if self.can_afford(SPAWNINGPOOL):
                        print("Building Pool now! @ " + self.time_formatted)
                        drone.build(SPAWNINGPOOL, pos)

        # warren if spawing pool
        if self.structures(SPAWNINGPOOL).ready and self.can_afford(ROACHWARREN) and not self.structures(ROACHWARREN).exists and not self.already_pending(ROACHWARREN):
            pos = await self.find_placement(ROACHWARREN, self.townhalls.ready.random.position)
            if pos is not None:
                print("Incoming Roach Warren! @ " + self.time_formatted)
                drone = self.workers.closest_to(pos)
                drone.build(ROACHWARREN, pos)

        # start some evo chambers if pool is done and warren started
        if self.supply_workers > 20 and self.can_afford(EVOLUTIONCHAMBER) and not self.already_pending(EVOLUTIONCHAMBER) + self.structures(EVOLUTIONCHAMBER).amount >= 2 and self.structures(ROACHWARREN).amount == 1:
            if self.can_afford(EVOLUTIONCHAMBER):
                #pos = await self.find_placement(SPAWNINGPOOL, townhallLocationFurthestFromOpponent, min_distance=6)
                pos = await self.find_placement(EVOLUTIONCHAMBER, self.townhalls.ready.random.position)
                if pos is not None:
                    drone = self.workers.closest_to(pos)
                    if self.can_afford(EVOLUTIONCHAMBER):
                        print("Building evo chamber now! @ " + self.time_formatted)
                        drone.build(EVOLUTIONCHAMBER, pos)
        
        # start hydra den if lair
        if self.structures(LAIR).ready.amount > 0 and not self.structures(HYDRALISKDEN).exists and not self.already_pending(HYDRALISKDEN):
            if self.can_afford(HYDRALISKDEN):
                #pos = await self.find_placement(SPAWNINGPOOL, townhallLocationFurthestFromOpponent, min_distance=6)
                pos = await self.find_placement(HYDRALISKDEN, self.townhalls.ready.random.position.to2)
                if pos is not None:
                    drone = self.workers.closest_to(pos)
                    if self.can_afford(HYDRALISKDEN) and drone:
                        print("Building hydra den now! @ " +
                              self.time_formatted)
                        drone.build(HYDRALISKDEN, pos)


    async def research_upgrades(self):
        # speedlings at 100 gas TODO: speed this waaaaaaaaaay up. i think it may be the way workers are spread early
        if self.structures(SPAWNINGPOOL).ready and self.vespene >= 100:
            self.structures(SPAWNINGPOOL).first.research(
                UpgradeId.ZERGLINGMOVEMENTSPEED)

        # start upgrades if evos are done
        if self.structures(EVOLUTIONCHAMBER).ready.idle.exists:
            if self.structures(EVOLUTIONCHAMBER).ready.idle:
                evo = self.structures(EVOLUTIONCHAMBER).ready.idle.random
                abilities = await self.get_available_abilities(evo)
                targetAbilities = [AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL1, AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL2, AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL3,
                                    AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL1, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL2, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL3]
                for ability in targetAbilities:
                    if self.can_afford(ability) and ability in abilities:
                            print("Researching " + ability.name)
                            evo(ability)

        # lair @ 35 workers
        if self.currentDroneCountIncludingPending > 20 and self.structures(LAIR).amount <= 1 and self.can_afford(LAIR) and self.townhalls.ready.idle.amount > 0 and not self.already_pending(LAIR):
            print("Upgrading to Lair @ " + self.time_formatted)
            self.townhalls.first(UPGRADETOLAIR_LAIR)

        # start upgrades if den are done
        if self.structures(HYDRALISKDEN).ready.idle.exists:
            if self.structures(HYDRALISKDEN).ready.idle:
                den = self.structures(HYDRALISKDEN).ready.idle.random
                abilities = await self.get_available_abilities(den)
                targetAbilities = [AbilityId.RESEARCH_GROOVEDSPINES, AbilityId.RESEARCH_MUSCULARAUGMENTS]
                for ability in targetAbilities:
                    if ability in abilities:
                        if self.can_afford(ability) and ability in abilities:
                            print("Researching " + ability.name)
                            den(ability)

    async def build_army(self):
        if self.larva:
            # make some hydras if den is ready
            if self.structures(HYDRALISKDEN).ready and self.can_afford(HYDRALISK):
                self.larva.random.train(HYDRALISK)
            
            # make some roaches if warren is ready
            if self.structures(ROACHWARREN).ready and self.can_afford(ROACH):
                if not self.units(HYDRALISK).idle:
                    self.larva.random.train(ROACH)
                elif self.units(ROACH).amount < self.units(HYDRALISK).amount * 0.75:
                    self.larva.random.train(ROACH)

            # spawn some zerglings if spawning pool is ready
            if self.structures(SPAWNINGPOOL).ready and self.can_afford(ZERGLING):
                if not self.units(ROACH).idle:
                    self.larva.random.train(ZERGLING)
                elif self.units(ZERGLING).amount < self.units(ROACH).amount / 2:
                    self.larva.random.train(ZERGLING)

        # queens @ pool
        if self.structures(SPAWNINGPOOL).ready and (self.units(QUEEN).amount + self.already_pending(QUEEN) < self.townhalls.ready.idle.amount * self.time / 90) and not self.units(QUEEN).amount >= 15 and self.can_afford(QUEEN):
            hatcheries = self.townhalls.ready.idle
            for hatch in hatcheries:
                hatch.train(QUEEN)
                print("Queen incoming!")


    async def intel(self):
        struct_dict = {
            HATCHERY: [4, (0, 255, 0)],
            LAIR: [4, (0, 255, 0)],
            HIVE: [4, (0, 255, 0)],
            EXTRACTOR: [2, (55, 200, 0)],
            SPAWNINGPOOL: [2, (200, 100, 0)],
            EVOLUTIONCHAMBER: [2, (150, 150, 0)],
            ROACHWARREN: [2, (255, 0, 0)],
            HYDRALISKDEN: [2, (255, 100, 0)],
        }

        unit_dict = {
            OVERLORD: [2, (20, 235, 0)],
            DRONE: [1, (55, 200, 0)],
            HYDRALISK: [1, (255, 100, 10)],
            ROACH: [1, (255, 10, 10)],
            ZERGLING: [1, (255, 0, 0)],
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

        main_base_names = ["nexus", "commandcenter", "hatchery", "lair", "hive"]
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

        if not self.supply_cap-self.supply_left == 0:
            military_weight = len(self.units({HYDRALISK, ROACH, ZERGLING})) / (self.supply_cap-self.supply_left)
        else:
            military_weight = 0

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

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    async def attack(self):
        # {UNIT: [n to fight, n to defend]}
        aggressive_units = {ZERGLING: [16, 5],
                            ROACH: [20, 6],
                            HYDRALISK: [10, 6]}
        
        for UNIT in aggressive_units:
            if len(self.units(UNIT).idle) > 0:
                target = False
                if self.time > self.do_something_after:
                    if self.use_model:
                        prediction = self.model.predict([self.flipped.reshape(-1,176,200,3)])
                        choice = np.argmax(prediction[0])

                        choice_dict = {0: "No Attack!",
                                    1: "Attack close to our Hatch!",
                                    2: "Attack Enemy Struture!",
                                    3: "Attack Enemy Start!"}
                        
                        print("Choice #{}:{}".format(choice, choice_dict[choice]))
                    else:
                        choice = random.randrange(0, 4)

                    
                    if choice == 0:
                        # no attack
                        wait = random.randrange(20, 165)
                        self.do_something_after = self.time  + wait

                    elif choice == 1:
                        #attack_unit_closest_nexus
                        if len(self.enemy_units) > 0:
                            target = self.enemy_units.closest_to(
                                random.choice(self.townhalls))

                    elif choice == 2:
                        #attack enemy structures
                        if len(self.enemy_structures) > 0:
                            target = random.choice(self.enemy_structures)

                    elif choice == 3:
                        #attack_enemy_start
                        
                        target = self.enemy_start_locations[0]
            
                    if target:
                        for unit in self.units(UNIT).idle:
                            unit.attack(target)
                        # if self.units(UNIT).ready.amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                        #     for s in self.units(UNIT).idle:
                        #         s.attack(await self.find_target(self.state))

                        # elif self.units(UNIT).ready.amount > aggressive_units[UNIT][1]:
                        #     if len(self.enemy_units) > 0:
                        #         for s in self.units(UNIT).idle:
                        #             s.attack(random.choice(self.enemy_units))
                    y = np.zeros(4)
                    y[choice] = 1
                    #print(choice)
                    self.train_data.append([y, self.flipped])
        

    async def queen_micro(self):
        # inject hatch if queen has energy
        if self.units(QUEEN).amount > 0:
            for hatch in self.townhalls:
                queen = self.units(QUEEN).closest_to(hatch.position)
                if queen.energy >= 25 and queen.is_idle and not hatch.has_buff(QUEENSPAWNLARVATIMER):
                    queen(EFFECT_INJECTLARVA, hatch)
                # elif queen.energy >= 25 and queen.is_idle and hatch.has_buff(QUEENSPAWNLARVATIMER):
                #     pos = await self.find_placement(UnitTypeId.CREEPTUMOR, queen.position)
                #     queen(BUILD_CREEPTUMOR_QUEEN, pos)
    
    def on_end(self, game_result):
        print("Game and ended")
        print(game_result)

        with open("log.txt", "a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))


def start_bot():
        run_game(maps.get("Abyssal Reef LE"), [
            Bot(Race.Zerg, DeetssBot()),
            Computer(Race.Random, Difficulty.Hard)
        ], realtime=False)

if __name__ == '__main__':
    while True:
        p = multiprocessing.Process(target=start_bot)
        p.start()
        p.join()
