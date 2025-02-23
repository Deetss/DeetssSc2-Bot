#!/usr/bin/env python
import os
import sys
from absl import app, flags
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
from agent.observer import ObserverAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("replay", "", "Path to a replay file.")
flags.DEFINE_integer("player_id", 1, "ID of the observed player.")

def main(argv):
    if FLAGS.replay == "":
        sys.exit("You must provide a replay file via --replay")
    
    run_config = run_configs.get()
    if not os.path.exists(FLAGS.replay):
        sys.exit("Replay file '%s' doesn't exist." % FLAGS.replay)
    
    print("Training agent with replay:", FLAGS.replay)
    # Instantiate the agent (change map name if needed)
    agent = ObserverAgent(map_name="SimpleMap")
    
    # Invoke training from replay (this internally resets the agent,
    # steps through the replay, records transitions and performs learning)
    agent.learn_from_replay(FLAGS.replay, FLAGS.player_id)
    print("Training complete.")

if __name__ == "__main__":
    app.run(main)