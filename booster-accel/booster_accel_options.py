#!/usr/bin/env pythone

import argparse

class Options:
    turns = 15071
    final_energy = 8000.0 # MeV
    generate_bunch = False

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", help="The number of turns for the simulation",
                    type=int)
    args = parser.parse_args()
    print("args.turns: ", args.turns)
    if args.turns:
        Options.turns = parser.turns

if __name__ == "__main__":
    parse_options()
    for opt in dir(Options):
        if len(opt) < 2 or opt[0:2] == "__":
            continue
        print("option ", opt, getattr(Options, opt))

