#!/usr/bin/env pythone

import argparse

class Options:
    turns = 15071
    injection_energy = 800.0 # MeV
    final_energy = 8000.0 # MeV
    generate_bunch = False
    particles_file = "pip-ii-injected-58k.h5" # openPMD file converted from Synergia
    full_booster_charge = 6.7e12 # full PIP-II intensity
    harmonic_number = 84
    full_buckets = 81 # 3 buckets are empty for injection and extraction

    # not really options but useful values to have around
    # as calculated by MAD-X. I especially need gamma_tr to know
    # when transition occurs.
    alfa_x = -1.298673960026007664e-02
    beta_x = 3.373645362843065243e01
    alfa_y = 6.089861210659328755e-03
    beta_y = 5.252517912567207681e00
    disp_x = 3.785167992
    disp_px = 0.001377568703
    gamma_tr = 5.449167323


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

