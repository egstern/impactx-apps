#!/usr/bin/env python3

import sys, os
import numpy as np
import pandas as pd
import openpmd_api as io
import pytest

def main(file1, file2, turn1, turn2):
    ser1 = io.Series(sys.argv[1], io.Access.read_only)
    ser2 = io.Series(sys.argv[2], io.Access.read_only)
    iter1 = list(ser1.iterations)[turn1]
    iter2 = list(ser2.iterations)[turn2]
    beam1 = ser1.iterations[iter1].particles["beam"]
    beam2 = ser2.iterations[iter2].particles["beam"]

    charge_ref1 = beam1.get_attribute('charge_ref')
    charge_ref2 = beam2.get_attribute('charge_ref')
    if charge_ref1 != pytest.approx(charge_ref2, rel=5.0e-15):
        print("charge_ref differences: ", charge_ref1, charge_ref2)
    print("charge_ref: ", charge_ref1)

    bchg1 = beam1.get_attribute("charge_C")
    bchg2 = beam1.get_attribute("charge_C")
    if bchg1 != pytest.approx(bchg2, rel=5.0e-15):
        print("bunch charge differences: ", bchg1, bchg2)
    print("bunch chargeC: ", bchg1)

    mass1 = beam1.get_attribute("mass_ref")
    mass2 = beam1.get_attribute("mass_ref")
    if mass1 != pytest.approx(mass2, rel=5.0e-15):
        print("mass differences: ", mass1, mass2)
    print("mass: ", mass1)

    bg1 = beam1.get_attribute("beta_gamma_ref")
    bg2 = beam2.get_attribute("beta_gamma_ref")
    if bg1 != pytest.approx(bg2, rel=5.0e-15):
        print("beta gamma differences: ", bg1, bg2)
    print("beta*gamma: ", bg1)

    df1 = beam1.to_df()
    df2 = beam2.to_df()

    if len(df1) != len(df2):
        print("number of particles does not match: ", len(df1), len(df2))
    print("number particles: ", len(df1))

    coords = [ "position_x", "position_y", "position_t",
               "momentum_x", "momentum_y", "momentum_t"]

    diffs = {}
    for c in coords:
        cstd = (df1[c] - df2[c]).std()
        diffs[c] = cstd
    
    print("coord diff stds:")
    for c in coords:
        print(f"{c}:\t\t{diffs[c]}")
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    pass
