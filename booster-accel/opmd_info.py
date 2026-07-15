#!/usr/bin/env python
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpmd_api as io

def main():
    print('file: ', sys.argv[1])
    series = io.Series(sys.argv[1], io.Access.read_only)
    iterations = list(series.iterations)
    print(f"number of iterations: {len(iterations)}")
    iter0 = iterations[0]
    beam0 = series.iterations[iter0].particles["beam"]
    charge_ref = beam0.get_attribute("charge_ref")
    charge_C = beam0.get_attribute("charge_C")
    df = beam0.to_df()
    Npart = len(df)
    w = beam0.to_df()["weighting"][0]
    print("particles in first iteration: ", Npart)
    print("particles from weights: ", round((charge_C/charge_ref)/w))

    assert Npart == round((charge_C/charge_ref)/w)

    for i, iter in enumerate(iterations):
        beam = series.iterations[iter].particles["beam"]
        cC = beam.get_attribute("charge_C")
        s_i = beam.get_attribute("s_ref")
        npart_i = round((cC/charge_ref)/w)
        print(i, s_i, npart_i)

    series.close()
    return

if __name__ == "__main__":
    main()
    pass
