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
    print("iterations in file: ", iterations)

    cnt = 0
    for it in iterations:
        df = series.iterations[it].particles["beam"].to_df()
        print(f"{cnt}: iteration: {it}, nparticles: {len(df)}")
    series.close()
    return

if __name__ == "__main__":
    main()
    pass
