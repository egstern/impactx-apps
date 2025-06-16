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
    niter = len(iterations)

    numpart = np.zeros(niter)
    for i in range(niter):
        it = iterations[i]
        beam = series.iterations[it].particles["beam"].to_df()
        numpart[i] = len(beam['position_x'])
        print(it, numpart[i])

    series.close()
    pass

if __name__ == "__main__":
    main()
    pass
