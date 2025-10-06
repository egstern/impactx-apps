#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
import PyNAFF as pnf

# Read monitor data from openPMD file that is assumed to the output of
# a monitor element for an ImpacTX run with 51 x test particles and 50
# y test particles.

NX=51 # number of x test particles
NY=51 # number of y test particles

def main(mfile):
    series = io.Series(mfile, io.Access.read_only)
    iters = list(series.iterations)
    niters = len(iters)

    xdata = np.zeros((NX, niters))
    ydata = np.zeros((NY, niters))

    # loop over iterations
    cnt = 0
    for iter in iters:
        print('iteration: ', iter)
        iterdata = series.iterations[iter].particles["beam"].to_df()
        xdata[:, cnt] = iterdata['position_x'][:NX]
        ydata[:, cnt] = iterdata['position_y'][51:51+NY]
        cnt = cnt + 1
        #print('iterdata.shape: ', iterdata.shape)
        #print('position x data [:10]: ', iterdata['position_x'][:10])
        del iterdata
        pass

    np.save('xdata.npy', xdata)
    np.save('ydata.npy', ydata)
    sys.exit(0)

    xtunes = np.zeros(NX)
    ytunes = np.zeros(NY)
    for i in range(NX):
        print(f'calculating x tune {i}')
        xtunes[i] = pnf.naff(xdata[i, :], turns=niters, nterms=1)[0][1]
        pass
    for i in range(NY):
        print(f'calculating y tune {i}')
        ytunes[i] = pnf.naff(ydata[i, :], turns=niters, nterms=1)[0][1]
        pass

    plt.figure()
    plt.plot(xtunes, '*', label='x tunes')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(ytunes, '*', label='y tunes')
    plt.legend(loc='best')
        
    pass

if __name__ == "__main__":
    main(sys.argv[1])
    pass
