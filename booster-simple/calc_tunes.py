#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
import PyNAFF as pnf

# read xdata.npy and ydata.npy, calculate tunes


def main():

    xdata = np.load('xdata.npy')
    ydata = np.load('ydata.npy')
    
    NX = xdata.shape[0]
    NY = ydata.shape[0]
    niters = 2048
    
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

    plt.show()

    pass

if __name__ == "__main__":
    main()
    pass
