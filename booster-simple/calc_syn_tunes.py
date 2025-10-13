#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import openpmd_api as io
import PyNAFF as pnf

def main(h5file):

    print('Reading data from ', h5file)
    h5 = h5py.File(h5file, 'r')
    print('data shape: ', h5.get('track_coords').shape)

    xdata = h5.get('track_coords')[:, 0:51, 0].transpose()
    ydata = h5.get('track_coords')[:, 51:102, 2].transpose()

    npart = xdata.shape[0]
    nturns = xdata.shape[1]

    print('xdata: ', xdata.shape)
    print('ydata: ', ydata.shape)

    # set mean to 0 to eliminate the 0 frequency component

    xmean = xdata.mean(axis=1).reshape((npart, 1))
    ymean = ydata.mean(axis=1).reshape((npart, 1))
    print('xmean.shape: ', xmean.shape)
    print('ymean.shape: ', ymean.shape)
    xdata = xdata - xmean
    ydata = ydata - ymean
    print('new xdata means: ', xdata.mean(axis=1))
    print('new ydata means: ', ydata.mean(axis=1))

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
        print('mean value: ', ydata[i,:].mean())
        foo = ydata[0, :]
        print('foo.shape: ', foo.shape, ' mean foo: ', foo.mean())
        print('foo: ', foo)
        print('naff(foo): ', pnf.naff(foo, turns=niters, nterms=1))
        print(pnf.naff(ydata[i, :], turns=niters, nterms=1))
        ytunes[i] = pnf.naff(ydata[i, :], turns=niters, nterms=1)[0][1]
        pass

    print('x tunes: ', xtunes)
    print()
    print('y tunes: ', ytunes)
    print()

    plt.figure()
    plt.plot(xtunes, '*', label='x tunes')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(ytunes, '*', label='y tunes')
    plt.legend(loc='best')

    plt.show()

    pass

if __name__ == "__main__":
    main(sys.argv[1])
    pass
