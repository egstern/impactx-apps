#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpmd_api as io
import pyNAFF as pnf

datafile = "diags/openPMD/monitor.bp5"

# these lattice functions are calculated with Synergia during
# construction of the booster-asymm lattice
alpha_x = -0.03374203811363915
beta_x = 33.432491467233
alpha_y = 0.011403474422954477
beta_y = 5.2633267186119905

def read_series(opmdfile):
    series = io.Series(opmdfile, io.Access.read_only)
    iters = list(series.iterations)
    # first get ids for the test particles from the first step
    beam0 = series.iterations[iters[0]].particles["beam"]
    df0 = beam0.to_df()
    # test particles have weights == 0
    # x test particles start with y==0
    # y test particles start with x==0
    xtestparticles = df0[ (df0['weighting'] == 0.0) &
                          (df0['position_y'] == 0.0) ]
    ytestparticles = df0[ (df0['weighting'] == 0.0) &
                          (df0['position_x'] == 0.0) ]
    x0 = xtestparticles['position_x'].to_numpy()
    y0 = ytestparticles['position_y'].to_numpy()
    print(len(xtestparticles), 'x test particles')
    print(x0)
    print(len(ytestparticles), 'y test particles')
    print(y0)

    x_ids = xtestparticles['id']
    y_ids = ytestparticles['id']

    del df0, beam0, xtestparticles, ytestparticles

    print('accumulating over turns')
    nturns = len(iters)
    npart = len(x_ids)
    xdata = np.zeros((nturns, npart))
    pxdata = np.zeros((nturns, npart))
    ydata = np.zeros((nturns, npart))
    pydata = np.zeros((nturns, npart))
    for turn in range(len(iters)):
        print('Turn ', turn)
        beam = series.iterations[iters[turn]].particles["beam"]
        df = beam.to_df()
        xtestparticles = df[ (df['weighting'] == 0.0) &
                             (df['id'].isin(x_ids)) ]
        ytestparticles = df[ (df['weighting'] == 0.0) &
                             (df['id'].isin(y_ids)) ]
        
        xdata[turn, :] = xtestparticles['position_x']
        pxdata[turn, :] = xtestparticles['momentum_x']
        ydata[turn, :] = ytestparticles['position_y']
        pydata[turn, :] = ytestparticles['momentum_y']

    return xdata, pxdata, ydata, pydata

def main():
    xdata, pxdata, ydata, pydata = read_series(datafile)
    print("xdata.shape: ", xdata.shape)
    print("ydata.shape: ", ydata.shape)

    sqrtbetax = np.sqrt(beta_x)
    sqrtbetay = np.sqrt(beta_y)
    zx = complex(xdata/sqrtbetax, -(pxdata*sqrtbetax + alpha_x*xdata/sqrtbetax))
    zy = complex(ydata/sqrtbetay, -(pydata*sqrtbetay + alpha_y*ydata/sqrtbetay))

    xtunes = np.zeros(npart)
    ytunes = np.zeros(npart)

    for p in range(npart):
        print('getting x spectrum for particle ', p)
        out = pnf.naff(zx[:, p],
                       turns=zx.shape[0]-1,
                       nterms=1,
                       skipTurns=0,
                       getFullSpectrum=True,
                       window=1)
        if len(out) == 0:
            print("failed to find spectrum for particle ", p)
        else:
            print("tune: ", out[0, 1])
            xtunes[p] = out[0, 1]

        print('getting y spectrum for particle ', p)
        out = pnf.naff(zy[:, p],
                       turns=zy.shape[0]-1,
                       nterms=1,
                       skipTurns=0,
                       getFullSpectrum=True,
                       window=1)
        if len(out) == 0:
            print("failed to find spectrum for particle ", p)
        else:
            print("tune: ", out[0, 1])
            ytunes[p] = out[0, 1]

    return

if __name__ == "__main__":
    main()


    
