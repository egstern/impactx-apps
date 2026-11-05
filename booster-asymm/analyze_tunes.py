#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpmd_api as io
from pyNAFF import PyNAFF as pnf

if len(sys.argv) < 2:
    # no arguments. don't use space charge
    SC_solver = None
elif len(sys.argv) == 2:
    # one argument, is the space charge solver to use
    SC_solver = sys.argv[1]

datafile = f"diags_{SC_solver}/test_particle_data.npy"

# these lattice functions are calculated with Synergia during
# construction of the booster-asymm lattice
alpha_x = -0.03374203811363915
beta_x = 33.432491467233
alpha_y = 0.011403474422954477
beta_y = 5.2633267186119905


def main():

    test_particle_data = np.load(datafile, allow_pickle=True)
    # turn = test_particle_data[:, 0]
    # idcpu = test_particle_data[:, 1]
    # xdata = test_particle_data[:, 2]
    # pxdata = test_particle_data[:, 3]
    # ydata = test_particle_data[:, 4]
    # pydata = test_particle_data[:, 5]

    df = pd.DataFrame(test_particle_data, columns=["turn", "idcpu", "x", "px", "y", "py", "t", "pt"])

    print("type(df): ", type(df))
    print("df.shape: ", df.shape)

    # Group by particle ids, without changing their relative order
    grouped_by_idcpu = df.groupby("idcpu", sort=True)
    particle_ids = df["idcpu"].unique().tolist()

    print("type(grouped_by_idcpu): ", type(grouped_by_idcpu), flush=True)
    
    # total number of particles is half x and half y
    nptot = len(particle_ids)
    nturns = len(df[df["idcpu"] == particle_ids[0]])

    print("number of particles: ", nptot)
    print("number of turns: ", nturns)

    npx = nptot//2
    npy = nptot-npx
    if npx != npy:
        raise RuntimeError("number of particles is not even number")

    sqrtbetax = np.sqrt(beta_x)
    sqrtbetay = np.sqrt(beta_y)

    plt.figure()
    plt.title("particle_ids")
    plt.plot(particle_ids)

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].set_title("x tracks")
    axs[1, 0].set_title("pt for x tracks")
    axs[0, 1].set_title("y tracks")
    axs[1, 1].set_title("pt for y tracks")
    axs[2, 0].set_title("t vs pt for x tracks")
    axs[2, 1].set_title("t vs pt for y tracks")

    # accumulate normalized coordinates for both x and y
    xtracks = np.ndarray((npx, nturns), dtype=np.complex128)
    ytracks = np.ndarray((npy, nturns), dtype=np.complex128)

    cnt = 0
    for idcpu, track in grouped_by_idcpu:
        xarr = track["x"].to_numpy()
        pxarr = track["px"].to_numpy()
        yarr = track["y"].to_numpy()
        pyarr = track["py"].to_numpy()
        tarr = track["t"].to_numpy()
        ptarr = track["pt"].to_numpy()

        # remove offset to eliminate 0 frequency component
        xarr = xarr - xarr.mean()
        pxarr = pxarr - pxarr.mean()
        yarr = yarr - yarr.mean()
        pyarr = pyarr - pyarr.mean()

        if cnt%5 == 0 and cnt < 25:
            axs[0, 0].plot(xarr, label=f"x track {cnt}", lw=2)
            axs[1, 0].plot(ptarr, label=f"pt track {cnt}", lw=2)
            axs[2, 0].plot(tarr, ptarr, '.', label=f"longitudinal orbit particle {cnt}")
        if cnt%5 == 0 and cnt >= 25:
            axs[0, 1].plot(yarr, label=f"y track {cnt}", lw=2)
            axs[1, 1].plot(ptarr, label=f"pt track {cnt}", lw=2)
            axs[2, 1].plot(tarr, ptarr, '.', label=f"longitudinal orbit particle {cnt}")

        if cnt < npx:
            xtracks[cnt, :] = xarr/sqrtbetax - 1j * (pxarr*sqrtbetax + alpha_x*xarr/sqrtbetax)
        if cnt >= npx:
            ytracks[cnt-npx, :] = yarr/sqrtbetay - 1j * (pyarr*sqrtbetay + alpha_y*yarr/sqrtbetay)

        cnt = cnt + 1

    for i in range(3):
        for j in range(2):
            axs[i, j].legend(loc='best')

    # print(xtracks[1, :20])

    f, axs = plt.subplots(2, 1)
    for p in range(0, npx, 5):
        axs[0].plot(xtracks[p, :].real, xtracks[p, :].imag, '.', label=f'particle {p}x')
    axs[0].set_title('x orbits')
    for p in range(0, npy, 5):
        axs[1].plot(ytracks[p, :].real, ytracks[p, :].imag, '.', label=f'particle {p}y')
    axs[1].set_title('y orbits')

    axs[0].legend(loc='best')
    axs[1].legend(loc='best')

    # plt.show()
    # sys.exit(0)

    xtunes = np.zeros(npx)
    ytunes = np.zeros(npy)

    for p in range(npx):
        print('getting x spectrum for particle ', p)
        out = pnf.naff(xtracks[p, :],
                       turns=xtracks.shape[1]-1,
                       nterms=1,
                       skipTurns=0,
                       getFullSpectrum=True,
                       window=1)
        if len(out) == 0:
            print("failed to find spectrum for particle ", p)
        else:
            tune = out[0, 1]
            if tune < 0:
                tune = tune + 1
            xtunes[p] = tune
            print("tune: ", tune)

        print('getting y spectrum for particle ', p)
        out = pnf.naff(ytracks[p, :],
                       turns=ytracks.shape[1]-1,
                       nterms=1,
                       skipTurns=0,
                       getFullSpectrum=True,
                       window=1)
        if len(out) == 0:
            print("failed to find spectrum for particle ", p)
        else:
            tune = out[0, 1]
            if tune < 0:
                tune = tune + 1
            ytunes[p] = tune
            print("tune: ", tune)

    np.save('xtunes.npy', xtunes)
    np.save('ytunes.npy', ytunes)

    f, axs = plt.subplots(2, 1)
    axs[0].plot(xtunes, 'o', label='x tunes')
    axs[0].legend(loc='best')
    axs[1].plot(ytunes, 'd', label='y tunes')
    axs[1].legend(loc='best')

    plt.show()

    return

if __name__ == "__main__":
    main()


    
