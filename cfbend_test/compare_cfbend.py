#!/usr/bin/env python
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pytest

def read_txtone():
    turn0 = np.loadtxt('cfbend.txtone', skiprows=9, usecols=(2,3,4,5,6,7), max_rows=24)
    uturn0 = np.ndarray((1, 24, 6))
    uturn0[0, :, :] = turn0

    print('shape uturn0: ', uturn0.shape)
    turn1 = np.loadtxt('cfbend.txtone', skiprows=34, usecols=(2,3,4,5,6,7), max_rows=24)
    uturn1 = np.ndarray((1, 24, 6))
    uturn1[0, :, :] = turn1
    print('shape uturn1: ', uturn1.shape)
    c = np.vstack((uturn0, uturn1))
    print('c.shape: ', c.shape)
    return c

def read_synh5():
    h5 = h5py.File('cftracks.h5','r')
    trks = h5.get('track_coords')[()]
    h5.close()
    return trks

def main():
    trks_madx = read_txtone()
    trks_syn = read_synh5()

    # Check transverse initial coords
    for i in range(24):
        for j in range(4):
            print(i, j, trks_madx[0, i, j], trks_syn[0, i, j], flush=True)
            assert trks_madx[0, i, j] == pytest.approx(trks_syn[0, i, j])

    # check transverse final coords
    for i in range(24):
        for j in range(4):
            print(i, j, trks_madx[1, i, j], trks_syn[1, i, j], flush=True)
            assert trks_madx[1, i, j] == pytest.approx(trks_syn[1, i, j], rel=1.2e-4)


    pass

if __name__ == "__main__":
    main()
    pass
