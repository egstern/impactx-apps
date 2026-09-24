#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpmd_api as io

def main(saveplt):
    series = io.Series('diags/openPMD/monitor.bp5', io.Access_Type.read_only)
    iterations = list(series.iterations)
    runstatus = pd.read_csv('runstatus.txt', delimiter='\s+')

    niter = len(iterations)
    turn = np.zeros(niter)
    s = np.zeros(niter)
    sig_x = np.zeros(niter)
    sig_px = np.zeros(niter)
    min_x = np.zeros(niter)
    max_x = np.zeros(niter)
    min_px = np.zeros(niter)
    max_px = np.zeros(niter)
    sig_y = np.zeros(niter)
    sig_py = np.zeros(niter)
    min_y = np.zeros(niter)
    max_y = np.zeros(niter)
    min_py = np.zeros(niter)
    max_py = np.zeros(niter)
    sig_t = np.zeros(niter)
    min_t = np.zeros(niter)
    max_t = np.zeros(niter)
    sig_pt = np.zeros(niter)
    min_pt = np.zeros(niter)
    max_pt = np.zeros(niter)
    mean_x = np.zeros(niter)
    mean_px = np.zeros(niter)
    mean_y = np.zeros(niter)
    mean_py = np.zeros(niter)
    mean_t = np.zeros(niter)
    mean_pt = np.zeros(niter)
    print('Reading statistics from monitor file')
    for i, iter in enumerate(iterations):
        print('iteration ', i, '->', iter)
        beam = series.iterations[iter].particles["beam"]
        s[i] = beam.get_attribute('s_ref')
        sig_x[i] = beam.get_attribute('sig_x')
        min_x[i] = beam.get_attribute('min_x')
        max_x[i] = beam.get_attribute('max_x')
        min_px[i] = beam.get_attribute('min_px')
        max_px[i] = beam.get_attribute('max_px')
        sig_y[i] = beam.get_attribute('sig_y')
        min_y[i] = beam.get_attribute('min_y')
        max_y[i] = beam.get_attribute('max_y')
        min_py[i] = beam.get_attribute('min_py')
        max_py[i] = beam.get_attribute('max_py')
        sig_t[i] = beam.get_attribute('sig_t')
        min_t[i] = beam.get_attribute('min_t')
        max_t[i] = beam.get_attribute('max_t')
        min_pt[i] = beam.get_attribute('min_pt')
        max_pt[i] = beam.get_attribute('max_pt')
        sig_px[i] = beam.get_attribute('sig_px')
        sig_py[i] = beam.get_attribute('sig_py')
        sig_pt[i] = beam.get_attribute('sig_pt')
        mean_x[i] = beam.get_attribute('mean_x')
        mean_px[i] = beam.get_attribute('mean_px')
        mean_y[i] = beam.get_attribute('mean_y')
        mean_py[i] = beam.get_attribute('mean_py')
        mean_t[i] = beam.get_attribute('mean_t')
        mean_pt[i] = beam.get_attribute('mean_pt')
        del beam
        

    # mean x,y
    f,ax = plt.subplots(2, 2)
    ax[0, 0].plot(1000.0 * mean_x, label='mean_x [mm]')
    ax[0, 0].legend(loc='best')
    ax[0, 0].grid(True)

    ax[0, 1].plot(1.0e6*mean_px, label='mean_px [mm-mr]')
    ax[0, 1].legend(loc='best')
    ax[0, 1].grid('True')
    
    ax[1, 0].plot(1000.0*mean_y, label='mean_y [mm]')
    ax[1, 0].legend(loc='best')
    ax[1, 0].grid(True)
    
    ax[1, 1].plot (1.0e6 * mean_py, label='mean_py [mm-mr]')
    ax[1, 1].legend(loc='best')
    ax[1, 1].grid(True)
    ax[1, 0].set_xlabel('turn')
    ax[1, 1].set_xlabel('turn')
    if saveplt: plt.savefig('mean_x-px-y-py.png')

    # sig x, y
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(1000.0 * sig_x, label='sig_x [mm]')
    ax[0, 0].legend(loc='best')
    ax[0, 0].grid(True)
    
    ax[0, 1].plot(1.0e6 * sig_px, label='sig_px [mm-mr]')
    ax[0, 1].legend(loc='best')
    ax[0, 1].grid(True)
    
    ax[1, 0].plot(1000.0 * sig_y, label='sig_y [mm]')
    ax[1, 0].legend(loc='best')
    ax[1, 0].grid(True)
    
    ax[1, 1].plot(1.0e6 * sig_py, label='sig_py [mm-mr]')
    ax[1, 1].legend(loc='best')
    ax[1, 1].grid(True)
    ax[1, 0].set_xlabel('turn')
    ax[1, 1].set_xlabel('turn')
    if saveplt: plt.savefig('sig-x-px-y-py.png')

    # min/max x, px, y, py
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(1000.0 *  min_x, label='min_x [mm]')
    ax[0, 0].plot(1000.0 * max_x, label='max_x [mm]')
    ax[0, 0].grid(True)
    ax[0, 0].legend(loc='best')

    ax[0, 1].plot(1.0e6 * min_px, label='min_px [mm-mr]')
    ax[0, 1].plot(1.0e6 * max_px, label='max_px [mm-mr]')
    ax[0, 1].legend(loc='best')
    ax[0, 1].grid(True)

    ax[1, 0].plot(1000.0 * min_y, label='min_y [mm]')
    ax[1, 0].plot(1000.0 * max_y, label='max_y [mm]')
    ax[1, 0].legend(loc='best')
    ax[1, 0].grid(True)
    
    ax[1, 1].plot(1.0e6 * min_py, label='min_py [mm-mr]')
    ax[1, 1].plot(1.0e6 * max_py, label='max_py [mm-mr]')
    ax[1, 1].legend(loc='best')
    ax[1, 1].grid(True)
    ax[1, 0].set_xlabel('turn')
    ax[1, 1].set_xlabel('turn')
    if saveplt: plt.savefig('min-max-x-px-y-py.png')

    # mean, sig t, pt
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(mean_t, label='mean_t [s]')
    ax[0, 0].legend(loc='best')
    ax[0, 0].grid(True)
    
    ax[0, 1].plot(sig_t, label='sig_t [s]')
    ax[0, 1].legend(loc='best')
    ax[0, 1].grid(True)

    ax[1, 0].plot(mean_pt, label='mean_pt')
    ax[1, 0].legend(loc='best')
    ax[1, 0].grid(True)
    
    ax[1, 1].plot(sig_pt, label='sig_pt')
    ax[1, 1].legend(loc='best')
    ax[1, 1].grid(True)
    ax[1, 0].set_xlabel('turn')
    ax[1, 1].set_xlabel('turn')
    if saveplt: plt.savefig('sig-t-pt.png')
    
    # min/max t pt
    f, ax = plt.subplots(2, 1)
    ax[0].plot(min_t, label='min_t')
    ax[0].plot(max_t, label='max_t')
    ax[0].legend(loc='best')

    ax[1].plot(min_pt, label='min_pt')
    ax[1].plot(max_pt, label='max_pt')
    ax[1].legend(loc='best')
    ax[1].set_xlabel('turn')
    if saveplt: plt.savefig('minmax-t-pt.png')

    # V, particles, phase
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(runstatus['particles'], label='N particles')
    ax[0, 0].legend(loc='best')
    ax[0, 0].grid(True)
    
    ax[0, 1].plot(runstatus['V'], label='total RF Voltage [MV]')
    ax[0, 1].legend(loc='best')
    ax[0, 1].grid(True)
    
    ax[1, 0].plot(runstatus['gamma'], label='gamma')
    ax[1, 0].legend(loc='best')
    ax[1, 0].grid(True)
    
    ax[1, 1].plot(runstatus['phase'], label='RF phase [rad]')
    ax[1, 1].legend(loc='best')
    ax[1, 1].grid(True)
    
    ax[1, 0].set_xlabel('turn')
    ax[1, 1].set_xlabel('turn')
    if saveplt: plt.savefig('N-V-gamma-phase.png')
    
    plt.show()

    return

if __name__ == "__main__":
    main(len(sys.argv) > 1)
