#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpmd_api as io
from scipy.constants import c, eV
def foo():
    series = io.Series('diags/openPMD/monitor.h5', io.Access_Type.read_only)
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
    ax[0, 0].plot(s, mean_x, label='mean_x')
    ax[0, 0].legend(loc='best')

    ax[0, 1].plot(s, mean_px, label='mean_px')
    ax[0, 1].legend(loc='best')

    ax[1, 0].plot(s, mean_y, label='mean_y')
    ax[1, 0].legend(loc='best')

    ax[1, 1].plot(s, mean_py, label='mean_py')
    ax[1, 1].legend(loc='best')

    # sig x, y
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(s, sig_x, label='sig_x')
    ax[0, 0].legend(loc='best')

    ax[0, 1].plot(s, sig_px, label='sig_px')
    ax[0, 1].legend(loc='best')

    ax[1, 0].plot(s, sig_y, label='sig_y')
    ax[1, 0].legend(loc='best')

    ax[1, 1].plot(s, sig_py, label='sig_py')
    ax[1, 1].legend(loc='best')

    # min/max x, px, y, py
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(s, min_x, label='min_x')
    ax[0, 0].plot(s, max_x, label='max_x')
    ax[0, 0].legend(loc='best')

    ax[0, 1].plot(s, min_px, label='min_px')
    ax[0, 1].plot(s, max_px, label='max_px')
    ax[0, 1].legend(loc='best')

    ax[1, 0].plot(s, min_y, label='min_y')
    ax[1, 0].plot(s, max_y, label='max_y')
    ax[1, 0].legend(loc='best')

    ax[1, 1].plot(s, min_py, label='min_py')
    ax[1, 1].plot(s, max_py, label='max_py')
    ax[1, 1].legend(loc='best')

    # mean, sig t, pt
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(s, mean_t, label='mean_t')
    ax[0, 0].legend(loc='best')

    ax[0, 1].plot(s, sig_t, label='sig_t')
    ax[0, 1].legend(loc='best')

    ax[1, 0].plot(s, mean_pt, label='mean_pt')
    ax[1, 0].legend(loc='best')

    ax[1, 1].plot(s, sig_pt, label='sig_pt')
    ax[1, 1].legend(loc='best')
    
    # min/max t pt
    f, ax = plt.subplots(2, 1)
    ax[0].plot(s, min_t, label='min_t')
    ax[0].plot(s, max_t, label='max_t')
    ax[0].legend(loc='best')

    ax[1].plot(s, min_pt, label='min_pt')
    ax[1].plot(s, max_pt, label='max_pt')
    ax[1].legend(loc='best')

    # V, particles, phase
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(runstatus['particles'], label='N particles')
    ax[0, 0].legend(loc='best')

    ax[0, 1].plot(runstatus['V'], label='Voltage')
    ax[0, 1].legend(loc='best')

    ax[1, 0].plot(runstatus['gamma'], label='gamma')
    ax[1, 0].legend(loc='best')

    ax[1, 1].plot(runstatus['phase'], label='phase')
    ax[1, 1].legend(loc='best')

    plt.show()

    return

def plot_iter(series, iter):
    beam = series.iterations[iter].particles["beam"]
    mass_ref = beam.get_attribute("mass_ref")
    mass_mev = 1.0e-6 * mass_ref * c**2/eV
    print(f"mass ref: {mass_ref}, ({mass_mev} MeV)")
    beta_gamma_ref = beam.get_attribute("beta_gamma_ref")
    gamma_ref = beam.get_attribute("gamma_ref")
    should_be_one = (gamma_ref - beta_gamma_ref) * (gamma_ref + beta_gamma_ref)
    assert abs(should_be_one - 1) < 1.0e-15
    pref = mass_mev * beta_gamma_ref
    eref = mass_mev * gamma_ref
    print("beta_gamma_ref: ", beta_gamma_ref)
    print("gamma_ref: ", gamma_ref)
    print("pref: ", pref, "MeV")
    print("eref: ", eref, "MeV")
    df = beam.to_df()
    print(len(df), "particles")
    
    energy = -pref * df['momentum_t'] + eref
    too_smalls = ((energy - eref) < mass_mev)
    if too_smalls.any():
        print(too_smalls.sum(), "energy too small")

    print("mean_x from particles: ", df['position_x'].mean(),
          "from RBC: ", beam.get_attribute('mean_x'))
    print("sigma_x from particles: ", df['position_x'].std(),
          "from RBC: ", beam.get_attribute('sigma_x'))
    print("min_x from particles: ", df['position_x'].min(),
          "from RBC: ", beam.get_attribute('min_x'))
    print("max_x from particles: ", df['position_x'].max(),
          "from RBC: ", beam.get_attribute('max_x'))

    print("mean_y from particles: ", df['position_y'].mean(),
          "from RBC: ", beam.get_attribute('mean_y'))
    print("sigma_y from particles: ", df['position_y'].std(),
          "from RBC: ", beam.get_attribute('sigma_y'))
    print("min_y from particles: ", df['position_y'].min(),
          "from RBC: ", beam.get_attribute('min_y'))
    print("max_y from particles: ", df['position_y'].max(),
          "from RBC: ", beam.get_attribute('max_y'))

    print("mean_t from particles: ", df['position_t'].mean(),
          "from RBC: ", beam.get_attribute('mean_t'))
    print("sigma_t from particles: ", df['position_t'].std(),
          "from RBC: ", beam.get_attribute('sigma_t'))
    print("min_t from particles: ", df['position_t'].min(),
          "from RBC: ", beam.get_attribute('min_t'))
    print("max_t from particles: ", df['position_t'].max(),
          "from RBC: ", beam.get_attribute('max_t'))

    print("mean_pt from particles: ", df['momentum_t'].mean(),
          "from RBC: ", beam.get_attribute('mean_pt'))
    print("sigma_pt from particles: ", df['momentum_t'].std(),
          "from RBC: ", beam.get_attribute('sigma_pt'))
    print("min_pt from particles: ", df['momentum_t'].min(),
          "from RBC: ", beam.get_attribute('min_pt'))
    print("max_pt from particles: ", df['momentum_t'].max(),
          "from RBC: ", beam.get_attribute('max_pt'))


    plt.title("position_x")
    plt.hist(df['position_x'], 100)

    plt.figure()
    plt.title('position_y')
    plt.hist(df['position_y'], 100)

    plt.figure()
    plt.title('position_t')
    plt.hist(df['position_t'], 100)

    plt.figure()
    plt.title('momentum_t')
    plt.hist(df['momentum_t'], 100)

    plt.show()
    
def main():
    fname = sys.argv[1]
    iter = int(sys.argv[2])
    print('plotting file ', fname, 'iteration: ', iter)
    series = io.Series(fname, io.Access_Type.read_only)
    plot_iter(series, iter)
    series.close()

if __name__ == "__main__":
    main()
