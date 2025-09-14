#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpmd_api as io

def main(mfile):

    series = io.Series(mfile, io.Access.read_only)
    iters = list(series.iterations)
    niters = len(iters)

    s = np.zeros(niters)

    x_mean = np.zeros(niters)
    x_std = np.zeros(niters)
    x_emit = np.zeros(niters)
    x_beta = np.zeros(niters)

    y_mean = np.zeros(niters)
    y_std = np.zeros(niters)
    y_emit = np.zeros(niters)
    y_beta = np.zeros(niters)

    for c, it in enumerate(iters):
        s[c] = series.iterations[it].particles["beam"].get_attribute('s_ref')
        x_mean[c] = series.iterations[it].particles["beam"].get_attribute('x_mean')
        x_std[c] = series.iterations[it].particles["beam"].get_attribute('sig_x')
        x_emit[c] = series.iterations[it].particles["beam"].get_attribute('emittance_x')
        x_beta[c] = series.iterations[it].particles["beam"].get_attribute('beta_x')

        y_mean[c] = series.iterations[it].particles["beam"].get_attribute('y_mean')
        y_std[c] = series.iterations[it].particles["beam"].get_attribute('sig_y')
        y_emit[c] = series.iterations[it].particles["beam"].get_attribute('emittance_y')
        y_beta[c] = series.iterations[it].particles["beam"].get_attribute('beta_y')

    f, axs = plt.subplots(2, sharex=True)
    axs[0].plot(s, x_mean, label='x mean')
    axs[0].set_ylabel('x [m]')
    axs[0].legend(loc='best')
    axs[1].plot(s, y_mean, label='y mean')
    axs[1].set_ylabel('y [m]')
    axs[1].set_xlabel('s [m]')
    axs[1].legend(loc='best')

    f, axs = plt.subplots(2, sharex=True)
    axs[0].plot(s, x_std, label='x std')
    axs[0].set_ylabel('x [m]')
    axs[0].legend(loc='best')
    axs[1].plot(s, y_std, label='y std')
    axs[1].set_ylabel('y [m]')
    axs[1].set_xlabel('s [m]')
    axs[1].legend(loc='best')
    
    f, axs = plt.subplots(2, sharex=True)
    axs[0].plot(s, x_emit, label='x emittance')
    axs[0].set_ylabel('x emittance [m-rad]')
    axs[0].legend(loc='best')
    axs[1].plot(s, y_emit, label='y emittance')
    axs[1].set_ylabel('y emittance[m rad]')
    axs[1].set_xlabel('s [m]')
    axs[1].legend(loc='best')
    
    f, axs = plt.subplots(2, sharex=True)
    axs[0].plot(s, x_beta, label='beta x')
    axs[0].set_ylabel('beta x [m]')
    axs[0].legend(loc='best')
    axs[1].plot(s, y_beta, label='beta y')
    axs[1].set_ylabel('beta y [m]')
    axs[1].set_xlabel('s [m]')
    axs[1].legend(loc='best')

    plt.show()
    rbc = pd.read_csv('diags/reduced_beam_characteristics.0.0', delimiter='\\s+', skiprows=3)
    f, axs = plt.subplots(2, sharex=True)

    axs[0].plot(rbc['s'], rbc['emittance_x'], label='x emittance [m-rad]')
    axs[0].set_ylabel('x emittance [m-rad]')
    axs[0].legend(loc='best')
    axs[1].plot(rbc['s'], rbc['emittance_y'], label='y emittance [m-rad]')
    axs[1].set_ylabel('y emittance [m-rad]')
    axs[1].legend(loc='best')
    axs[1].set_xlabel('s [m]')

    f, axs = plt.subplots(2, sharex=True)

    axs[0].plot(rbc['s'], rbc['beta_x'], label='beta x [m]')
    axs[0].set_ylabel('beta x [m]')
    axs[0].legend(loc='best')
    axs[1].plot(rbc['s'], rbc['beta_y'], label='beta y [m]')
    axs[1].set_ylabel('ybeta y [m]')
    axs[1].legend(loc='best')
    axs[1].set_xlabel('s [m]')

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
    pass

