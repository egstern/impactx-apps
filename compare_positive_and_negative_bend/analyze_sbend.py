import sys, os
import numpy as np
import openpmd_api as io
import pandas as pd
from scipy.stats import moment
from pytest import approx

def compare_rel(s, x, y, tol):
    print(f'{s}: {x} should be {y}, ', end='')
    if abs(x) < tol or abs(y) < tol:
        print(f'abs difference: {abs(x-y)}')
        assert x == approx(y, abs=tol)
    else:
        print(f'rel difference: {abs(x-y)/(abs(y))}')
        assert x == approx(y, rel=tol)

# open the data file
series_pos = io.Series("diags/openPMD/monitor_pos.h5", io.Access.read_only)
series_neg = io.Series("diags/openPMD/monitor_neg.h5", io.Access.read_only)
iterations_pos = list(series_pos.iterations)
iterations_neg = list(series_neg.iterations)

start_pos = series_pos.iterations[iterations_pos[0]]
finish_pos = series_pos.iterations[iterations_pos[-1]]
start_neg = series_neg.iterations[iterations_neg[0]]
finish_neg = series_neg.iterations[iterations_neg[-1]]

start_beam_pos = start_pos.particles["beam"].to_df()
finish_beam_pos = finish_pos.particles["beam"].to_df()
start_beam_neg = start_neg.particles["beam"].to_df()
finish_beam_neg = finish_neg.particles["beam"].to_df()

mp = finish_pos.particles["beam"].get_attribute('mass_ref')
beta_ref = finish_pos.particles["beam"].get_attribute('beta_ref')
gamma_ref = finish_pos.particles["beam"].get_attribute('gamma_ref')
beta_gamma_ref = finish_pos.particles["beam"].get_attribute('beta_gamma_ref')

x_pos_start = start_beam_pos[['position_x', 'position_y', 'position_t', 'momentum_x', 'momentum_y', 'momentum_t']][0:2]

x_neg_start = start_beam_neg[['position_x', 'position_y', 'position_t', 'momentum_x', 'momentum_y', 'momentum_t']][0:2]

print('xpos_start: ')
print(x_pos_start)
print('xneg_start: ')
print(x_neg_start)

pd.set_option('display.precision', 12)
x_pos_finish = finish_beam_pos[['position_x', 'position_y', 'position_t', 'momentum_x', 'momentum_y', 'momentum_t']][0:2]

x_neg_finish = finish_beam_neg[['position_x', 'position_y', 'position_t', 'momentum_x', 'momentum_y', 'momentum_t']][0:2]

print('xpos_finish: ')
print(x_pos_finish)
print('xneg_finish: ')
print(x_neg_finish)

# Check position_t for both pos and neg sbends

R = 1.0
d = -0.001
L0 = np.pi/2 * R
# Pos bend
L1 = R * (np.pi - np.arccos(d/R))

# neg bend
L2 = R * np.arccos(d/R)

dt_pos = (L1 - L0)/beta_ref

dt_neg = (L2 - L0)/beta_ref

print('dt_pos: ', dt_pos)
print('dt_neg: ', dt_neg)

print('frac. deviation pos dt: ', abs(dt_pos/x_pos_finish['position_t'][1] - 1))
print('frac. deviation neg dt: ', abs(dt_neg/x_neg_finish['position_t'][1] - 1))

assert abs(dt_pos/x_pos_finish['position_t'][1] - 1) < 5.0e-13
assert abs(dt_neg/x_neg_finish['position_t'][1] - 1) < 5.0e-13


