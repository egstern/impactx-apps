import sys, os
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
from scipy.stats import moment
from scipy.constants import speed_of_light as c, elementary_charge as qe
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
series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)

iterations = list(series.iterations)

print('iterations: ', iterations)

start = series.iterations[iterations[0]]
finish = series.iterations[iterations[-1]]

start_beam = start.particles["beam"].to_df()
finish_beam = finish.particles["beam"].to_df()

print('start_beam.shape: ', start_beam.shape)
print('finish_beam.shape: ', finish_beam.shape)

mp_kg = start.particles["beam"].get_attribute('mass_ref')
mp_mev = mp_kg * c**2 * 1.0e-6/qe

print('mp (MeV): ', mp_mev)
start_beta = start.particles["beam"].get_attribute('beta_ref')
start_gamma = start.particles["beam"].get_attribute('gamma_ref')
start_betagamma = start.particles["beam"].get_attribute('beta_gamma_ref')


finish_beta = finish.particles["beam"].get_attribute('beta_ref')
finish_gamma = finish.particles["beam"].get_attribute('gamma_ref')
finish_betagamma = finish.particles['beam'].get_attribute('beta_gamma_ref')

assert start_beta == approx(finish_beta, rel=1.0e-15)
assert start_gamma == approx(finish_gamma, rel=1.0e-15)
assert start_betagamma == approx(finish_betagamma, rel=1.0e-15)

print('start beta: ', start_beta)
print('finish beta: ', finish_beta)

start_t = start_beam['position_t']
finish_t = finish_beam['position_t']

start_pt = start_beam['momentum_t']
finish_pt = finish_beam['momentum_t']

plt.figure()
plt.title('positions')
plt.plot(start_t, 'D', label='Start t')
plt.plot(finish_t, 'o', label='Finish t')
plt.legend(loc='best')

plt.figure()
plt.title('pt')
plt.plot(start_t, start_pt, 'o', label='start pt')
plt.plot(start_t, finish_pt, 'd', label='finish pt')
plt.legend(loc='best')

# calculate delta-E from pt
dfinish_gamma = -start_betagamma * finish_pt

volt = (1.0/22)
plt.figure()
plt.title('Energy change [MeV]')
plt.plot(start_t, dfinish_gamma*mp_mev, '-', label='dE [MeV)')
plt.xlabel('position t [m]')
plt.ylabel('dE [MeV]')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('dE.png')

plt.show()

