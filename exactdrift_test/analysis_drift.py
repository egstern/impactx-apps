import numpy as np
import openpmd_api as io
from scipy.stats import moment
from pytest import approx

def compare_rel(s, x, y, tol):
    print(f'{s}: {x} should be {y}, ', end='')
    if abs(x) < tol and abs(y) < tol:
        print(f'abs difference: {abs(x-y)}')
        assert x == approx(y, abs=tol)
    else:
        print(f'rel difference: {abs(x-y)/(abs(x)+abs(y))}')
        assert x == approx(y, rel=tol)

# open the data file
series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)

iterations = list(series.iterations)


start = series.iterations[iterations[0]]
finish = series.iterations[iterations[-1]]

start_beam = start.particles["beam"].to_df()
finish_beam = finish.particles["beam"].to_df()

mp = finish.particles["beam"].get_attribute('mass_ref')
beta_ref = finish.particles["beam"].get_attribute('beta_ref')
gamma_ref = finish.particles["beam"].get_attribute('gamma_ref')
beta_gamma_ref = finish.particles["beam"].get_attribute('beta_gamma_ref')
# The 0 particle should remain at 0
x1 = finish_beam['position_x']
y1 = finish_beam['position_y']
t1 = finish_beam['position_t']
px1 = finish_beam['momentum_x']
py1 = finish_beam['momentum_y']
pt1 = finish_beam['momentum_t']

# The 0 particle should stay at 0
assert abs(x1[0]) < 1.0e-14
assert abs(y1[0]) < 1.0e-14
assert abs(t1[0]) < 1.0e-14
assert abs(px1[0]) < 1.0e-14
assert abs(py1[0]) < 1.0e-14
assert abs(pt1[0]) < 1.0e-14

# particle 1
compare_rel('rel diff x1[1]', x1[1], 0.001, 1.0e-14)
# test time offset
L = np.sqrt(0.002**2 + 1)
ct0 = 1.0/beta_ref
ct1 = L/beta_ref
dt = (ct1 - ct0)
compare_rel('rel diff t1[1]', t1[1], dt, 1.0e-14)

# particle 2
# particle has dE/p of 1.e-2. Calculate it's beta
print('pt1[2]: ', pt1[2])
new_gamma = -beta_gamma_ref * pt1[2] + gamma_ref
new_beta_gamma = np.sqrt(new_gamma**2 - 1)
new_beta = new_beta_gamma/new_gamma
print('orig_beta: ', beta_ref)
print('new_beta: ', new_beta)
L = 1.0 # this particle goes straight
new_ct = L/new_beta
orig_ct = L/beta_ref
print('orig ct: ', orig_ct)
print('new ct: ', new_ct)
dt = (new_ct - orig_ct)
print('ct_new - ct_orig: ', dt)
print('ct particle 0: ', t1[0])
print('ct particle 2: ', t1[2])
compare_rel('rel diff particle 2 ct', t1[2], dt, 1.0e-14)

