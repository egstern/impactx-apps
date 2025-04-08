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
series1 = io.Series("diags/openPMD/monitor1.h5", io.Access.read_only)
seriesn = io.Series("diags/openPMD/monitorn.h5", io.Access.read_only)

lastiter1 = list(series1.iterations)[-1]
lastitern = list(seriesn.iterations)[-1]

beam_1slice = series1.iterations[lastiter1].particles['beam'].to_df()
beam_nslice = seriesn.iterations[lastitern].particles['beam'].to_df()

t_1slice = beam_1slice['position_t']
t_nslice = beam_nslice['position_t']

print('particle 2 t (1 slice): ', t_1slice[2])
print('particle 2 t (n slice): ', t_nslice[2])
