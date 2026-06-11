#!/usr/bin/env python

import sys, os
import numpy as np
import openpmd_api as io

series = io.Series(sys.argv[1], io.Access.read_only)
iterations = list(series.iterations)

print('len(sys.argv): ', len(sys.argv))
print(sys.argv)

if len(sys.argv) > 2:
    iter = iterations[int(sys.argv[2])]
else:
    iter = iterations[0]

df = series.iterations[iter].particles["beam"].to_df()

part = df[['position_x', 'momentum_x', 'position_y', 'momentum_y',
            'position_t', 'momentum_t']]

cov = np.cov(part, rowvar=False)

print(part.shape[0], end=' ')
for j in range(6):
    for i in range(6):
        print(float(cov[i, j]), end=' ')
print()
