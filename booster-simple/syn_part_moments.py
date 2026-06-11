#!/usr/bin/env python3

import sys, os
import numpy as np
import h5py

h5 = h5py.File(sys.argv[1], 'r')
part = h5.get('particles')
#print('part.shape: ', part.shape)
cov = np.cov(part[:, 0:6], rowvar=False)

#print(cov.shape)
#print(cov)

print(part.shape[0], end=' ')
for j in range(6):
    for i in range(6):
        print(float(cov[i, j]), end=' ')
print()

