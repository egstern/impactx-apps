#!/usr/bin/env python

import sys, os
import numpy as np
import h5py
import openpmd_api as io
import PyNAFF as pnf


def main(synfile, opmdfile):

    h5 = h5py.File(synfile, 'r')
    print('number syn particles: ', h5.get('particles').shape[0])
    print('First ten syn particles')
    print(np.array2string(h5.get('particles')[0:10, 0:6], max_line_width=200))

    series = io.Series(opmdfile, io.Access.read_only)
    iters = list(series.iterations)
    niters = len(iters)
    print('opmd file niters: ', len(iters))
    print('iters: ', list(iters))

    # get data last iteration
    lastiter = iters[-1]
    
    iterdata = series.iterations[lastiter].particles["beam"].to_df()

    print('shape of opmd data: ', iterdata.shape)
    print(iterdata.columns)
    # build particle array from opmd data
    npart = h5.get('particles').shape[0]

    assert npart == len(iterdata)
    
    columns = zip(range(5),
        ['position_x', 'momentum_x', 'position_y', 'momentum_y', 'position_t'])
    
    
    for c, cc in columns:
        syn = h5.get('particles')[:, c]
        o = iterdata[cc]
        cmp = (syn - o) == 0.0
        if cmp.all():
            print('column ',c,'checks')

    sys.exit(0)
  
    cnt = 0
    for iter in iters:
        print('iteration: ', iter)
        iterdata = series.iterations[iter].particles["beam"].to_df()
        xdata[:, cnt] = iterdata['position_x'][:NX]
        ydata[:, cnt] = iterdata['position_y'][51:51+NY]
        cnt = cnt + 1
        #print('iterdata.shape: ', iterdata.shape)
        #print('position x data [:10]: ', iterdata['position_x'][:10])
        del iterdata
        pass

    np.save('xdata.npy', xdata)
    np.save('ydata.npy', ydata)
    return

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    pass
