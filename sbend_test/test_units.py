#!/usr/bin/env python3
import sys, os
import numpy as np
import impactx

from scipy.constants import c, eV, m_p

mp = 1.0e-9*m_p*c**2/eV
print('proton mass: ', mp, "GeV")

KE = 0.8

def main():
    E = mp + KE
    gamma = E/mp
    betagamma = np.sqrt(gamma**2 - 1)
    beta = betagamma/gamma

    brho = c*1.0e-9
    print('brho: ', brho)

    # make 4 m long magnet with 15 degree CFbend
    L = 4
    theta = np.pi/12
    # BL/Brho = theta
    #
    # The k_0 coefficient is the normalized magnetic field or
    # B/Brho = 1/rho
    
    # rho is L/theta

    phi = theta*180/np.pi

    one_over_R = theta/L


    sb_obj1 = impactx.elements.ExactSbend(ds=L,
                                          phi=phi,
                                            nslice=4,
                                            name='foo')

    print('obj1 contains: ', dir(sb_obj1))
    print('obj1 dict: ', sb_obj1.to_dict())

    # what does it look like if I use unnormalized units?
    p = mp*betagamma
    pnorm = p/brho

    B = pnorm * one_over_R
    print('B: ', B, 'T')

    # sb_obj2 = impactx.elements.ExactSbend(ds=L, phi=None,
    #                                         B=B,
    #                                         nslice=4,
    #                                         name='foo')

    # print('obj2 dict: ', sb_obj2.to_dict())

if __name__ == "__main__":
    main()
    pass
