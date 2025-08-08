#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np
#import transformation_utilities as pycoord
from scipy.constants import c, e as qe, m_p
mp_gev = m_p * c**2 * 1.0e-9/qe

import amrex.space3d as amr
from impactx import Config, ImpactX, elements


# beam is protons with kinetic energy of 0.8 GeV
Ebeam = mp_gev + 0.8
gamma0 = Ebeam/mp_gev
betagamma0 = np.sqrt(gamma0**2 - 1)
beta0 = betagamma0/gamma0

# Create the simulator for n slices
def create_sim(nslice):
    sim = ImpactX()


    return sim

def main():
    slices = [1, 5]

    for s in slices:
        sim = create_sim(s)


        sim.finalize()

    return

if __name__ == "__main__":
    main()
    pass

