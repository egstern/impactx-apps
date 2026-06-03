#!/usr/bin/env python3

from impactx import elements, synmadx

lattice_file = "sbbooster-cooked.madx"
lattice_line = "booster"

from syn2_to_impactx import syn2_to_impactx

# read the lattice from sbbooster-cooked.madx. This lattice has the
# corrector elements set for proper tune and chromaticity but has no
# RF. Return a list of ImpactX elements.
def get_lattice():
    reader = synmadx.MadX_reader()
    s_lattice = reader.get_lattice(lattice_line, lattice_file)
    ix_lattice = syn2_to_impactx(s_lattice)
    return ix_lattice

def main():
    ix_lattice = get_lattice()
    print(ix_lattice)
    return

if __name__ == "__main__":
    main()
    
