#!/usr/bin/env python3

import impactx

elem1 = impactx.elements.Drift(ds=1.0, name="d1")
elem2 = impactx.elements.Drift(ds=2.0, name="d2")

sim = impactx.ImpactX()

sim.lattice.extend([elem1, elem2, elem1, elem2])

print('the lattice: ', sim.lattice)
print()

print('length of lattice object', len(sim.lattice))
print()

print('directory of lattice object: ', dir(sim.lattice))

print('object IDs in sim.lattice:')
for i, e in enumerate(sim.lattice):
    print(i, id(e), e.to_dict())

print()
print('does enumeration change object ID?')
cnt = 0
for e in sim.lattice:
    print(i, id(e), e.to_dict())
    cnt = cnt + 1

print()
print('are elements indexable')
print(sim.lattice[2])

print()
print('change third element length')
sim.lattice[2].ds=3.0

print('changed lattice')
for i, e in enumerate(sim.lattice):
    print(i, id(e), e.to_dict())
