#!/usr/bin/env python3

import impactx

elem1 = impactx.elements.Drift(ds=1.0, name="d1")
elem2 = impactx.elements.Drift(ds=2.0, name="d2")

sim = impactx.ImpactX()

objs = [elem1, elem2, elem1, elem2]

print('object IDs in objs:')
for i, e in enumerate(objs):
    print(i, id(e), e.to_dict())

print()
print('change third element length')
objs[2].ds=3.0

print('changed objs')
for i, e in enumerate(objs):
    print(i, id(e), e.to_dict())
