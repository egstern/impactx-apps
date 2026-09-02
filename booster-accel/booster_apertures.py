#!
# Add apertures to Booster lattice
#

from impactx import elements

inches_to_m = 0.0254
short_straight_aperture = 4.5 # diameter in inches
long_straight_aperture = 3.125 # diameter in inches
rf_aperture = 2.25 # diameter in inches

# These aperture definitions come from digitizing Figure 1 of
# preprint FERMILAB-CONF-12-194-AD.
fmag_vertices = [(3.74,0.506), (-2.16, +1.09), (-2.16, -1.09), (3.74, -0.506)] # in inches
dmag_vertices = [(3.50,1.52), (-2.40, +0.901), (-2.40, -0.901), (3.50, -1.52)] # in inches

# construct the list of vertice indices
fmag_vertex_x = [fv[0]*inches_to_m for fv in fmag_vertices] + [fmag_vertices[0][0]*inches_to_m]
fmag_vertex_y = [fv[1]*inches_to_m for fv in fmag_vertices] + [fmag_vertices[0][1]*inches_to_m]

# find distance from center to aperture edge
# need the equation for the between (-2.54, 0.925) to (3.46, 0.925)
# line that goes between (x1, y1) and (x2, y2)
#  A = (y2-y1), B = -(x2-x1), C = -x1*(y2-y1) + y1*(x2-x1)
# min_radius2 = C**2/(A**2 + B**2)
A = fmag_vertices[1][1] - fmag_vertices[0][1]
B = -(fmag_vertices[1][0] - fmag_vertices[0][0])
C = -fmag_vertices[0][0]*(fmag_vertices[1][1] - fmag_vertices[0][1]) + fmag_vertices[0][1]*(fmag_vertices[1][0] - fmag_vertices[0][0])

fmag_min_radius2 = inches_to_m**2 * C**2/(A**2 + B**2)
                    
dmag_vertex_x = [dv[0]*inches_to_m for dv in dmag_vertices] + [dmag_vertices[0][0]*inches_to_m]
dmag_vertex_y = [dv[1]*inches_to_m  for dv in dmag_vertices] + [dmag_vertices[0][1]*inches_to_m]

# get min_aperture1 for dmag
A = dmag_vertices[1][1] - dmag_vertices[0][1]
B = -(dmag_vertices[1][0] - dmag_vertices[0][0])
C = -dmag_vertices[0][0]*(dmag_vertices[1][1] - dmag_vertices[0][1]) + dmag_vertices[0][1]*(dmag_vertices[1][0] - dmag_vertices[0][0])

dmag_min_radius2 = inches_to_m**2 * C**2/(A**2 + B**2)

FMAG_aperture = elements.PolygonAperture(
    fmag_vertex_x, fmag_vertex_y, fmag_min_radius2)
DMAG_aperture = elements.PolygonAperture(
    dmag_vertex_x, dmag_vertex_y, dmag_min_radius2)


# The short straight section of the Booster is roughly the 1.5m in
# between the two focussing gradient magnets including the correction
# package elements extending on both sides to the nearest defocussing
# magnet.

short_straight_names = r"(sa)|(hsxx)|(vsxx)|(qsxx)|(bpms)|(qssxx)|(sxsxx)|(sssxx)|(sb)|(mins)"

# lattice must have type KnownElementsList
def set_short_straight_apertures(lattice):
    short_straights = lattice.select(name=short_straight_names)
    for elem in short_straights:
        elem.aperture_x = short_straight_aperture*inches_to_m/2.0
        elem.aperture_y = short_straight_aperture*inches_to_m/2.0
    return len(short_straights)

# The long straight section of the Booster is the roughly 6m space between the
# two defocussing gradient magnets including the correction package
# elements.

long_straight_names = r"(dlong)|(hlxx)|(vlxx)|(qlxx)|(qlsxx)|(sxlxx)|(sslxx)|(drifta)|(driftb)|(dmidls)|(drifte)"

# lattice must have type KnownElementsList
def set_long_straight_apertures(lattice):
    long_straights = lattice.select(name=long_straight_names)
    for elem in long_straights:
        elem.aperture_x = long_straight_aperture*inches_to_m/2.0
        elem.aperture_y = long_straight_aperture*inches_to_m/2.0
    return len(long_straights)

rf_name = "drrf"

# lattice must have type KnownElementsList
def set_rf_apertures(lattice):
    rf_elems = lattice.select(name=rf_name)
    for elem in rf_elems:
        elem.aperture_x = rf_aperture*inches_to_m/2.0
        elem.aperture_y = rf_aperture*inches_to_m/2.0
    return len(rf_elems)

# copy elements from old_lattice into a new lattice inserting
# proper polygon apertures around the FMAG and DMAG magnets

def insert_apertures(old_lattice):
    new_lattice = elements.KnownElementsList([])
    for elem in old_lattice:
        elem_name = elem.name
        # is it a CFbend
        if elem.to_dict()['type'] == "ExactCFbend":
            # OK is it a F magnet
            if elem_name[0:4] == "fmag":
                new_lattice.append(FMAG_aperture)
                new_lattice.append(elem)
                new_lattice.append(FMAG_aperture)
            elif elem_name[0:4] == "dmag": # is it a D magnet?
                new_lattice.append(DMAG_aperture)
                new_lattice.append(elem)
                new_lattice.append(DMAG_aperture)
            else: # What? bend magnet that's not F or D?
                raise RuntimeError("unknown bending magnet")
        else:
            # Some other kind of element
            new_lattice.append(elem)
    return new_lattice

# turn on short_straight, long_straight, and RF apertures by adding
# element parameters in-place
def set_apertures(lattice):
    set_short_straight_apertures(lattice)
    set_long_straight_apertures(lattice)
    set_rf_apertures(lattice)
    use_lattice = insert_apertures(lattice)
    return use_lattice
    
