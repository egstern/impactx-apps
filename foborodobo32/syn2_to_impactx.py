#!/usr/bin/env python

# Create and return a list of ImpactX elements from a
# Synergia lattice

import synergia
import impactx

ET = synergia.lattice.element_type


def cnv_drift(elem):
    ds = elem.get_length()
    nm = elem.get_name()
    return impact.elements.ExactDrift(ds, nslice=1, name=nm)

def cnv_sbend(elem):
    # lots of different kinds of bends
    if elem.has_attr('k2'):
        print('k2 attribute not supported for sbend in ImpactX')
    if elem.has_attr('k1s'):
        print('k1s attribute not supported for sbend in ImpactX')
    if elem.has_attr('h1'):
        print('h1 attribute not supported for sbend in ImpactX')
    if elem.has_attr('h2'):
        print('h2 attribute not supported for sbend in ImpactX')
    if elem.has_attr('e1') or elem.has_attr('e2'):
        print('e1/e2 attribute not yet supported for sbend in ImpactX')

    if elem.has_attr('k1'):
        ds = elem.get_length()
        angle = elem.get_bend_angle()
        rc = angle/ds
        k1 = elem.get_double_attribute('k1')
        return impactx.elements.CFbend(ds, rc, k1, nslice=1, name=elem.get_name())
    else:
        # normal bend, no higher order moments
        ds = elem.get_length()
        # What kind of screwy program accepts angles in degrees?
        phi = elem.get_bend_angle() * 180/np.pi
        return impactx.elements.ExactSbend(ds, phi, nslice=1, name=elem.get_name())
    raise RuntimeError('should not happen')

def cnv_quadrupole(elem):
    ds = elem.get_length()
    k1 = elem.get_double_attribute('k1')
    return impactx.element.ChrQuad(ds, k1, unit=0, nslice=1, name=elem.get_name())

def cnv_multipole(elem):
    pass

def cnv_rfcavity(elem):
    mp = refpart.get_mass()
    rfvolt = elem.get_double_attribute('volt')/1000.0 # get the voltage in GV
    freq = elem.get_double_attribute('freq')*1.0e6 # get the freq in Hz
    phase = elem.get_double_attribute('lag', 0.0)*360.0-90.0
    # if cavity length > 0, create two drifts to sandwich it
    
    return impactx.element.ShortRF(rfvolt/mp, freq, phase, name=elem.get_name())

def syn2_to_impactx(lattice):
    # lattice must have a reference particle
    try:
        refpart = lattice.get_reference_particle()
    except:
        print("cannot get reference particle.")
        return None

    impactx_lattice = []

    for elem in lattice.get_elements():
        etype = elem.get_type()

        if etype == ET.drift:
            impactx_lattice.append(cnv_drift(elem))
        elif etype == ET.sbend:
            impactx_lattice.append(cnv_sbend(elem))
        elif etype == ET.quadrupole:
            impactx_lattice.append(cnv_quadrupole(elem))
        elif etype == ET.multipole:
            importx_lattice.append(cnv_multipole(elem))
        elif etype == ET.sextupole:
            importx_lattice.append(cnv_sextupole(elem))
        elif etype == ET.octupole:
            importx_lattice.append(cnv_octupole(elem))
        elif etype == ET.hkicker or etype == ET.vkicker or etype == ET.kicker:
            # both H and V kickers handled by same routine since
            # ImpactX has only one type of kicker element
            impactx_lattice.append(cnv_kicker(elem))
        elif etype == ET.nllens:
            impactx_lattice.append(cnv_nllens(elem))
        elif etype == ET.dipedge:
            impactx_lattice.append(cnv_dipedge(elem))
        elif etype == ET.rfcavity:
            rfelem = cnv_rfcavity(elem, refpart)
            if isinstance(rfelem, list):
                impactx_lattice.extend(rfelem)
            else:
                impactx_lattice.append(rfelem)
        elif etype == ET.
        else:
            print('warning: unsupported element: ', etype)

        pass

    return impactx_lattice
    
