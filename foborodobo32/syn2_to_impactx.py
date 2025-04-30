#!/usr/bin/env python

# Create and return a list of ImpactX elements from a
# Synergia lattice

# This probably has to be called after all the ImpactX initialization with
# the reference particle.

import numpy as np
import synergia
import impactx

ET = synergia.lattice.element_type


def cnv_drift(elem):
    ds = elem.get_length()
    nm = elem.get_name()
    return impactx.elements.ExactDrift(ds, nslice=1, name=nm)

def cnv_sbend(elem):
    # lots of different kinds of bends
    if elem.get_double_attribute('k2') != 0.0:
        print('k2 attribute not supported for sbend in ImpactX')
    if elem.has_double_attribute('k1s'):
        print('k1s attribute not supported for sbend in ImpactX')
    if elem.get_double_attribute('h1') != 0.0:
        print('h1 attribute not supported for sbend in ImpactX')
    if elem.get_double_attribute('h2') != 0.0:
        print('h2 attribute not supported for sbend in ImpactX')
    if elem.get_double_attribute('e1') != 0.0 or elem.get_double_attribute('e2') != 0.0:
        print('e1/e2 attribute not yet supported for sbend in ImpactX')

    if elem.get_double_attribute('k1') != 0.0:
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
    return impactx.elements.ChrQuad(ds, k1, unit=0, nslice=1, name=elem.get_name())

def cnv_multipole(elem):
    # ImpactX multipole elements only have a single order each so
    # we have to peel each order and possibly create multiple elements to get the
    # full description.
    ename = elem.get_name()
    knlvect = elem.get_vector_attribute('knl', [])
    kslvect = elem.get_vector_attribute('ksl', [])
    nlen = len(knlvect)
    slen = len(kslvect)
    maxlen = max(nlen, slen)
    norm_mom = np.zeros(maxlen)
    skew_mom = np.zeros(maxlen)
    norm_mom[:nlen] = knlvect
    skew_mom[:slen] = kslvect
    elem_list = []
    for order in range(maxlen):
        # is there anything this order?
        if norm_mom[order] == 0.0 and skew_mom[order] == 0.0:
            pass
        elem_list.append(impactx.elements.Multipole(order+1, norm_mom[order], skew_mom[order],
                                                    name=f'{ename}_{order}'))
    return elem_list


def cnv_rfcavity(elem, refpart):
    mp = refpart.get_mass()
    rfvolt = elem.get_double_attribute('volt')/1000.0 # get the voltage in GV
    freq = elem.get_double_attribute('freq')*1.0e6 # get the freq in Hz
    phase = elem.get_double_attribute('lag', 0.0)*360.0-90.0
    # if cavity length > 0, create two drifts to sandwich it
    L = elem.get_length()
    # if cavity length > 0, create two drifts to sandwich it
    RFelem = impactx.elements.ShortRF(rfvolt/mp, freq, phase, name=elem.get_name())
    if L == 0.0:
        # exactly 0 should be allowable for == comparison
        return RFelem
    else:
        halfL = L/2
        d1 = impactx.elements.ExactDrift(halfL, nslice=1, nm=f'{elem.get_name()}_U')
        d2 = impactx.elements.ExactDrift(halfL, nslice=1, nm=f'{elem.get_name()}_D')   
        RFunit = [ d1,
                   RFelem,
                   d2]
    return RFunit

# Hi Rob,

# Here's the full expression for  converting MAD8 multipoles to MADX.

# For MAD8 multipole KnL=str and Tn=angle where n could be 1,2,3,4, ....

# The MADX normal component is COS((n+1)*angle)
# The MADX skew component is -SIN((n+1)*angle)

# Yes, that is a negative sign only on the skew component because the
# derivation contains a rotation around the negative angle.

# MAD8 multipole
# MULTIPOLE, K2L=str, T2=angle

# Converts to MADX statement
# MULTIPOLE, KNL={0, 0, str*cos(3*angle)}, KSL={0, 0, -str*sin(3*angle))};

# Your usual multipole elements that specify an angle in the Recycler
# latticelike the one below  have written out values for angles, so for
# instance T2=pi/6 which is the normal skew angle for sextupoles. The normal
# component vanishes and the skew component receives negative the strength.

# MP114BS: MULTIPOLE, TYPE=RGF021_S_BODY, K1L=0.16E-4, T1=0.785398163397,
# K2L=&
# 0.808E-3*DIR, T2=0.523598775598, K3L=0.064894, T3=0.392699081699, K4L=&
# 15.028637*DIR, T4=0.314159265359

# Converts to
# MP114BS: MULTIPOLE, TYPE=RGF021_S_BODY, KSL={-0.16E-4,  -0.808E-3*DIR,
# -0.064894, -15.028637*DIR};


#     Eric

# ImpactX doesn't current support long sextupoles so this
# is a kludge with thin sextupole between drifts
def cnv_sextupole(elem):
    L = elem.get_length()
    k2 = elem.get_double_attribute('k2')
    # The Booster lattice includes elements with the tilt attribute
    tilt = elem.get_double_attribute('tilt', 0.0)
    k2nl = k2 * np.cos(3*tilt)
    k2sl = -k2 * np.sin(3*tilt)
    sxelem = impactx.elements.Multipole(3, k2nl, k2sl, name=elem.get_name())    
    if L == 0.0:
        return sxelem
    else:
        halfL = L/2
        d1 = impactx.elements.ExactDrift(halfL, nslice=1, nm=f'{elem.get_name()}_U')
        d2 = impactx.elements.ExactDrift(halfL, nslice=1, nm=f'{elem.get_name()}_D')   
        SXunit = [ d1, sxelem, d2]
        return SXunit

# ImpactX doesn't current support long octtupoles so this
# is a kludge with thin octupole between drifts
def cnv_octupole(elem):
    L = elem.get_length()
    k3 = elem.get_double_attribute('k3')
    # The Booster lattice includes elements with the tilt attribute
    tilt = elem.get_double_attribute('tilt', 0.0)
    k3nl = k3 * np.cos(4*tilt)
    k3sl = -k3 * np.sin(4*tilt)
    ocelem = impactx.elements.Multipole(4, k3nl, k3sl, name=elem.get_name())    
    if L == 0.0:
        return ocelem
    else:
        halfL = L/2
        d1 = impactx.elements.ExactDrift(halfL, nslice=1, nm=f'{elem.get_name()}_U')
        d2 = impactx.elements.ExactDrift(halfL, nslice=1, nm=f'{elem.get_name()}_D')   
        OCunit = [ d1, ocelem, d2]
        return OCunit

def syn2_to_impactx(lattice):
    # lattice must have a reference particle
    try:
        refpart = lattice.get_reference_particle()
    except:
        print("cannot get reference particle.")
        return None


    impactx_lattice = []

    # Always begin with a monitor element
    monitor = impactx.elements.BeamMonitor("monitor", backend="h5")
    impactx_lattice.append(monitor)

    # peel elements from the synergia lattice, converting to ImpactX elements
    for elem in lattice.get_elements():
        etype = elem.get_type()

        if etype == ET.drift:
            impactx_lattice.append(cnv_drift(elem))
        elif etype == ET.sbend:
            impactx_lattice.append(cnv_sbend(elem))
        elif etype == ET.quadrupole:
            impactx_lattice.append(cnv_quadrupole(elem))
        elif etype == ET.sextupole:
            sxelem = cnv_sextupole(elem)
            if isinstance(sxelem, list):
                impactx_lattice.extend(sxelem)
            else:
                impactx_lattice.append(sxelem)
        elif etype == ET.octupole:
            ocelem = cnv_octupole(elem)
            if isinstance(ocelem, list):
                impactx_lattice.extend(ocelem)
            else:
                impactx_lattice.append(ocelem)
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
        elif etype == ET.multipole:
            mpelem = cnv_multipole(elem)
            if isinstance(mpelem, list):
                impactx_lattice.extend(mpelem)
            else:
                impactx_lattice.append(mpelem)
        else:
            print('warning: unsupported element: ', etype)

        pass

    return impactx_lattice
    
