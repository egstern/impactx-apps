#!/usr/bin/env python

# Create and return a list of ImpactX elements from a
# Synergia lattice

# This probably has to be called after all the ImpactX initialization with
# the reference particle.

import numpy as np
import synergia
import impactx

ET = synergia.lattice.element_type

nslice_by_elem_type = {
    'drift': 1,
    'sbend': 1,
    'quadrupole': 1,
    'sextupole': 1,
    'octupole': 1,
    'multipole': 1,
    'rfcavity': 1}

def cnv_drift(elem):
    ds = elem.get_length()
    nm = elem.get_name()
    ns = nslice_by_elem_type['drift']
    return impactx.elements.ExactDrift(ds, nslice=ns, name=nm)

def cnv_dipedge(elem):
    rc = 1/elem.get_double_attribute('h', 0.0)
    e1 = elem.get_double_attribute('e1', 0.0)
    fint = elem.get_double_attribute('fint', 0.0)
    hgap = elem.get_double_attribute('hgap', 0.0)
    ix_elem = impactx.elements.DipEdge(psi=e1, rc=rc, g=hgap,
                                       K2=fint, name=elem.get_name())
    return ix_elem

def cnv_sbend(elem):
    bendangle = elem.get_bend_angle()
    length = elem.get_length()
    radius_of_curvature = length/bendangle
    nm = elem.get_name()
    k1 = elem.get_double_attribute('k1', 0.0)
    k1s = elem.get_double_attribute('k1s', 0.0)
    k2 = elem.get_double_attribute('k2', 0.0)
    e1 = elem.get_double_attribute('e1', 0.0)
    e2 = elem.get_double_attribute('e2', 0.0)
    fint = elem.get_double_attribute('fint', 0.0)
    hgap = elem.get_double_attribute('hgap', 0.0)
    us_dipedge = None
    ds_dipedge = None
    cf = (k1 != 0.0) or (k2 != 0.0) or (k1s != 0.0)

    if e1 != 0.0:
        us_dipedge = impactx.elements.Dipedge(e1, radius_of_curvature, \
                                              hgap, fint, \
                                              name = nm+"_usedge")
        pass
    if e2 != 0.0:
        ds_dipedge = impactx.elements.Dipedge(-e2, radius_of_curvature, \
                                              hgap, fint, \
                                              name = nm+"_dsedge")
        pass

    if elem.get_double_attribute('h1') != 0.0:
        print('h1 attribute not supported for sbend in ImpactX')
    if elem.get_double_attribute('h2') != 0.0:
        print('h2 attribute not supported for sbend in ImpactX')

    if not cf:
        # normal bend, no higher order moments
        ds = length
        # What kind of screwy program accepts angles in degrees?
        phi = bendangle * 180/np.pi
        main_bend_elem = \
            impactx.elements.ExactSbend(ds, phi, nslice=1, name=nm)
        pass
    else:
        # CF bend
        if (k2 == 0.0):
            knormal = [1/radius_of_curvature, k1, k2]
            kskew = [0,0, k1s, 0.0]
        else:
            knormal = [1/radius_of_curvature, k1]
            kskew = [0,0, k1s]
        main_bend_elem = impactx.elements.ExactCFbend(ds=length, \
                                                      k_normal=knormal, \
                                                      k_skew = kskew, \
                                                      order=2, \
                                                      nslice=nslice_by_elem_type['sbend'], \
                                                      name=nm)
        
        pass
    
    # collect the pieces
    ixelem = []
    if us_dipedge:
        ixelem.append(us_depedge)
        pass
    ixelem.append(main_bend_elem)
    if ds_dipedge:
        ixelem.append(ds_dipedge)
        pass

    return(ixelem)

def cnv_rbend(elem):
    # the RBEND is converted to an SBEND with dipedge elements
    # before an after to get parallel faces

    bendangle = elem.get_bend_angle()
    straight_length = elem.get_length()
    # MAD-X RBEND langths is the straight length.
    radius_of_curvature = straight_length/(2*np.sin(0.5*bendangle))
    length = radius_of_curvature*bendangle

    nm = elem.get_name()
    k1 = elem.get_double_attribute('k1', 0.0)
    k1s = elem.get_double_attribute('k1s', 0.0)
    k2 = elem.get_double_attribute('k2', 0.0)
    e1 = elem.get_double_attribute('e1', 0.0)
    e2 = elem.get_double_attribute('e2', 0.0)
    fint = elem.get_double_attribute('fint', 0.0)
    hgap = elem.get_double_attribute('hgap', 0.0)
    us_dipedge = None
    ds_dipedge = None
    cf = (k1 != 0.0) or (k2 != 0.0) or (k1s != 0.0)

    if e1 != 0.0:
        us_dipedge = impactx.elements.Dipedge(e1, radius_of_curvature,
                                              hgap, fint,
                                              name = nm+"_usedge")
        pass
    if e2 != 0.0:
        ds_dipedge = impactx.elements.Dipedge(-e2, radius_of_curvature,
                                              hgap, fint,
                                              name = nm+"_dsedge")
        pass

    if elem.get_double_attribute('h1') != 0.0:
        print('h1 attribute not supported for sbend in ImpactX')
    if elem.get_double_attribute('h2') != 0.0:
        print('h2 attribute not supported for sbend in ImpactX')

    if not cf:
        # normal bend, no higher order moments
        ds = length
        # What kind of screwy program accepts angles in degrees?
        phi = bendangle * 180/np.pi
        main_bend_elem =  impactx.elements.ExactSbend(ds, phi,
                                                      nslice=1, name=nm)
        pass
    else:
        # CF bend
        if (k2 == 0.0):
            knormal = [1/radius_of_curvature, k1, k2]
            kskew = [0,0, k1s, 0.0]
        else:
            knormal = [1/radius_of_curvature, k1]
            kskew = [0,0, k1s]
        main_bend_elem = impactx.elements.ExactCFbend(ds=length,
                                                      k_normal=knormal,
                                                      k_skew = kskew,
                                                      order=2,
                                                      nslice=nslice_by_elem_type['sbend'],
                                                      name=nm)
        
        pass
    
    # collect the pieces
    ixelem = []
    if us_dipedge:
        ixelem.append(us_depedge)
        pass
    ixelem.append(main_bend_elem)
    if ds_dipedge:
        ixelem.append(ds_dipedge)
        pass

    return(ixelem)
def cnv_quadrupole(elem):
    ds = elem.get_length()
    k1 = elem.get_double_attribute('k1')
    nm = elem.get_name()
    return impactx.elements.ChrQuad(ds, k1, unit=0,
                                    nslice=nslice_by_elem_type['quadrupole'],
                                    name=nm)

def cnv_multipole(elem):
    # ImpactX multipole elements only have a single order each so
    # we have to peel each order and possibly create multiple elements to get the
    # full description.
    nm = elem.get_name()
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
                                                    name=f'{nm}_{order}'))
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

def cnv_sextupole(elem):
    L = elem.get_length()
    k2 = elem.get_double_attribute('k2')
    # The Booster lattice includes elements with the tilt attribute
    tilt = elem.get_double_attribute('tilt', 0.0)
    k2n = k2 * np.cos(3*tilt)
    k2s = -k2 * np.sin(3*tilt)
    knorm = np.array([0, 0, k2n])
    kskew = np.array([0, 0, k2s])

    sxelem = impactx.elements.ExactMultipole(ds=L, k_normal=knorm, \
                                             k_skew=kskew, order=4, \
                                             nslice=nslice_by_elem_type['sextupole'],
                                             name=elem.get_name()
                                             )
    return sxelem


# octupole follows similar logic as sextupole
def cnv_octupole(elem):
    L = elem.get_length()
    k3 = elem.get_double_attribute('k3')
    # The Booster lattice includes elements with the tilt attribute
    tilt = elem.get_double_attribute('tilt', 0.0)
    k3n = k2 * np.cos(4*tilt)
    k3s = -k2 * np.sin(4*tilt)
    knorm = np.array([0, 0, 0, k3n])
    kskew = np.array([0, 0, 0, k3s])

    ocelem = impactx.elements.ExactMultipole(ds=L, k_normal=knorm, \
                                             k_skew=kskew, order=4, \
                                             nslice=nslice_by_elem_type['octupole'],
                                             name=elem.get_name()
                                             )
    return ocelem

def syn2_to_impactx(lattice, init_monitor=True, final_monitor=True):
    # lattice must have a reference particle
    try:
        refpart = lattice.get_reference_particle()
    except:
        print("cannot get reference particle.")
        return None


    impactx_lattice = []

    # begin with a monitor element if requested
    if init_monitor:
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

    if final_monitor and init_monitor:
        # can't have final monitor if there was no initial monitor to define
        # the monitor element
        impactx_lattice.append(monitor)

    return impactx_lattice
    
