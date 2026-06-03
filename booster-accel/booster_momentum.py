#!/usr/bin/env python3

import numpy as np

f = 20 # Hz
w = 2 * np.pi * f

tmin = 725.0e-6 # t at Bmin [s]
tmax = 1/(2*f) + tmin # should be 0.025725
ke_inj = 800.0 # MeV
ke_max = 8000.0 # MeV
t_inj = 1000.0e-6 # time when energy is at ke_inj [s]
sn1000 = 0.5*(1-np.cos(w * (t_inj - tmin)))

class Booster_momentum:
    # According the Chandra's document, the Booster magnets reach a minumum
    # at t=725 us (offset timing, injection starts at 450 us). At minimum, the
    # energy is 798.1 MeV. At t=1000 us, the magnets reach 800 MeV. This is
    # supposed ramping at 20 Hz.

    def __init__(self, refpart):
        self.refpart = refpart
        self.mp = self.refpart.mass_MeV
        self.p1000 = self.KE_to_p(ke_inj)
        self.pmax = self.KE_to_p(ke_max)
        # p = pmin + (pmax-pmin)*sin(w* (t-t_inj))
 

        #print('sn1000: ', sn1000)

        # p1000 = pmin + (pmax - pmin) * sn1000
        # -pmin + pmin*sn1000 = pmax*sn1000 - p1000
        # pmin*(sn1000-1) = pmax*sn1000- p1000
        # pmin = [ pmax*sn1000 - p1000 ] / (sn1000 - 1)

        self.pmin = (self.pmax*sn1000 - self.p1000)/(sn1000-1)


    # Excluding hysteresis, the magnetic field and thus momentum of the Booster
    # oscillates at frequency f. We know the top and bottom KE so calculate the
    # momentum and use that as the limits of the oscillations.

    def KE_to_p(self, ke):
        # sqrt((ke + mp)**2 - mp**2)
        # = sqrt( ke**2 + 2*ke*mp + mp**2 - mp**2)
        p = np.sqrt(ke**2 + 2*ke*self.mp)
        return p

    # return momentum at given t
    def p_vs_t(self, t):
        return np.where(t<t_inj, self.p1000, self.pmin + (self.pmax-self.pmin)*0.5*(1-np.cos(w * (t-tmin))))

    def e_vs_t(self, t):
        p = self.p_vs_t(t)
        return np.sqrt(p**2 + self.mp**2)

    def ke_vs_t(self, t):
        p = self.p_vs_t(t)
        return np.sqrt(p**2 + self.mp**2)-self.mp

    def dpdt_vs_t(self, t):
        return np.where(t<t_inj, 0.0, 0.5*(self.pmax-self.pmin)*w*np.sin(w*(t-tmin)))

    def dEdt_vs_t(self, t):
        # dp/dt = 0.5*(pmax-pmin)*w*sin(w(t-tmin))
        # E = sqrt(p**2 + m**2)
        # dE/dt = (p/E) dp/dt
        return np.where(t<t_inj, 0.0, (self.p_vs_t(t)/self.e_vs_t(t)) * self.dpdt_vs_t(t))


def main():
    import impactx
    import matplotlib.pyplot as plt

    sim = impactx.ImpactX()
    sim.init_grids()

    refpart = sim.beam.ref
    refpart.set_species("proton")

    bmom = Booster_momentum(refpart)

    print('tmin: ', tmin)
    print('tmax: ', tmax)
    print('pmin: ', bmom.pmin)
    print('pmax: ', bmom.pmax)

    print('KE min: ', np.sqrt(bmom.pmin**2 + bmom.mp**2)-bmom.mp)

    t = np.arange(25000.0)/1.0e6 + 0.000725 # 0 - 0.025
    plt.title('KE vs. t')
    plt.plot(t, bmom.ke_vs_t(t))
    plt.xlabel('time [s]')
    plt.ylabel('KE [GeV]')

    plt.figure()
    plt.title('dp/dt vs. t')
    plt.plot(t, bmom.dpdt_vs_t(t))
    plt.xlabel('time [s]')
    plt.ylabel('dp/dt [GeV/s^2')

    plt.figure()
    plt.title('dE/dt vs. t')
    plt.plot(t, bmom.dEdt_vs_t(t))
    plt.xlabel('time [s]')
    plt.ylabel('dp/dt [GeV/s^2')

    plt.show()

    sim.finalize()

if __name__ == "__main__":
    main()
