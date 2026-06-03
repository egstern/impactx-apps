#!/usr/bin/env python

import sys, os
import numpy as np
from scipy import interpolate



# The Booster RF ramp looks approximately like:

# time [s], voltage [MV]
# x,Curve1
# 0.0003652,0.20466 vstep 1
# 0.0008989,0.20078 vstep 2
# 0.0012079,0.99955 ... etc
# 0.0019944,1.10259
# 0.01,1.09963
# 0.0128933,0.83842
# 0.0132022,0.83646
# 0.0140449,1.09814
# 0.0171067,1.09701
# 0.0201404,0.96489
# 0.0226685,0.64477
# 0.0252528,1.09216
# 0.0255618,1.08835
# 0.0257022,0.15841


class PIP2ramp:
    # there are 14 vsteps with the final end point
    # the voltages are all in MV but all getter routines will
    # return GV
    rf_times_and_volts = np.array([ [0.000365, 0.2],  # start injection
                          [0.0009, 0.2],  # ramp for squeeze
                          [0.0012, 1.0],   # continue squeese
                          [0.0020, 1.1],  # start acceleration
                          [0.01, 1.1],    # start rampdown for transition
                          #[0.01280, 0.84], # near transition (original ramp)
                          #[0.01316, 0.84],  # past transition, start ramp up (original ramp)
                          [0.011723077, 0.94], # lowered RF at transition (pretty good)
                          [0.013483077, 0.94], # lowered RF at transition (pretty good)

                          # [0.011076923, 1.0], # 
                          # [0.013676923, 1.0], # new ramp
                          #[0.011723077, 1.1], # constant RF at transition (bad)
                          #[0.013483077, 1.1], # constant RF at transition (bad)
                          [0.01400, 1.1],  # continue acceleration after transition
                          [0.01700, 1.1],   # begin rampdown preparing for bunch rotation
                          [0.02000, 0.97],  #
                          [0.02250, 0.65],   # increase dE prepare for bunch rotation
                          [0.025085, 1.1],   # end of dE increase
                          [0.025315, 1.1],  # hold dE near max until this time
                          [0.025400, 0.3],
                          [0.025488, 0.17]])  # Drop voltage for snap RF rotation
                          

    def __init__(self):
        self.volt_interp = interpolate.interp1d(
            self.rf_times_and_volts[:, 0], self.rf_times_and_volts[:, 1], kind="linear",
            fill_value="extrapolate")
        pass

    
    # CBhat time injection starts at 0.000450 continues to 0.001000.
    # Booster magnet minimum occors at 0.000725.
    # The sin wave of the magnet current is aligned so that beam kinetic energy
    # of 0.8 GeV occurs at t=0.001.

    # return RF voltage at particular time given in seconds
    def get_rf_voltage_by_time(self, time):
        if time < self.rf_times_and_volts[0, 0]:
            return self.rf_times_and_volts[0, 1]
        # if time exceeds tabulated values, return last voltage
        if time >= self.rf_times_and_volts[-1, 0]:
            return self.rf_times_and_volts[-1, 1]
        return self.volt_interp(time)


def main():
    import matplotlib.pyplot as plt
    pip2ramp = PIP2ramp()
    t = 0.0255*np.arange(1000)/1000
    v = np.array([pip2ramp.get_rf_voltage_by_time(tt) for tt in t])
    plt.plot(t, v)
    plt.xlabel('time [s]')
    plt.ylabel('voltage [MV]')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
