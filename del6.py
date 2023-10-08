import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import Decimal, getcontext
from matplotlib.animation import FuncAnimation, PillowWriter

N=91

def RK4_0(f, y0, h):
    # RK4 which ends at y = 0. We want to find t at this point.
    # if
    getcontext().prec = 40
    y = np.array([Decimal(str(x)) for x in y0], dtype=object)
    t = Decimal(0)
    while True:
        k1 = h*f(t, y)
        k2 = h*f(t + h/2, y + k1/2)
        k3 = h*f(t + h/2, y + k2/2)
        k4 = h*f(t + h, y + k3)
        y_new = y + (k1 + 2*k2 + 2*k3 + k4)/6
        t_new = t + h
        if y_new[0] < 0:
            if abs(y_new[0]) < Decimal('1e-30'):
                # use linear interpolation between last two points
                return t_new
            h /= 2
            continue
        y = y_new
        t = t_new

f6 = lambda t, y: np.array([-y[1], y[0]])

t = RK4_0(f6, [1, 0], Decimal('1e-5'))
pi_est = 2*t
pi = Decimal('3.141592653589793238462643383279')
# want it correct to 20 decimal places
est_str = f'{pi_est:.22f}'
pi_str = f'{pi:.22f}'
print(est_str)
print(pi_str)