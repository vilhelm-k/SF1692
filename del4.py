import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import Decimal, getcontext
from matplotlib.animation import FuncAnimation, PillowWriter

N=91

def RK4(f, y0, t):
    n = len(t)
    # if y0 is a scalar:
    if np.isscalar(y0):
        y = np.zeros(n)
    else:
        y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i]
        k1 = h*f(t[i], y[i])
        k2 = h*f(t[i] + h/2, y[i] + k1/2)
        k3 = h*f(t[i] + h/2, y[i] + k2/2)
        k4 = h*f(t[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6
    return y

f4 = lambda t, y: y**2

C = N/100
h = 1e-6
t = np.linspace(0, 1.5, int(1.5/h))
y = RK4(f4, C, t)
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')
t_last = t[(np.where(np.isfinite(y)))][-1]
plt.title(f'RK4 y(0) = 0.91, end={t_last:.6f}')
plt.show()