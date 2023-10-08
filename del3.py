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

f3 = lambda t, y: math.sqrt(abs(y))

h = 0.1
t = np.linspace(0, 10, int(10/h))
y = RK4(f3, 0, t)
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')
plt.title('RK4 y(0) = 0')

# t, y = RK4(f3, -1, -1, 9, 1e-3)
h = 1e-3
t = np.linspace(-1, 9, int(10/h))
y = RK4(f3, -1+0.999999999999, t)
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')
plt.title('RK4 y(-1) = -1')

plt.show()