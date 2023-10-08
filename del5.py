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

f5 = lambda t, y: y

k = 6
max = 2**k
h = 1e-3
t = np.linspace(0, max, int(max/h)+1)
y = RK4(f5, N, t)

for i in range(k+1):
    y_num = y[int(2**i/h)]
    y_an = N*math.exp(2**i)
    if abs(y_num - y_an) < 0.005:
        print(f'i={i}, y(2^{i})={y_num}')

