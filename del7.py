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

def f_gravity_vector(x_own, x_body, body_mass):
    diff_vector = x_own-x_body
    return body_mass * diff_vector/np.linalg.norm(diff_vector)**3

def f7(t, y):
    G = 1
    return np.concatenate((
        y[2:], 
        -G*f_gravity_vector(y[:2], np.array([0, 0]), 1)
        ))

def semi_implicit_euler(f, y0, t):
    n = len(t)
    m = len(y0)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i]
        dv = h*f(t[i], y[i])[m//2:]
        y[i+1][m//2:] = y[i][m//2:] + dv
        y[i+1][:m//2] = y[i][:m//2] + h*y[i+1][m//2:]
    return y

y0 = np.array([1, 0.5, -1, 0])  
# B0
tau0 = np.linspace(0, 1000, 10000)
y = RK4(f7, y0, tau0)
plt.plot(y[:,0], y[:,1], label = 'B0: h = 0.1')

# B1
tau1 = np.linspace(0, 1000, 20000)
y = RK4(f7, y0, tau1)
plt.plot(y[:,0], y[:,1], label = 'B1: h = 0.05')

# B2
tau2 = np.linspace(0, 1000, 100000)
y = RK4(f7, y0, tau2)
plt.plot(y[:,0], y[:,1], label = 'B3: h = 0.01')

# B3
y = semi_implicit_euler(f7, y0, tau2)
plt.plot(y[:,0], y[:,1], label = 'B3: h = 0.05')

plt.title('y0 = [1, 0.5, -1, 0], t = [0, 10], G=M=1')
plt.plot(y0[0], y0[1], 'o', label='Start')
plt.plot(0, 0, 'o', label='Solen')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(-0.5, 2.5)
plt.ylim(-0.7, 0.7)

plt.show()