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

def f8(t, y):
    G = 4*math.pi**2
    M_sun = 1
    M_jupiter = 1/1047.945205479452
    M_earth = 1/332946.0483870968
    sun_pos = np.array([0, 0])
    jupiter_pos = y[0:2]
    earth_pos = y[2:4]
    rocket_pos = y[4:6]
    # jupiter, earth, rocket
    return np.concatenate((
        y[6:8], # = jupiter vel
        y[8:10], # = earth vel
        y[10:12], # = rocket vel
        -G*(f_gravity_vector(jupiter_pos, sun_pos, M_sun) + f_gravity_vector(jupiter_pos, earth_pos, M_earth)),  # = jupiter acc
        -G*(f_gravity_vector(earth_pos, sun_pos, M_sun) + f_gravity_vector(earth_pos, jupiter_pos, M_jupiter)), # = earth acc
        -G*(f_gravity_vector(rocket_pos, sun_pos, M_sun) + f_gravity_vector(rocket_pos, jupiter_pos, M_jupiter) + f_gravity_vector(rocket_pos, earth_pos, M_earth)) # rocket acc
    ))
    
GM = 4 * math.pi**2
earth_pos = np.array([1, 0])
earth_vel = np.array([0, np.sqrt(GM/np.linalg.norm(earth_pos))])
rocket_pos = np.array([1.01, 0])
rocket_vel_magnitude = 0.95*np.sqrt(2*GM/np.linalg.norm(rocket_pos))
rocket_direction = np.array([0, 1])
rocket_vel = rocket_vel_magnitude*rocket_direction/np.linalg.norm(rocket_direction)

jupiter_orbit_T = 11.857824421031035
pass_t = 1.439
p = [-4.193889869202012, 3.0756470115284253]
g_assist_theta = np.arctan2(p[1], p[0])
theta = g_assist_theta - 2*math.pi*pass_t/jupiter_orbit_T
jupiter_pos = 5.2*np.array([np.cos(theta), np.sin(theta)])
print(jupiter_pos)
jupiter_vel_magnitude = np.sqrt(GM/np.linalg.norm(jupiter_pos))
jupiter_vel = jupiter_vel_magnitude * np.array([-np.sin(theta), np.cos(theta)])

n = 10000

# same step size. used for when jupiter is in narnia
shit_jupiter_pos = np.array([0, 5.2])
shit_jupiter_vel = np.array([-jupiter_vel_magnitude, 0])
y0 = np.concatenate((shit_jupiter_pos, earth_pos, rocket_pos, shit_jupiter_vel, earth_vel, rocket_vel))
t = np.linspace(0, 12, n)
y = RK4(f8, y0, t)

# smaller step size around pass, used for real run
y0 = np.concatenate((jupiter_pos, earth_pos, rocket_pos, jupiter_vel, earth_vel, rocket_vel))
d = 0.01
t_close_start = pass_t - d
t_close_end = pass_t + d

t = np.linspace(0, t_close_start, n)
y = RK4(f8, y0, t)

y1 = y[-1]
t_close = np.linspace(t_close_start, t_close_end, n)
y_close = RK4(f8, y1, t_close)

y2 = y_close[-1]
t_rest = np.linspace(t_close_end, 5, n)
y_rest = RK4(f8, y2, t_rest)

t = np.concatenate((t, t_close, t_rest))
y = np.concatenate((y, y_close, y_rest))

# first value where rocket is morem than 5.2 AU from sun
# used to initially set the position of jupiter. 
# ran once, then updated jupiter_pos and ran again
mask = np.where(np.sqrt(y[:,4]**2+y[:,5]**2) > 5.2)
t_rocket_crosses_orbit = t[mask][0]
pass_pos = y[mask][0, 4:6]
print('t_rocket_crosses_orbit', t_rocket_crosses_orbit)
print('pass_pos', list(pass_pos))

# the closest point between the rocket and jupiter.
# used this to update jupiter's position twice in the same way as above
rocket_jupiter_diff = y[:,4:6] - y[:,0:2]
min_dist = np.min(np.linalg.norm(rocket_jupiter_diff, axis=1))
idx = np.argmin(np.linalg.norm(rocket_jupiter_diff, axis=1))
pos_mindist = y[idx, 4:6]
print('min_dist', min_dist)
print('idx', idx)
print('t_mindist', t[idx])
print('pos_mindist', list(pos_mindist))

force = -1*f_gravity_vector(y[idx, 4:6], y[idx, 0:2], 1/1047.945205479452)
force_sun = -1*f_gravity_vector(y[idx, 4:6], np.array([0, 0]), 1)
print('max_force_jupiter', force)

# final velocity of the rocket in the y direction
final_vel = y[-1, 10:12]
final_vel_y = final_vel[1]
final_pos = y[-1, 4:6]
escape_vel_final_pos = np.sqrt(2*GM/np.linalg.norm(final_pos))
print('final_vel', list(final_vel))
print('final_pos', list(final_pos))
print('final_vel_y', final_vel_y)
projected_vel = np.dot(final_vel, final_pos)/np.linalg.norm(final_pos)**2 * final_pos
print('projected_vel', list(projected_vel))
print('abs(projected_vel)', np.linalg.norm(projected_vel))
print('escape_vel_final_pos', escape_vel_final_pos)

plt.plot(y[:,0], y[:,1], label='Jupiter')
plt.plot(y[:,2], y[:,3], label='Earth')
plt.plot(y[:,4], y[:,5], label='Rocket')
plt.plot(0, 0, 'o', label='Sun')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Orbits without gravity assist')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.axis('equal')
plt.show()

y_subsample = y[idx-1000:idx+1000:10]
fig, ax = plt.subplots()
ax.plot(0, 0, 'o', label='Sun')
ln1, = ax.plot([], [], 'r-', label='Jupiter')
ln2, = ax.plot([], [], 'g-', label='Earth')
ln3, = ax.plot([], [], 'b-', label='Rocket')


d = 0.001
def init():
    ax.set_xlim(pos_mindist[0]-d, pos_mindist[0]+d)
    ax.set_ylim(pos_mindist[1]-d, pos_mindist[1]+d)
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_title('Rocket close to Jupiter')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ln1, ln2, ln3,

def update(frame):
    ln1.set_data(y_subsample[:frame, 0], y_subsample[:frame, 1])
    ln2.set_data(y_subsample[:frame, 2], y_subsample[:frame, 3])
    ln3.set_data(y_subsample[:frame, 4], y_subsample[:frame, 5])
    return ln1, ln2, ln3,

ani = FuncAnimation(fig, update, frames=len(y_subsample), init_func=init, blit=True)
ani.save("newton2.1.gif", writer=PillowWriter(fps=10))