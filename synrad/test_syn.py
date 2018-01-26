"""
Program to find t_ret and w(t_ret) at any point given an arbitrary
trajectory.
"""

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib
from synchrotron_bisection import *
from constants import *


t_start1 = 0.0

stretch = 3000.0
r1 = np.mgrid[0:1, -5.0:5.0:500j, -5.0:5.0:500j]
r1 = r1[:, 0, :, :]
r1 = r1[::-1, :, :]
r1 = stretch * r1
r2 = np.mgrid[0:1, -5.0:5.0:10j, -5.0:5.0:10j]
r2 = r2[:, 0, :, :]
r2 = r2[::-1, :, :]
r2 = stretch * r2

radius = 2803.95  # bending radius at the LHC
v_lin = 0.95 * c

traj = np.zeros([NumIter, 4])
traj[:, 0] = np.linspace(0.0, NumIter * dt, num=NumIter, endpoint=False)
traj[:, 1] = radius * np.cos(v_lin * traj[:, 0] / radius)
traj[:, 2] = radius * np.sin(v_lin * traj[:, 0] / radius)

# Make a list of particle velocities and times (and accelerations).
traj_v = np.zeros([NumIter, 4])
traj_a = np.zeros([NumIter, 4])
traj_v[:, 0] = np.linspace(0.0, NumIter * dt, num=NumIter, endpoint=False)
traj_a[:, 0] = np.linspace(0.0, NumIter * dt, num=NumIter, endpoint=False)
traj_v[:, 1] = (-1.0) * v_lin * np.sin(v_lin * traj[:, 0] / radius)
traj_v[:, 2] = v_lin * np.cos(v_lin * traj[:, 0] / radius)
traj_a[:, 1] = (-1) * (v_lin ** 2 / radius) * np.cos(v_lin * traj[:, 0] / radius)
traj_a[:, 2] = (-1) * (v_lin ** 2 / radius) * np.sin(v_lin * traj[:, 0] / radius)

tnow = 7.0e-4  # 8.11732e-4 #<- to check this at t_ret[350,260]
lowertime = tnow - 10.0 * stretch * np.sqrt(2.0) / c

t0up = np.ones((500, 500)) * tnow  # (tnow+1.0e-4
t0low = np.ones((500, 500)) * lowertime
t02up = np.ones((10, 10)) * tnow
t02low = np.ones((10, 10)) * lowertime
t_ret = bs_iterate(r1, tnow, t0up, t0low, traj,t_start1)
t_ret2 = bs_iterate(r2, tnow, t02up, t02low, traj,t_start1)

final = E_field(r1, t_ret, traj, traj_v, traj_a,t_start1)
magnitudes = np.linalg.norm(final, axis=0)
final2 = E_field(r2, t_ret2, traj, traj_v, traj_a,t_start1)
magnitudes2 = np.linalg.norm(final2, axis=0)
directions = final2 / magnitudes2
current_pos = traj[np.rint((tnow - t_start1) / dt).astype(int), 1:] * 0.001
print(current_pos)
r2 = r2 * 0.001
stretch = stretch * 0.001

font = {'style': 'normal',
        'weight': 'bold',
        'size': 18}

matplotlib.rc('font', **font)
pyplot.figure()
# pyplot.title('Charged Particle moving at temporary a - radiation fields.')
pyplot.xlabel('x-position / 10^3 metres')
pyplot.ylabel('y-position / 10^3 metres')
pyplot.axis('equal')
pyplot.imshow(magnitudes, extent=(-stretch * 5.0, stretch * 5.0, -stretch * 5.0, stretch * 5.0),
              origin='lower', vmin=0.0, vmax=5.0e-16, cmap='jet')
pyplot.colorbar(orientation='vertical')
pyplot.quiver(r2[0], r2[1], directions[0], directions[1], pivot='mid')

# pyplot.xticks(np.arange(-15000.0, 20000.0, 10000.0))

pyplot.plot(current_pos[0], current_pos[1], marker='o')
pyplot.show()
