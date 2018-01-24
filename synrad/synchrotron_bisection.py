"""
Program to find t_ret and w(t_ret) at any point given an arbitrary
trajectory.
"""

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib
from constants import *

c = 299792458.0
NumIter = int(1e6)
dt = 1.0e-9
Q = 1.60217662e-19
epsilon = 8.854187817e-12
mu = 4.0 * np.pi * 10.0 ** -7.0

t_start1 = 0.0


def zero_func(r, t, tr, traj1): # will need to make sure that tr is already a grid
                        # here.
    traj_ret = traj1[np.array(np.rint((tr-t_start1)/dt),dtype=int),1:]
    traj_ret = np.transpose(traj_ret,[2,0,1])
    return c * (t-tr) - np.linalg.norm(r - traj_ret, axis = 0)


def bisect(r, t, tup, tlow, traj1):
    middle = tlow + 0.5*(tup - tlow)
    maskarr = zero_func(r ,t, middle, traj1) < 0.0
    oppmask = np.logical_not(maskarr)
    np.putmask(tup,maskarr,middle)
    np.putmask(tlow,oppmask,middle)
    return tup, tlow


def bs_iterate(r, t, tup, tlow, traj1):
    max_iter = 60
    #tro = tr0
    for i in range(max_iter):
        tup, tlow = bisect(r, t, tup, tlow, traj1)
        tup = dt * np.ceil(tup/dt)
        tlow = dt * np.floor(tlow/dt)
        if np.array_equal(tup,tlow+dt):
            break
    trn = rev_interpolate(r,t,tup,tlow,traj1)
    return trn #tup, tlow


def interpolate(tr,hist):
    tl = np.floor(tr/dt) * dt
    tu = np.ceil(tr/dt) * dt
    lower = hist[np.array(np.rint((tl-t_start1)/dt),dtype=int),1:]
    upper = hist[np.array(np.rint((tu-t_start1)/dt),dtype=int),1:]
    lower = np.transpose(lower,[2,0,1])
    upper = np.transpose(upper,[2,0,1])
    return lower + (upper - lower) * (tr - tl) / (tu - tl)


def rev_interpolate(r,t,tup,tlow, traj1):
    #tl = np.floor(tr/dt) * dt
    #tu = np.ceil(tr/dt) * dt
    yl = zero_func(r,t,tlow, traj1)
    yu = zero_func(r,t,tup, traj1)
    # check that yl and yu have opposite signs!
    xa = tlow - ((tup-tlow)/(yu-yl)) * yl
    return xa


def nibli(r,tr, traj1):
    return r - interpolate(tr,traj1)


def u_vel(r,tr,traj1): # need to build up trajectory of particle speeds!
    nib = nibli(r,tr,traj1)
    return (c*nib/np.linalg.norm(nib,axis=0))-interpolate(tr,traj_v)


def E_field(r,tr,traj1): # disregard acceleration dependent term and run again
                    # then compare to see what difference the acceleration makes.
    const = Q / (4*np.pi*epsilon)
    nib = nibli(r,tr, traj1)
    u0 = u_vel(r,tr, traj1)
    dot_prod = nib[0]*u0[0]+nib[1]*u0[1]+nib[2]*u0[2]
    v = interpolate(tr,traj_v)
    a = interpolate(tr,traj_a)
    
    term1 = const * np.linalg.norm(nib,axis=0) / dot_prod**3.
    term2 = (c**2.-np.linalg.norm(v,axis=0)**2.)*u0
    term3 = np.cross(nib,np.cross(u0,a,axis=0),axis=0)
    return term1 * ( term3 +term2 )


def poynting(r,tr, traj1):
    nib = nibli(r,tr, traj1)
    nib_hat = nib/np.linalg.norm(nib,axis=0)
    E1 = E_field(r,tr, traj1)
    return (1.0/mu*c) * np.cross(E1,np.cross(nib_hat,E1,axis=0),axis=0)


stretch= 3000.0
r1 = np.mgrid[0:1,-5.0:5.0:500j,-5.0:5.0:500j]
r1 = r1[:,0,:,:]
r1 = r1[::-1,:,:]
r1 = stretch*r1
r2 = np.mgrid[0:1,-5.0:5.0:10j,-5.0:5.0:10j]
r2 = r2[:,0,:,:]
r2 = r2[::-1,:,:]
r2 = stretch*r2

radius = 2803.95 # bending radius at the LHC
v_lin = 0.95 * c    
    
traj = np.zeros([NumIter,4])
traj[:,0] = np.linspace(0.0,NumIter*dt,num=NumIter,endpoint=False)
traj[:,1] = radius * np.cos(v_lin*traj[:,0] / radius)
traj[:,2] = radius * np.sin(v_lin*traj[:,0] / radius)

# Make a list of particle velocities and times (and accelerations).
traj_v = np.zeros([NumIter,4])
traj_a = np.zeros([NumIter,4])
traj_v[:,0] = np.linspace(0.0,NumIter*dt,num=NumIter,endpoint=False)
traj_a[:,0] = np.linspace(0.0,NumIter*dt,num=NumIter,endpoint=False)
traj_v[:,1] = (-1.0)*v_lin * np.sin(v_lin*traj[:,0] / radius)
traj_v[:,2] = v_lin * np.cos(v_lin*traj[:,0] / radius)
traj_a[:,1] = (-1)*(v_lin**2/radius) * np.cos(v_lin*traj[:,0] / radius)
traj_a[:,2] = (-1)*(v_lin**2/radius) * np.sin(v_lin*traj[:,0] / radius)

tnow = 7.0e-4 #8.11732e-4 #<- to check this at t_ret[350,260]
lowertime = tnow - 10.0*stretch*np.sqrt(2.0) / c

t0up = np.ones((500,500))*tnow #(tnow+1.0e-4
t0low = np.ones((500,500))*lowertime
t02up = np.ones((10,10))*tnow
t02low = np.ones((10,10))*lowertime
t_ret = bs_iterate(r1,tnow,t0up,t0low, traj)
t_ret2 = bs_iterate(r2,tnow,t02up,t02low, traj)


final = E_field(r1,t_ret,traj)
magnitudes = np.linalg.norm(final, axis=0)
final2 = E_field(r2,t_ret2,traj)
magnitudes2 = np.linalg.norm(final2, axis=0)
directions = final2 / magnitudes2
current_pos = traj[np.rint((tnow-t_start1)/dt).astype(int), 1:]*0.001
print(current_pos)
r2 = r2*0.001
stretch = stretch * 0.001

font = {'style' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
pyplot.figure()
#pyplot.title('Charged Particle moving at temporary a - radiation fields.')
pyplot.xlabel('x-position / 10^3 metres')
pyplot.ylabel('y-position / 10^3 metres')
pyplot.axis('equal')
pyplot.imshow(magnitudes, extent=(-stretch*5.0,stretch*5.0,-stretch*5.0,stretch*5.0),
              origin = 'lower',vmin = 0.0,vmax = 5.0e-16,cmap='jet')
pyplot.colorbar(orientation = 'vertical')
pyplot.quiver(r2[0],r2[1],directions[0],directions[1],pivot='mid')

#pyplot.xticks(np.arange(-15000.0, 20000.0, 10000.0))

pyplot.plot(current_pos[0],current_pos[1],marker='o')
pyplot.show()


