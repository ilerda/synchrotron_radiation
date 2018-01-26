"""
Program to find t_ret and w(t_ret) at any point given an arbitrary
trajectory.
"""

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib
from constants import *

#c = 299792458.0
NumIter = int(1e6)
dt = 1.0e-9
#Q = 1.60217662e-19
#epsilon = 8.854187817e-12
#mu = 4.0 * np.pi * 10.0 ** -7.0



def zero_func(r, t, tr, traj1,t_start): # will need to make sure that tr is already a grid
                        # here.
    traj_ret = traj1[np.array(np.rint((tr-t_start)/dt),dtype=int),1:]
    traj_ret = np.transpose(traj_ret,[2,0,1])
    return c * (t-tr) - np.linalg.norm(r - traj_ret, axis = 0)


def bisect(r, t, tup, tlow, traj1,t_start):
    middle = tlow + 0.5*(tup - tlow)
    maskarr = zero_func(r ,t, middle, traj1,t_start) < 0.0
    oppmask = np.logical_not(maskarr)
    np.putmask(tup,maskarr,middle)
    np.putmask(tlow,oppmask,middle)
    return tup, tlow


def bs_iterate(r, t, tup, tlow, traj1, t_start):
    max_iter = 60
    #tro = tr0
    for i in range(max_iter):
        tup, tlow = bisect(r, t, tup, tlow, traj1,t_start)
        tup = dt * np.ceil(tup/dt)
        tlow = dt * np.floor(tlow/dt)
        if np.array_equal(tup,tlow+dt):
            break
    trn = rev_interpolate(r,t,tup,tlow,traj1, t_start)
    return trn #tup, tlow


def interpolate(tr,hist, t_start):
    tl = np.floor(tr/dt) * dt
    tu = np.ceil(tr/dt) * dt
    lower = hist[np.array(np.rint((tl-t_start)/dt),dtype=int),1:]
    upper = hist[np.array(np.rint((tu-t_start)/dt),dtype=int),1:]
    lower = np.transpose(lower,[2,0,1])
    upper = np.transpose(upper,[2,0,1])
    return lower + (upper - lower) * (tr - tl) / (tu - tl)


def rev_interpolate(r,t,tup,tlow, traj1,t_zero):
    #tl = np.floor(tr/dt) * dt
    #tu = np.ceil(tr/dt) * dt
    yl = zero_func(r,t,tlow, traj1,t_zero)
    yu = zero_func(r,t,tup, traj1,t_zero)
    # check that yl and yu have opposite signs!
    xa = tlow - ((tup-tlow)/(yu-yl)) * yl
    return xa


def nibli(r,tr, traj1, t_start):
    return r - interpolate(tr,traj1, t_start)


def u_vel(r,tr,traj1,traj_v1, t_start): # need to build up trajectory of particle speeds!
    nib = nibli(r,tr,traj1, t_start)
    return (c*nib/np.linalg.norm(nib,axis=0))-interpolate(tr,traj_v1, t_start)


def E_field(r,tr,traj1,traj_v1,traj_a1, t_start): # disregard acceleration dependent term and run again
                    # then compare to see what difference the acceleration makes.
    const = Q / (4*np.pi*epsilon)
    nib = nibli(r,tr, traj1,t_start)
    u0 = u_vel(r,tr, traj1,traj_v1, t_start)
    dot_prod = nib[0]*u0[0]+nib[1]*u0[1]+nib[2]*u0[2]
    v = interpolate(tr,traj_v1, t_start)
    a = interpolate(tr,traj_a1, t_start)
    
    term1 = const * np.linalg.norm(nib,axis=0) / dot_prod**3.
    term2 = (c**2.-np.linalg.norm(v,axis=0)**2.)*u0
    term3 = np.cross(nib,np.cross(u0,a,axis=0),axis=0)
    return term1 * ( term3 +term2 )


def poynting(r,tr, traj1,traj_v1,traj_a1, t_start):
    nib = nibli(r,tr, traj1)
    nib_hat = nib/np.linalg.norm(nib,axis=0)
    E1 = E_field(r,tr, traj1,traj_v1,traj_a1, t_start)
    return (1.0/mu*c) * np.cross(E1,np.cross(nib_hat,E1,axis=0),axis=0)


