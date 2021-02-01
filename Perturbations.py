import numpy as np
import matplotlib.pyplot as plt
from numba import njit as jit, prange
from numpy import pi, sin, cos, sqrt

import planetary_data as pd
import CoordTransforms as ct

from importlib import reload
reload(ct)


class OrbitPropagator:
    
    def __init__(self, states0, tspan, dt, rv=False, cb=pd.earth):
        
        if rv:
            self.states = 0
            
        else:
            self.states = states0
           
        self.tspan = tspan
        self.dt = dt
        self.cb = cb
    
    def cartesian_representation(self):
    # Returns the cartesian state representation of states for vis. purposes
        N_t = self.states.shape[0]
        N_frag = self.states.shape[2]
        cartesian_states = np.empty(shape=(N_t, 2, N_frag, 3))
        
        for i in prange(self.states.shape[0]):
            cartesian_states[i, :, :] = ct.coe2rv_many_new(self.states[i, :, :])
        
        return cartesian_states
    
    def propagate_orbit(self):
       
        mu = 398600.4418 #km^3s^-2
        
        times      = np.arange(self.tspan[0], self.tspan[-1], self.dt)
       
        # Mean anomaly rate of change
        M_dt       = sqrt(mu/self.states[0, :]**3)

        Nd         = len(M_dt)
        Nt         = len(times)

        # Mean anomaly over time
        M_t        = np.deg2rad(self.states[5, :, None]) + M_dt[:, None] * times[None, :]
        M_t        = np.rad2deg(np.mod(M_t, 2*pi))

        # Eccentric anomaly over time. Note need to use E_t in rad, thus convert to deg after using it in
        # x1 and x2
        E_t        = np.empty(shape=(Nd, Nt), dtype=np.float32)
        E_t        = M2E(self.states[1], np.deg2rad(M_t))

        x1         = sqrt(1 + self.states[1, :])[:, None] * sin(E_t / 2)
        x2         = sqrt(1 - self.states[1, :])[:, None] * cos(E_t / 2)
        E_t        = np.rad2deg(E_t)

        # True anomaly over time
        nu_t       = (2*np.arctan2(x1, x2) % (2*pi))
        nu_t       = np.rad2deg(nu_t).T
        
        n_times    = nu_t.shape[0]
        states     = np.empty(shape = (n_times, self.states.shape[0], self.states.shape[1]))
        
        for i in prange(n_times):
            state = self.states.copy()
            state[6, :] = nu_t[i, :]
            states[i] = state
        
        self.states = states

        
        
# Modified from OrbitalPy.utilities
@jit(parallel=True, fastmath=True)
def M2E(e_deb, M_t, tolerance=1e-14):
#Convert mean anomaly to eccentric anomaly.
#Implemented from [A Practical Method for Solving the Kepler Equation][1]
#by Marc A. Murison from the U.S. Naval Observatory
#[1]: http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
    n_deb = M_t.shape[0]
    n_times = M_t.shape[1]

    E_t = np.empty_like(M_t)

    for i in prange(n_deb):
        e = e_deb[i]
        for j in prange(n_times):
            M = M_t[i, j]

            MAX_ITERATIONS = 100
            Mnorm = np.mod(M, 2 * pi)
            E0 = M + (-1 / 2 * e ** 3 + e + (e ** 2 + 3 / 2 * cos(M) * e ** 3) * cos(M)) * sin(M)
            dE = tolerance + 1
            count = 0
            while dE > tolerance:
                t1 = cos(E0)
                t2 = -1 + e * t1
                t3 = sin(E0)
                t4 = e * t3
                t5 = -E0 + t4 + Mnorm
                t6 = t5 / (1 / 2 * t5 * t4 / t2 + t2)
                E = E0 - t5 / ((1 / 2 * t3 - 1 / 6 * t1 * t6) * e * t6 + t2)
                dE = np.abs(E - E0)
                E0 = E
                count += 1
                if count == MAX_ITERATIONS:
                    print('Did not converge, increase number of iterations')
            E_t[i, j] = E
    return E_t







