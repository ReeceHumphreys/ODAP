import numpy as np
import matplotlib.pyplot as plt
from numba import njit as jit, prange
from numpy import pi, sin, cos, sqrt
from scipy import integrate
from scipy.special import iv

import planetary_data as pd
import CoordTransforms as ct
import Aerodynamics as aero
from importlib import reload

from importlib import reload
reload(ct)

def null_perts():
    return {
        'J2': False,
        'aero': False,
        'moon_grav': False,
        'solar_grav': False

    }

class OrbitPropagator:

    def __init__(self, states0, A, M, tspan, dt, rv=False, cb=pd.earth, perts=null_perts()):

        if rv:
            self.states = 0

        else:
            self.states = states0

        # Setting the areas and masses
        self.A = A
        self.M = M

        # Integration information
        self.tspan = tspan
        self.dt = dt

        # Central body properties
        self.cb = cb

        # Defining perturbations being considered
        self.perts = perts

    def cartesian_representation(self):
    # Returns the cartesian state representation of states for vis. purposes
        N_t = self.states.shape[0]
        N_frag = self.states.shape[2]
        cartesian_states = np.empty(shape=(N_t, 2, N_frag, 3))

        for i in prange(self.states.shape[0]):
            cartesian_states[i, :, :] = ct.coe2rv_many_new(self.states[i, :, :])

        return cartesian_states


    def diffy_q(self, t, state):

        mu = self.cb['mu'] #[m^3 * s^-2]

        e, a, omega, Omega = state.reshape(4, len(self.A))

        dedt = np.zeros_like(e)
        dadt = np.zeros_like(a)
        domegadt = np.zeros_like(omega)
        dOmegadt = np.zeros_like(Omega)

        I_doom = a*(1-e) < self.cb['radius'] + 50*1e3 # 50 km above earth sats are doomed
        a[I_doom] = 0
        e[I_doom] = 0

        if self.perts['aero']:
            c_d = 2.2 # Drag coefficient
            drag_coef = (-(c_d * (self.A)) / self.M) #[m^2 * kg^-1]

            # Altitude
            z = a - self.cb['radius']

            # Air density
            rho = aero.atmosphere_density(z) #[kg * m^-3]
            #atm_density[np.argwhere(np.isnan(atm_density))] = 0 # Need to determine why some are NaN

            # dedt
            I       = (e>=0.001)
            dedt[I] = (drag_coef[I] * np.sqrt(mu / a[I]) * rho[I])
            I       = I & (e<0.01)
            x       = (a[I] * e[I]) / aero.scale_height(z[I])
            dedt[I] *= iv(1,x) + (e[I]/2)*(iv(0,x) + iv(2,x))

            # dadt
            dadt    = drag_coef * np.sqrt(mu * a) * rho
            x       = (a * e) / aero.scale_height(z)
            I       = (e>=0.001) & (e < 0.01)
            dadt[I] *= (iv(0,x[I]) + 2*e[I]*iv(1,x[I]))
            I       = (e >= 0.01)
            dadt[I] *= iv(0, x[I]) + 2*e[I]*iv(1, x[I]) + (3/4)*e[I]**2*(iv(0, x[I]) + iv(2, x[I])) + (e[I]**3/4)*(3*iv(1, x[I]) + iv(3, x[I]))

            dedt[np.argwhere(np.isnan(dedt))] = 0
            dadt[np.argwhere(np.isnan(dadt))] = 0

        if self.perts['J2']:
            # Semi-latus rectum
            p = a * (1 - e**2)

            # Mean motion
            n = np.zeros_like(a)
            I = a > 0
            n[I] = np.sqrt(self.cb['mu'] / a[I]**3)

            # Inclination, assuming constant for now, maybe will change
            i = self.states[-1, 2, :]

            base = (3/2) * self.cb['J2'] * (self.cb['radius']**2/p**2) * n

            domegadt = base * (2 - (5/2)*sin(i)**2)
            dOmegadt = -base * cos(i)

            domegadt[np.argwhere(np.isnan(domegadt))] = 0
            dOmegadt[np.argwhere(np.isnan(dOmegadt))] = 0

        y0 = np.concatenate((dedt, dadt, domegadt, dOmegadt))
        return y0


    def propagate_perturbations(self):

        reload(aero)
        # Need to introduce notion of randomizing the positions of the fragments as they dont matter once we start
        # introducing orbital perturbations
        times  = np.arange(self.tspan[0], self.tspan[-1], self.dt)

        e0 = self.states[-1, 1, :]
        a0 = self.states[-1, 0, :]
        omega0 = self.states[-1, 4, :]
        Omega0 = self.states[-1, 3, :]
        y0     = np.concatenate((e0, a0, omega0, Omega0))

        print('Initializing perturbations with the following effects:')
        if self.perts['aero']:
            print('Aerodynamic perturbations ...')
        if self.perts['J2']:
            print('J2 perturbations ...')

        output = integrate.solve_ivp(self.diffy_q, [times[0], times[-1]], y0, t_eval = times, vectorized=True)

        # Unpacking output
        de = output.y[0:len(self.A), :]
        da = output.y[len(self.A):2*len(self.A), :]
        domgea = output.y[2*len(self.A):3*len(self.A), :]
        dOmega = output.y[3*len(self.A):output.y.shape[0], :]

        return de, da, domgea, dOmega

    def propagate_orbit(self):

        times      = np.arange(self.tspan[0], self.tspan[-1], self.dt)

        # Mean anomaly rate of change
        M_dt       = sqrt(self.cb['mu']/self.states[0, :]**3)

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
