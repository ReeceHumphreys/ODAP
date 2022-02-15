import numpy as np
from numba import njit as jit, prange
from numpy import pi, sin, cos, sqrt
from scipy import integrate
from scipy.special import iv

# User defined libearayr
import planetary_data as pd
import CoordTransforms as ct
import Aerodynamics as aero


def null_perts():
    return {
        'J2': False,
        'aero': False,
        'moon_grav': False,
        'solar_grav': False
    }


class OrbitPropagator:

    def __init__(self, states0, A, M, tspan, dt, rv=False, cb=pd.earth, perts=null_perts()):

        # Need to add support for initializing with radius and velocity
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

        # Defining constants for aerodynamic drag
        if self.perts['aero']:
            self.K_a = np.matrix([[1, 0, 0, 0, 0, 0, 0],
                                  [0, 2, 0, 0, 0, 0, 0],
                                  [3/4, 0, 3/4, 0, 0, 0, 0],
                                  [0, 3/4, 0, 1/4, 0, 0, 0],
                                  [21/64, 0, 28/64, 0, 7/64, 0, 0],
                                  [0, 30/64, 0, 15/64, 0, 3/64, 0]])

            self.K_e = np.matrix([[0, 1, 0, 0, 0, 0, 0],
                                  [1/2, 0, 1/2, 0, 0, 0, 0],
                                  [0, -5/8, 0, 1/8, 0, 0, 0],
                                  [-5/16, 0, -4/16, 0, 1/16, 0, 0],
                                  [0, -18/128, 0, -1/128, 0, 3/128, 0],
                                  [-18/256, 0, -19/256, 0, 2/256, 0, 3/256]])

    def cartesian_representation(self):
        # Returns the cartesian state representation of states for vis. purposes
        N_t = self.states.shape[0]
        N_frag = self.states.shape[2]
        cartesian_states = np.empty(shape=(N_t, 2, N_frag, 3))

        for i in prange(self.states.shape[0]):
            cartesian_states[i, :, :] = ct.coe2rv_many_new(
                self.states[i, :, :])

        return cartesian_states

    def diffy_q(self, t, state):
        e, a, i, Omega, omega = state.reshape(5, len(self.A))
        N_f = len(self.A)

        # Central body information
        mu = self.cb['mu']
        radius = self.cb['radius']  # [m]
        J2 = self.cb['J2']

        # Local variables
        delta_e = np.zeros_like(e)
        delta_a = np.zeros_like(a)
        delta_i = np.zeros_like(i)
        delta_Omega = np.zeros_like(Omega)
        delta_omega = np.zeros_like(omega)

        # Current orbital information
        peri = a * (1 - e)  # [m]
        p = a * (1 - e**2)  # [m] (Semi parameter)
        n = np.sqrt(mu / a**3)  # (Mea motion)

        ############### Drag effects ###############
        if self.perts['aero']:
            h_p = (peri - radius)  # [m]
            rho = aero.atmosphere_density(h_p/1e3)  # [kg * m^-3]
            H = aero.scale_height(h_p/1e3) * 1e3  # [m]

            z = a*e / H
            Cd = 0.7
            tilt_factor = 1
            delta = Cd * (self.A[0] * tilt_factor) / self.M[0]

            e_T = np.array([np.ones_like(e), e, e**2, e**3, e**4, e**5])
            I_T = np.array([iv(i, z) for i in range(7)])
            k_a = delta * np.sqrt(mu * a) * rho
            k_e = k_a / a

            delta_e = np.zeros_like(e)
            delta_a = np.zeros_like(a)

            # CASE e < 0.001
            delta_e = np.zeros_like(e)
            delta_a = -k_a

            # CASE e>= 0.001
            I = e >= 0.001
            trunc_err_a = a[I]**2 * rho[I] * \
                np.exp(-z[I]) * iv(0, z[I]) * e[I]**6
            trunc_err_e = a[I] * rho[I] * np.exp(-z[I]) * iv(1, z[I]) * e[I]**6

            transform_e = e_T.T.dot(self.K_e) * I_T
            coef_e = np.array([transform_e[i, i] for i in range(N_f)])[I]

            transform_a = e_T.T.dot(self.K_a) * I_T
            coef_a = np.array([transform_a[i, i] for i in range(N_f)])[I]

            delta_e[I] = -k_e[I] * np.exp(-z[I]) * (coef_e + trunc_err_e)
            delta_a[I] = -k_a[I] * np.exp(-z[I]) * (coef_a + trunc_err_a)

            delta_e[np.isnan(delta_e)] = 0
            delta_a[np.isnan(delta_a)] = 0

            # Deorbit check
            J = h_p < 100*1e3
            delta_a[J] = 0
            delta_e[J] = 0

        ###############  J2 effects  ###############
        if self.perts['J2']:
            base = (3/2) * self.cb['J2'] * (radius**2/p**2) * n
            i = np.deg2rad(i)
            delta_omega = base * (2 - (5/2)*np.sin(i)**2)
            delta_Omega = -base * np.cos(i)
            delta_omega = np.rad2deg(delta_omega) % 360
            delta_Omega = np.rad2deg(delta_Omega) % 360

        return np.concatenate((delta_e, delta_a, delta_i, delta_Omega, delta_omega))

    # Performing a regular propagation, i.e. w/ perturbations
    def propagate_perturbations(self):

        # Initial states
        a0, e0, i0, Omega0, omega0 = self.states[-1, :5, :]
        y0 = np.concatenate((e0, a0, i0, Omega0, omega0))

        # Propagation time
        T_avg = np.mean(self.states[-1, 8, :])
        times = np.arange(self.tspan[0], self.tspan[-1], self.dt)
        output = integrate.solve_ivp(
            self.diffy_q, self.tspan, y0, t_eval=times)

        # Unpacking output (Need to drop first timestep as sudden introduction of drag causes discontinuities)
        N_f = len(self.A)
        de = output.y[0:N_f, 1:]
        da = output.y[N_f:2*N_f, 1:]
        di = output.y[2*N_f:3*N_f, 1:]
        dOmega = output.y[3*N_f:4*N_f, 1:]
        domega = output.y[4*N_f:, 1:]
        dnu = np.random.uniform(low=0., high=360., size=domega.shape)
        dp = da * (1 - de**2)

        # Results
        return de, da, di, dOmega, domega, dnu, dp

    # Performing a Keplerian propagation, i.e. w/o perturbations
    def propagate_orbit(self):

        times = np.arange(self.tspan[0], self.tspan[-1], self.dt)

        # Mean anomaly rate of change
        M_dt = sqrt(self.cb['mu']/self.states[0, :]**3)

        Nd = len(M_dt)
        Nt = len(times)

        # Mean anomaly over time
        M_t = np.deg2rad(self.states[5, :, None]) + \
            M_dt[:, None] * times[None, :]
        M_t = np.rad2deg(np.mod(M_t, 2*pi))

        # Eccentric anomaly over time. Note need to use E_t in rad, thus convert to deg after using it in
        # x1 and x2
        E_t = np.empty(shape=(Nd, Nt), dtype=np.float32)
        E_t = M2E(self.states[1], np.deg2rad(M_t))

        x1 = sqrt(1 + self.states[1, :])[:, None] * sin(E_t / 2)
        x2 = sqrt(1 - self.states[1, :])[:, None] * cos(E_t / 2)
        E_t = np.rad2deg(E_t)

        # True anomaly over time
        nu_t = (2*np.arctan2(x1, x2) % (2*pi))
        nu_t = np.rad2deg(nu_t).T

        n_times = nu_t.shape[0]
        states = np.empty(
            shape=(n_times, self.states.shape[0], self.states.shape[1]))

        for i in prange(n_times):
            state = self.states.copy()
            state[6, :] = nu_t[i, :]
            states[i] = state

        # Update internal states
        self.states = states


# Modified from OrbitalPy.utilities
@jit(parallel=True, fastmath=True)
def M2E(e_deb, M_t, tolerance=1e-14):
    # Convert mean anomaly to eccentric anomaly.
    # Implemented from [A Practical Method for Solving the Kepler Equation][1]
    # by Marc A. Murison from the U.S. Naval Observatory
    # [1]: http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
    n_deb = M_t.shape[0]
    n_times = M_t.shape[1]

    E_t = np.empty_like(M_t)

    for i in prange(n_deb):
        e = e_deb[i]
        for j in prange(n_times):
            M = M_t[i, j]

            MAX_ITERATIONS = 100
            Mnorm = np.mod(M, 2 * pi)
            E0 = M + (-1 / 2 * e ** 3 + e + (e ** 2 + 3 /
                                             2 * cos(M) * e ** 3) * cos(M)) * sin(M)
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
