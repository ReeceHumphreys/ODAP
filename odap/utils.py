import numpy as np
from numba import njit as jit, prange
from numpy import sqrt

"""
* Transform an y in [0;1] generated by an uniform distribution to an x which follows the
* properties of a power law distribution.
* @param x0 - the lower bound for the numbers (corresponds to the minimal L_c)
* @param x1 - the upper bound (correspond to the maximum of L_c of the two satellites or infinite if
*              there is no upper bound)
 * @param n - the exponent from the power law distribution, more precisely the exponent of the
*              probability density function (pdf)
* @param y - the value from the uniform distribution to transform
 * @return the transformed x following the power law distribution
*/
"""


def power_law(x0, x1, n, y):
    step = pow(x1, n + 1) - pow(x0, n + 1) * y + pow(x0, n + 1)
    return pow(step, 1 / (n + 1))


def _kepler_equation(E, M, ecc):
    return E_to_M(E, ecc) - M


def _kepler_equation_prime(E, M, ecc):
    return 1 - ecc * np.cos(E)


@jit
def norm(arr):
    return np.sqrt(arr @ arr)


@jit
def E_to_M(E, ecc):
    M = E - ecc * np.sin(E)
    return M


@jit
def E_to_nu(E, ecc):
    r"""True anomaly from eccentric anomaly.
    .. versionadded:: 0.4.0
    Parameters
    ----------
    E : float
        Eccentric anomaly in radians.
    ecc : float
        Eccentricity.
    Returns
    -------
    nu : float
        True anomaly, between -π and π radians.
    Warnings
    --------
    The true anomaly will be between -π and π radians,
    no matter the value of the eccentric anomaly.\
    """
    nu = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E / 2))
    return nu


@jit
def F_to_nu(F, ecc):
    r"""True anomaly from hyperbolic anomaly.
    Parameters
    ----------
    F : float
        Hyperbolic anomaly.
    ecc : float
        Eccentricity (>1).
    Returns
    -------
    nu : float
        True anomaly.
    """
    nu = 2 * np.arctan(np.sqrt((ecc + 1) / (ecc - 1)) * np.tanh(F / 2))
    return nu


def _M_to_nu(M, ecc):
    E = M_to_E(M, ecc)
    nu = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E / 2))
    return nu


def M_to_E(M, ecc):
    if -np.pi < M < 0 or np.pi < M:
        E0 = M - ecc
    else:
        E0 = M + ecc
    E = _newton_elliptic(E0, args=(M, ecc))
    return E


@jit
def Nu_to_E(nu, ecc):
    r"""Eccentric anomaly from true anomaly.
    .. versionadded:: 0.4.0
    Parameters
    ----------
    nu : float
        True anomaly in radians.
    ecc : float
        Eccentricity.
    Returns
    -------
    E : float
        Eccentric anomaly, betweenE -π and π radians.
    Warnings
    --------
    The eccentric anomaly will be between -π and π radians,
    no matter the value of the true anomaly.
    Notes
    -----
    The implementation uses the half-angle formula from [3]_:
    .. math::
        E = 2 \arctan \left ( \sqrt{\frac{1 - e}{1 + e}} \tan{\frac{\nu}{2}} \right)
        \in (-\pi, \pi]
    """
    E = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(nu / 2))
    return E


def newton_factory(func, fprime):
    def jit_newton_wrapper(x0, args=(), tol=1.48e-08, maxiter=50):
        p0 = float(x0)
        for _ in range(maxiter):
            fval = func(p0, *args)
            fder = fprime(p0, *args)
            newton_step = fval / fder
            p = p0 - newton_step
            if abs(p - p0) < tol:
                return p
            p0 = p

        return np.nan

    return jit_newton_wrapper


_newton_elliptic = newton_factory(_kepler_equation, _kepler_equation_prime)


def circle_area(characteristic_length):
    radius = characteristic_length / 2
    return np.pi * (radius**2)
