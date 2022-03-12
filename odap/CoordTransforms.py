import numpy as np
from numba import njit as jit, prange
from numpy import cos, cross, sin, sqrt
from numpy.linalg import norm


"""
Converting from Keplerian to Cartesian
--------------------------------------
Helpful links:
    https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
    https://gitlab.eng.auburn.edu/evanhorn/orbital-mechanics/blob/a850737fcf4c43e295e79decf2a3a88acbbba451/Homework1/kepler.py

Notes: Below code is from Poliastro source, elements.py
"""

@jit
def coe2rv(k, p, ecc, inc, raan, argp, nu):
    r"""Converts from classical orbital to state vectors.
    Classical orbital elements are converted into position and velocity
    vectors by `rv_pqw` algorithm. A rotation matrix is applied to position
    and velocity vectors to get them expressed in terms of an IJK basis.
    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    p : float
        Semi-latus rectum or parameter (km).
    ecc : float
        Eccentricity.
    inc : float
        Inclination (rad).
    raan : float
        Longitude of ascending node, omega (rad).
    argp : float
        Argument of perigee (rad).
    nu : float
        True anomaly (rad).
    Returns
    -------
    r_ijk: numpy.ndarray
        Position vector in basis ijk.
    v_ijk: numpy.ndarray
        Velocity vector in basis ijk.
    """
    pqw = rv_pqw(k, p, ecc, nu)
    rm = coe_rotation_matrix(inc, raan, argp)

    ijk = pqw @ rm.T

    return ijk

@jit
def rotation_matrix(angle, axis):
    assert axis in (0, 1, 2)
    angle = np.asarray(angle)
    c = cos(angle)
    s = sin(angle)

    a1 = (axis + 1) % 3
    a2 = (axis + 2) % 3
    R = np.zeros(angle.shape + (3, 3))
    R[..., axis, axis] = 1.0
    R[..., a1, a1] = c
    R[..., a1, a2] = -s
    R[..., a2, a1] = s
    R[..., a2, a2] = c
    return R

@jit
def coe_rotation_matrix(inc, raan, argp):
    """Create a rotation matrix for coe transformation"""
    r = rotation_matrix(raan, 2)
    r = r @ rotation_matrix(inc, 0)
    r = r @ rotation_matrix(argp, 2)
    return r

@jit
def rv_pqw(k, p, ecc, nu):
    r"""Returns r and v vectors in perifocal frame.
    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    p : float
        Semi-latus rectum or parameter (km).
    ecc : float
        Eccentricity.
    nu : float
        True anomaly (rad).
    Returns
    -------
    r: numpy.ndarray
        Position. Dimension 3 vector
    v: numpy.ndarray
        Velocity. Dimension 3 vector
    """
    pqw = np.array([[cos(nu), sin(nu), 0], [-sin(nu), ecc + cos(nu), 0]]) * np.array(
        [[p / (1 + ecc * cos(nu))], [sqrt(k / p)]]
    )
    return pqw
