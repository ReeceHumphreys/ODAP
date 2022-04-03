import numpy as np
from numba import njit as jit, prange
from numpy import cos, cross, sin, sqrt
import sys

from .utils import E_to_nu, F_to_nu, norm


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


@jit(parallel=sys.maxsize > 2**31)
def coe2rv_many(k, p, ecc, inc, raan, argp, nu):
    n = nu.shape[0]
    rr = np.zeros((n, 3))
    vv = np.zeros((n, 3))

    for i in prange(n):
        rr[i, :], vv[i, :] = coe2rv(
            k, p[i], ecc[i], inc[i], raan[i], argp[i], nu[i]
        )

    return rr, vv


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
    pqw = np.array(
        [[cos(nu), sin(nu), 0], [-sin(nu), ecc + cos(nu), 0]]
    ) * np.array([[p / (1 + ecc * cos(nu))], [sqrt(k / p)]])
    return pqw


@jit
def rv2coe(k, r, v, tol=1e-8):
    r"""Converts from vectors to classical orbital elements.
    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2)
    r : numpy.ndarray
        Position vector (km)
    v : numpy.ndarray
        Velocity vector (km / s)
    tol : float, optional
        Tolerance for eccentricity and inclination checks, default to 1e-8
    Returns
    -------
    p : float
        Semi-latus rectum of parameter (km)
    ecc: float
        Eccentricity
    inc: float
        Inclination (rad)
    raan: float
        Right ascension of the ascending nod (rad)
    argp: float
        Argument of Perigee (rad)
    nu: float
        True Anomaly (rad)
    """

    h = cross(r, v)
    n = cross([0, 0, 1], h)
    e = ((v @ v - k / norm(r)) * r - (r @ v) * v) / k
    ecc = norm(e)
    p = (h @ h) / k
    inc = np.arccos(h[2] / norm(h))

    circular = ecc < tol
    equatorial = abs(inc) < tol

    if equatorial and not circular:
        raan = 0
        argp = np.arctan2(e[1], e[0]) % (2 * np.pi)  # Longitude of periapsis
        nu = np.arctan2((h @ cross(e, r)) / norm(h), r @ e)
    elif not equatorial and circular:
        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        argp = 0
        # Argument of latitude
        nu = np.arctan2((r @ cross(h, n)) / norm(h), r @ n)
    elif equatorial and circular:
        raan = 0
        argp = 0
        nu = np.arctan2(r[1], r[0]) % (2 * np.pi)  # True longitude
    else:
        a = p / (1 - (ecc**2))
        ka = k * a
        if a > 0:
            e_se = (r @ v) / sqrt(ka)
            e_ce = norm(r) * (v @ v) / k - 1
            nu = E_to_nu(np.arctan2(e_se, e_ce), ecc)
        else:
            e_sh = (r @ v) / sqrt(-ka)
            e_ch = norm(r) * (norm(v) ** 2) / k - 1
            nu = F_to_nu(np.log((e_ch + e_sh) / (e_ch - e_sh)) / 2, ecc)

        raan = np.arctan2(n[1], n[0]) % (2 * np.pi)
        px = r @ n
        py = (r @ cross(h, n)) / norm(h)
        argp = (np.arctan2(py, px) - nu) % (2 * np.pi)

    nu = (nu + np.pi) % (2 * np.pi) - np.pi
    return p, ecc, inc, raan, argp, nu


@jit(parallel=sys.maxsize > 2**31)
def rv2coe_many(mu, rs, vs, tol=1e-8):
    n = rs.shape[0]
    ps = np.zeros((n))
    eccs = np.zeros((n))
    incs = np.zeros((n))
    raans = np.zeros((n))
    argps = np.zeros((n))
    nus = np.zeros((n))
    for i in prange(n):
        ps[i], eccs[i], incs[i], raans[i], argps[i], nus[i] = rv2coe(
            mu, rs[i], vs[i], tol
        )
    return np.stack((ps, eccs, incs, raans, argps, nus), axis=1)
