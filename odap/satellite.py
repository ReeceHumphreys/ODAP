import numpy as np
from .CoordTransforms import coe2rv

# kg
DEFAULT_MASS = 839.0

# "m3 / (s2)"
mu_Earth = 3.986004418e14


class Satellite:

    # All satellites should have orbital elements.
    # Then the carteisan state is constructed if needed

    def __init__(self, tle, mass=DEFAULT_MASS):
        """
        Constructs all the necessary attributes for the satellite object from a TLE.

        Parameters
        ----------
            tles : TLE
            mass : Mass of the Satellite in kg

        """
        self.tle = tle
        self.mass = mass

        self.ecc = tle.ecc
        self.a = tle.a
        self.inc = tle.inc
        self.raan = tle.raan
        self.argp = tle.argp
        self.n = tle.n
        self.nu = tle.nu
        self.M = tle.M

        # Setting computed properties to None
        self.r = None
        self.v = None

    # def __init__(self, orbital_elements, mass=DEFAULT_MASS):
    #     """
    #     Constructs all the necessary attributes for the satellite object from a Orbital Elements.

    #     Parameters
    #     ----------
    #         tles : TLE
    #         mass : Mass of the Satellite in kg

    #     """
    #     # TODO: Implement later
    #     pass

    @property
    def cartesian_state(self):
        if self.r == None or self.v == None:
            p = self.a * (1 - self.ecc**2)
            r, v = coe2rv(mu_Earth, p, self.ecc, np.deg2rad(self.inc),
                          np.deg2rad(self.raan), np.deg2rad(self.argp), np.deg2rad(self.nu))
            self.r = r
            self.v = v
        return self.r, self.v

    # Calculates the Characteristic Length assuming the mass is formed like a
    # sphere.

    def compute_characteristic_length(self):
        return ((6.0 * self.mass) / (92.937 * np.pi)) ** (1.0 / 2.26)
