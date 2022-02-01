import numpy as np
from .Parsing.TLEParser import TLEParser
from .OrbitalElements import OrbitalElements
from .CoordTransforms import coe2rv


class Satellite:

    # Mass in kg (TODO: Check this makes sense)
    def __init__(self, tle, mass=839.0):
        """
        Constructs all the necessary attributes for the satellite object.

        Parameters
        ----------
            tles : array
                an (3, ) NumPy array where each element corresponds to the line of a two-line element

        """

        line1_data, line2_data, line3_data = TLEParser().parse(tle)

        # Extracting data from the first line of the tle
        self.name = line1_data

        # Extracting data from the second line of the tle
        (self.num,
         self.classification,
         self.year,
         self.launch,
         self.piece,
         self.epoch_year,
         self.epoch_day,
         self.ballistic_coefficient,
         self.mean_motion_dotdot,
         self.bstar,
         self.ephemeris_type,
         self.element_number,
         self.datetime) = line2_data

        # Extracting data from the third line of the tle
        (self.inclination, self.raan, self.eccentricity, self.aop,
         self.mean_anomaly, self.mean_motion, self.epoch_rev) = line3_data

        # Setting mass
        self.mass = mass

    def get_orbital_elements(self):
        self.elements = OrbitalElements(self)
        mu_earth = 398600.4418  # [km^3 s^-2]
        e = self.elements.eccentricity
        p = (self.elements.a / 1e3) * (1 - e)
        i = self.elements.inclination
        raan = self.elements.raan
        aop = self.elements.aop
        nu = self.elements.true_anomaly
        r, v = coe2rv(mu_earth, p, e, i, raan, aop, nu)
        print(r, v)

    # Calculates the Characteristic Length assuming the mass is formed like a
    # sphere.
    def compute_characteristic_length(self):
        return ((6.0 * self.mass) / (92.937 * np.pi)) ** (1.0 / 2.26)
