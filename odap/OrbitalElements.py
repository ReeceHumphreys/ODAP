import numpy as np
from .utils import normalize_radians

mu_earth = 3.986004418e14  # [m^3 s^-2]


class OrbitalElements():

    def __init__(self, satellite):
        e = float(satellite.eccentricity)
        semiMajorAxis = self.semi_major_axis(float(satellite.mean_motion))
        elements = [satellite.inclination, satellite.raan,
                    satellite.aop, satellite.mean_anomaly]
        inclination, raan, aop, ma = [np.deg2rad(
            float(element)) for element in elements]
        ea = self.eccentric_anomaly(ma, e)

        self.a = semiMajorAxis
        self.inclination = inclination
        self.eccentricity = e
        self.raan = raan
        self.aop = aop
        self.eccentricAnomaly = ea
        self.true_anomaly = self._true_anomaly()

    # Mean motion (n) from tle in rev / day
    def semi_major_axis(self, n):
        a = mu_earth**(1 / 3) / ((2 * n * np.pi) / (24 * 60 * 60))**(2 / 3)
        return a

    def eccentric_anomaly(self, ma, e):
        ea = ma + e * np.sin(ma)
        return normalize_radians(self.newton_raphson(ma, e, ea))

    def _true_anomaly(self):
        e = self.eccentricity
        ea = self.eccentricAnomaly
        return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(ea / 2))

    def newton_raphson(self, ma, e, ea):
        accuracy = 1e-16
        max_loop = 100
        term = 0
        current_loop = 0
        while (abs(term / max(ea, 1.0))
               ) > accuracy and (current_loop < max_loop):
            term = self.kep_E(ea, ma, e) / self.d_kep_E(ea, e)
            ea = ea - term
            current_loop += 1
        return ea

    def kep_E(self, ma, e, ea):
        return (ea - e * np.sin(ea) - ma)

    def d_kep_E(self, ea, e):
        return (1.0 - e * np.cos(ea))
