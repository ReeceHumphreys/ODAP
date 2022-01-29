import numpy as np
from .utils import normalizeRadians

mu_earth = 3.986004418e14 # [m^3 s^-2]

class OrbitalElements():

    def __init__(self, satellite):
        semiMajorAxis  = self.meanMotion2SemiMajorAxis(float(satellite.mean_motion))
        elements = [satellite.inclination, satellite.raan, satellite.aop, satellite.mean_anomaly]
        inclination, raan, aop, ma = [np.deg2rad(float(element)) for element in elements]
        ea = self.meanAnomaly2EccentricAnomaly(ma, float(satellite.eccentricity))
        
        self.a = semiMajorAxis
        self.inclination = inclination
        self.eccentricity = satellite.eccentricity
        self.raan = raan
        self.aop = aop
        self.eccentricAnomaly = ea

    # Mean motion (n) from tle in rev / day
    def meanMotion2SemiMajorAxis(self, n):
        a = mu_earth**(1/3) / ((2 * n * np.pi) / (24 * 60 * 60))**(2/3)
        return a

    def meanAnomaly2EccentricAnomaly(self, ma, e):
        ea = ma + e * np.sin(ma)
        return normalizeRadians(self.newtonRaphson(ma, e, ea));
        
    def newtonRaphson(self, ma, e, ea):
        accuracy = 1e-16
        max_loop = 100
        term = 0
        current_loop = 0
        while (abs(term / max(ea, 1.0))) > accuracy and (current_loop < max_loop):
            term = self.kepE(ea, ma, e) / self.d_kepE(ea, e)
            ea = ea - term;
            current_loop+=1
        return ea

    def kepE(self, ma, e, ea):
        return (ea - e * np.sin(ea) - ma)
    
    def d_kepE(self, ea, e):
        return (1.0 - e * np.cos(ea))
