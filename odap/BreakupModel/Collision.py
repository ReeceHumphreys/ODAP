import numpy as np

from .Breakup import Breakup
from .SimulationConfiguration import SatType


class Collision(Breakup):
    @property
    def lc_power_law_exponent(self):
        return -2.71

    @property
    def delta_velocity_offset(self):
        return [0.9, 2.9]

    @property
    def max_characteristic_length(self):
        return self._max_characteristic_length

    @property
    def sat_type(self):
        return self._sat_type

    @property
    def input_mass(self):
        return self._input_mass

    def fragment_count(self, satellites, min_characteristic_length):
        satellite_1 = satellites[0]
        satellite_2 = satellites[1]
        self._max_characteristic_length = [
            satellite_1.characteristic_length,
            satellite_2.characteristic_length,
        ]
        self._sat_type = SatType.soc

        if satellite_1.type == SatType.rb or satellite_2.type == SatType.rb:
            self._sat_type = SatType.rb

        # satellite_1 should be the bigger satellite
        if (
            satellite_2.characteristic_length
            > satellite_1.characteristic_length
        ):
            satellite_1, satellite_2 = satellite_2, satellite_1

        self._input_mass = satellite_1.mass + satellite_2.mass
        mass = 0

        # The Relative Collision Velocity
        delta_velocity = np.linalg.norm(
            satellite_1.velocity - satellite_2.velocity
        )

        catastrophic_ratio = (
            satellite_2.mass * delta_velocity * delta_velocity
        ) / (2.0 * satellite_1.mass * 1000.0)

        if catastrophic_ratio < 40.0:
            self._is_catastrophic = False
            mass = satellite_2.mass * delta_velocity / 1000.0
        else:
            self._is_catastrophic = True
            mass = satellite_1.mass + satellite_2.mass

        return int(0.1 * pow(mass, 0.75) * pow(min_characteristic_length, -1.71))
