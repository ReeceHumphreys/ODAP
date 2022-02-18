from .Breakup import Breakup


class Explosion(Breakup):
    @property
    def lc_power_law_exponent(self):
        return -2.6

    @property
    def delta_velocity_offset(self):
        return [0.2, 1.85]

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
        satellite = satellites[0]
        self._max_characteristic_length = satellite.characteristic_length
        self._sat_type = satellite.type
        self._input_mass = satellite.mass

        S = 1
        return int(6 * S * (min_characteristic_length) ** (-1.6))

