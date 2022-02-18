from abc import ABC, abstractmethod


class Breakup(ABC):
    @property
    @abstractmethod
    def lc_power_law_exponent(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def delta_velocity_offset(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def max_characteristic_length(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def sat_type(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def input_mass(self):
        raise NotImplementedError

    @abstractmethod
    def fragment_count(self, satellites, min_characteristic_length):
        raise NotImplementedError
