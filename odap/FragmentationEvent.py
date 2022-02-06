from .SimulationConfiguration import SimulationType
from .SimulationConfiguration import SatType
from .utils import power_law
import numpy as np


class FragmentationEvent():

    _input_mass = 0
    _output_mass = 0

    # TODO: This is just for explosions, will need to refactor when adding collisions
    _lc_power_law_exponent = -2.6

    # Sats is an array containing the satellites involved in the fragmentation event
    # Will contain one for explosions and two for collisions
    def __init__(self, config, sats):

        self.sats = sats

        # Explosion or Collision
        self._simulation_type = config.simulationType

        # Setting the mass before the fragmentation event
        self._input_mass = np.sum([sat.mass for sat in sats])

        # Categorization of satellite: RB, SC, or SOC
        self._sat_type = config.sat_type

        # Setting characteristic lengths
        self._min_characteristic_length = config.minimalCharacteristicLength
        self._max_characteristic_length = sats[0].characteristic_length

    def run(self):
        print("Started run")
        # Compute the number of fragments generate in the fragmentation event
        count = self._fragment_count(self._min_characteristic_length)

        # Location the explosion occured
        r = self.sats[0].position

        # Assigning debris type and location
        debris = np.empty((count, 3, 3))
        debris[:, 0] = SatType.deb.index
        debris[:, 1] = r

        # Computing L_c for each debris following powerlaw
        for i in range(count):
            debris[i: 2] = self._characteristic_length_distribution()

        return debris

    def _fragment_count(self, min_characteristic_length):
        if self._simulation_type == SimulationType.explosion:
            S = 1
            return int(6 * S * (min_characteristic_length)**(-1.6))
        else:
            print("Computing Count for Collision")

    def _characteristic_length_distribution(self):
        # Sampling a value from uniform distribution
        y = np.random.uniform(0.0, 1.0)
        return power_law(self._min_characteristic_length,
                         self._max_characteristic_length,
                         self._lc_power_law_exponent,
                         y)
