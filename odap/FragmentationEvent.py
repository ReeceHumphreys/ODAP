from .SimulationConfiguration import SimulationType


class FragmentationEvent():

    _input_mass = 0
    _output_mass = 0

    def n_fragments(self, L_c, S=1):
        return 6 * S * (L_c)**(-1.6)

    def __init__(self, config, sat):
        # Setting characteristic lengths
        self._minimum_characteristic_length = config.minimalCharacteristicLength
        self._maximum_characteristic_length = sat.compute_characteristic_length()

        # Explosion or Collision
        self._simulation_type = config.simulationType

        # Setting the mass before the fragmentation event
        self._input_mass = sat.mass

    def run(self):
        # Compute the number of fragments generate in the fragmentation event
        return self._compute_fragment_count()

    def _compute_fragment_count(self):
        if self._simulationType == SimulationType.explosion:
            return self.n_fragments(self._minimum_characteristic_length)
        else:
            print("Computing Count for Collision")

    def _compute_characteristic_length_distribution():
        pass
