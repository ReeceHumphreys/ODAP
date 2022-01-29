from .Satellite import Satellite
from .SimulationConfiguration import SimulationType, SimulationConfiguration
class FragmentationEvent():

    _inputMass = 0
    _outputMass = 0

    def n_fragments(self, L_c, S = 1):
        return 6 * S * (L_c)**(-1.6)

    def __init__(self, config, sat):
        # Setting characteristic lengths
        self._minimumCharacteristicLength = config.minimalCharacteristicLength
        self._maximumCharacteristicLength = sat.computeCharacteristicLengthFromMass()

        # Explosion or Collision
        self._simulationType = config.simulationType

        # Setting the mass before the fragmentation event
        self._inputMass = sat.mass

    def run(self):
        # Compute the number of fragments generate in the fragmentation event
        return self._computeFragmentCount()

    def _computeFragmentCount(self):
        if self._simulationType == SimulationType.explosion:
            return self.n_fragments(self._minimumCharacteristicLength)
        else:
            print("Computing Count for Collision")




