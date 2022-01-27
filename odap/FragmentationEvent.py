from .Satellite import Satellite
from .SimulationConfiguration import SimulationType, SimulationConfiguration
class FragmentationEvent():

    _inputMass = 0
    _outputMass = 0

    def n_fragments_dist(self, L_c, S = 1):
        return 6 * S * (L_c)**(-1.6)

    def __init__(self, config):
        self._setMinimalCharacteristicLength(config.minimalCharacteristicLength)
        self._setSimulationType(config.simulationType)

    def _setMinimalCharacteristicLength(self, minCharacteristicLength):
        self._minimalCharacteristicLength = minCharacteristicLength

    def _setSimulationType(self, simulationType):
        self._simulationType = simulationType

    def run(self):
        # Compute the number of fragments generate in the fragmentation event
        self._computeFragmentCount()

    def _computeFragmentCount(self):
        if self._simulationType == SimulationType.explosion:
            print("Computing Count for Explosion")
        else:
            print("Computing Count for Collision")




